"""视觉路由：羊群计数（图片 / 视频流计数与越线统计）+ 塔基图像几何校正。"""
from __future__ import annotations

import asyncio
import base64
import json
import shutil
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

from app.config import settings
from app.data import repository
from app.data.db import connect
from app.schemas import CarryingCapacityRequest, err, ok
from app.vision.counter import count_sheep
from app.vision.detector import SheepDetector, draw_annotations
from app.vision.georectify import orthorectify, reprojection_error
from app.vision.grassland import carrying_capacity, coverage
from app.vision.line_crossing import LineCrossingCounter, line_from_ratio
from app.vision.track import ObjectTracker, TrackedDetection
from app.vision.video import process_frames
from app.vision.zoning import segment_plots

router = APIRouter(prefix="/api/vision", tags=["vision"])

# 模块级检测器（惰性加载模型）；测试中可被替换。
detector = SheepDetector()

# 视频作业串行化（单机 GPU 推理；同时上传多个视频时排队而非抢占显存）
_video_lock = threading.Lock()


def decode_image(data: bytes) -> np.ndarray:
    """将上传字节解码为 BGR 图像。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像文件")
    return img


def encode_image(
    image: np.ndarray, max_width: int = 1600, quality: int = 85, max_height: int | None = None
) -> str:
    """将图像编码为 JPEG 并返回 base64 字符串。

    限宽 1600px + 质量 85，控制响应体积（大图 base64 从 ~9MB 降至几百 KB 级）；
    ``max_height`` 额外限制高度（竖屏视频预览帧按高度缩放，避免长边过大）。
    """
    h, w = image.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / w
    if max_height and h * scale > max_height:
        scale = max_height / h
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    ok_flag, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok_flag:
        raise ValueError("无法编码标注图像")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@router.post("/count")
async def count_endpoint(file: UploadFile = File(...)):
    """上传图片 → 羊群检测 + 计数 + 标注图（base64）。"""
    try:
        img = decode_image(await file.read())
    except ValueError as e:
        return err(400, str(e))

    detections = detector.detect(img)
    result = count_sheep(detections)
    annotated = draw_annotations(img, detections)
    result["annotated_image"] = encode_image(annotated)

    conn = connect()
    try:
        repository.insert_count_record(
            conn,
            region=None,
            count=result["count"],
            per_class=result["per_class"],
            annotated_image=result["annotated_image"],
        )
        conn.commit()
    finally:
        conn.close()

    return ok(result)


@router.post("/coverage")
async def coverage_endpoint(file: UploadFile = File(...)):
    """上传草场图片 → 草地覆盖度 + 平均绿度。"""
    try:
        img = decode_image(await file.read())
    except ValueError as e:
        return err(400, str(e))

    result = coverage(img)

    conn = connect()
    try:
        repository.insert_grassland_metric(
            conn,
            region=None,
            coverage=result["coverage"],
            greenness=result["greenness"],
        )
        conn.commit()
    finally:
        conn.close()

    return ok(result)


@router.post("/orthorectify")
async def orthorectify_endpoint(
    file: UploadFile = File(...),
    points: str = Form(..., description="4 个地面参考点，JSON 数组 [[x,y],…]；像素或归一化(0–1)坐标"),
    dst_points: str | None = Form(None, description="正射平面目标点（可选，缺省按源四边形生成矩形）"),
):
    """塔基斜拍图 → 四点标定正射校正，并返回**校正前后**的覆盖度/绿度对比。

    对应覃星力等《AI 驱动的塔基农情遥感监测新技术》关键技术 1 的轻量实现：
    论文用云台参数 + 四元数模型自动配准，本项目用四点单应变换（无云台参数时的
    标准做法）。返回单应矩阵与重投影误差，便于报告中量化校正质量。
    """
    try:
        img = decode_image(await file.read())
    except ValueError as e:
        return err(400, str(e))

    try:
        src = json.loads(points)
        dst = json.loads(dst_points) if dst_points else None
    except json.JSONDecodeError as e:
        return err(400, f"参考点 JSON 解析失败：{e}")

    try:
        result = orthorectify(img, src, dst)
    except ValueError as e:
        return err(400, str(e))

    before = coverage(img)
    after = coverage(result["image"])
    return ok(
        {
            "corrected_image": encode_image(result["image"], max_width=1280, quality=82),
            "matrix": result["matrix"],
            "src_points": result["src_points"],
            "dst_points": result["dst_points"],
            "size": list(result["size"]),
            "reprojection_error": round(
                reprojection_error(result["matrix"], result["src_points"], result["dst_points"]), 4
            ),
            "coverage_before": round(before["coverage"], 4),
            "coverage_after": round(after["coverage"], 4),
            "greenness_before": round(before["greenness"], 4),
            "greenness_after": round(after["greenness"], 4),
        }
    )


@router.post("/segment")
async def segment_endpoint(
    file: UploadFile = File(...),
    n_plots: int = Form(3, ge=1, le=9),
):
    """草场图 → 自动划分管理分区（植被掩膜连通域 + 空间聚类）+ 每区覆盖度 + 叠加图。

    对应覃星力等《AI 驱动的塔基农情遥感监测新技术》关键技术 2（空间信息引导的
    SAM 地块分割）的轻量实现：无地理参考下的纯图像连通域聚类（报告中如实标注）。
    大图先降采样到 1280px（分区不需要原始分辨率）。
    """
    try:
        img = decode_image(await file.read())
    except ValueError as e:
        return err(400, str(e))
    h, w = img.shape[:2]
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        img = cv2.resize(
            img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    try:
        result = segment_plots(img, n_plots=n_plots)
    except ValueError as e:
        return err(400, str(e))

    payload = {k: v for k, v in result.items() if k != "image"}
    payload["annotated_image"] = (
        encode_image(result["image"], max_width=1280, quality=82)
        if result["image"] is not None
        else None
    )
    return ok(payload)


@router.post("/carrying-capacity")
async def carrying_capacity_endpoint(req: CarryingCapacityRequest):
    """参数 → 载畜量（NY/T 635 公式）+ 轮牧建议。"""
    result = carrying_capacity(
        yield_per_area=req.yield_per_area,
        area=req.area,
        utilization=req.utilization,
        grazing_days=req.grazing_days,
        daily_intake=req.daily_intake,
    )

    conn = connect()
    try:
        repository.insert_carrying_result(
            conn,
            params=req.model_dump(),
            recommended_stocking=result["recommended_stocking"],
            rotation_plan=result["rotation_plan"],
        )
        conn.commit()
    finally:
        conn.close()

    return ok(result)


# ---------------------------------------------------------------------------
# 阶段 6：视频流计数与越线统计
# ---------------------------------------------------------------------------

def draw_tracked_frame(
    image: np.ndarray,
    tracked: list[TrackedDetection],
    line: LineCrossingCounter | None = None,
) -> np.ndarray:
    """在帧上绘制跟踪框（含 ID）+ 越线警戒线，返回新图（不修改原图）。"""
    out = image.copy()
    for item in tracked:
        x1, y1, x2, y2 = (int(v) for v in item.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"#{item.track_id} {item.confidence:.2f}",
            (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    if line is not None:
        sx, sy = (int(v) for v in line.start)
        ex, ey = (int(v) for v in line.end)
        cv2.line(out, (sx, sy), (ex, ey), (0, 190, 255), 3)
    return out


def _iter_video_frames(cap: cv2.VideoCapture, max_frames: int):
    """逐帧读取（独立计数器兜底：CAP_PROP_FRAME_COUNT 可能为 0 或造假）。"""
    count = 0
    while count < max_frames:
        ok_frame, frame = cap.read()
        if not ok_frame:
            break
        count += 1
        yield frame


def _summary_payload(summary, line: LineCrossingCounter, fps: float, width: int, height: int) -> dict:
    return {
        "total_frames": summary.total_frames,
        "processed_frames": summary.processed_frames,
        "stride": summary.stride,
        "fps": round(float(fps or 0.0), 2),
        "width": int(width),
        "height": int(height),
        "max_simultaneous": summary.max_simultaneous,
        "unique_count": summary.unique_count,
        "inbound": summary.inbound,
        "outbound": summary.outbound,
        "net": summary.net,
        "per_class": summary.per_class,
        "line": line.line(),
        "frames": [
            {
                "frame_index": item.frame_index,
                "count": item.count,
                "inbound": item.inbound,
                "outbound": item.outbound,
                "net": item.net,
                "preview_jpeg": item.preview_jpeg,
            }
            for item in summary.frames
        ],
    }


def _process_video(cap: cv2.VideoCapture, stride: int) -> dict:
    """同步处理整段视频（在线程池中执行，避免阻塞事件循环）。"""
    with _video_lock:  # 串行化：单机推理，避免多请求抢占
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fallback_fps = fps if fps > 0 else 30.0
        max_frames = max(1, int(fallback_fps * settings.video_max_seconds) + 5)

        start, end = line_from_ratio(
            width or 1,
            height or 1,
            settings.line_x1,
            settings.line_y1,
            settings.line_x2,
            settings.line_y2,
        )
        line = LineCrossingCounter(start, end)

        def encode(frame: np.ndarray, tracked: list[TrackedDetection]) -> str:
            annotated = draw_tracked_frame(frame, tracked, line)
            # 预览帧限宽/限高（560px）：32 张约 1–2MB，前端播放不吃力（响应体积可控）
            return encode_image(annotated, max_width=560, quality=62, max_height=560)

        summary = process_frames(
            _iter_video_frames(cap, max_frames=max_frames),
            detector.detect,
            encode_fn=encode,
            tracker=ObjectTracker(frame_rate=fallback_fps),
            line=line,
            stride=stride,
            max_previews=settings.video_max_previews,
            total_frames=declared or None,
        )
        return _summary_payload(summary, line, fps, width, height)


def _looks_like_image(path: Path) -> bool:
    """文件头判断是否图片（OpenCV 可能把图片当单帧视频打开，需显式拦截）。"""
    try:
        head = path.read_bytes()[:12]
    except OSError:
        return False
    return (
        head.startswith(b"\x89PNG\r\n\x1a\n")  # PNG
        or head.startswith(b"\xff\xd8\xff")  # JPEG
        or head.startswith(b"GIF8")  # GIF
        or head.startswith(b"BM")  # BMP
        or head.startswith(b"RIFF") and head[8:12] == b"WEBP"  # WebP
    )


@router.post("/count-video")
async def count_video_endpoint(file: UploadFile = File(...), stride: int = Form(0)):
    """上传视频 → 逐帧检测 + 跟踪 + 越线统计 → 标注帧序列（base64 JPEG）+ 三条口径汇总。

    - 体积上限查 ``UploadFile.size``（不依赖 413）；时长上限按 fps × 帧数估算；
    - 落盘用流式拷贝（不整段进内存），``finally`` 关句柄后再删临时文件（Windows 句柄）；
    - 无法解码（如传 PNG）→ 400；作业串行化，避免并发抢占 GPU。
    """
    size = getattr(file, "size", None)
    if size is not None and size > settings.video_max_bytes:
        return err(
            400,
            f"视频过大：{size / 1048576:.1f} MB，上限 {settings.video_max_bytes / 1048576:.0f} MB",
        )

    workdir = Path(tempfile.mkdtemp(prefix="muyitong_video_"))
    tmp_path = workdir / "upload_video"
    try:
        with tmp_path.open("wb") as handle:
            await asyncio.to_thread(shutil.copyfileobj, file.file, handle)
    except Exception as e:  # 落盘失败
        shutil.rmtree(workdir, ignore_errors=True)
        return err(500, f"视频落盘失败：{e}")
    finally:
        await file.close()

    if _looks_like_image(tmp_path):
        shutil.rmtree(workdir, ignore_errors=True)
        return err(400, "该文件是图片而非视频（图片计数请使用 /api/vision/count）")

    cap = cv2.VideoCapture(str(tmp_path))
    try:
        if not cap.isOpened():
            return err(400, "无法打开视频文件（格式不支持或已损坏）")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps > 0 and declared > 0 and declared / fps > settings.video_max_seconds:
            return err(
                400,
                f"视频时长超限：{declared / fps:.1f}s，上限 {settings.video_max_seconds:.0f}s",
            )
        step = max(1, int(stride) if stride else settings.video_stride)
        payload = await asyncio.to_thread(_process_video, cap, step)
    except Exception as e:
        return err(500, f"视频处理失败：{e}")
    finally:
        cap.release()  # 先释放句柄，再删临时文件（Windows 上句柄占用会删不掉）
        shutil.rmtree(workdir, ignore_errors=True)

    return ok(payload)
