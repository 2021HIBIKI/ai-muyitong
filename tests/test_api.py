"""阶段 0.4 / 1.3：API 入口测试 —— 健康检查与羊群计数接口。"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from app.vision.detector import Detection


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["status"] == "ok"
    assert body["message"] == "ok"


def test_health_unified_envelope(client):
    """统一响应结构 {code, data, message}。"""
    body = client.get("/api/health").json()
    assert set(body.keys()) == {"code", "data", "message"}


class _MockDetector:
    def detect(self, img):
        return [
            Detection((0, 0, 10, 10), 18, "sheep", 0.9),
            Detection((10, 10, 20, 20), 18, "sheep", 0.8),
        ]


def _png_bytes():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:] = (0, 128, 0)
    ok_flag, buf = cv2.imencode(".png", img)
    assert ok_flag
    return buf.tobytes()


def test_vision_count(client, monkeypatch):
    """上传合成图返回 200 且含 count 与标注图（mock 检测器）。"""
    from app.routers import vision as vision_router

    monkeypatch.setattr(vision_router, "detector", _MockDetector())
    resp = client.post(
        "/api/vision/count",
        files={"file": ("sheep.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["count"] == 2
    assert body["data"]["per_class"] == {"sheep": 2}
    assert body["data"]["annotated_image"]  # base64 非空


def test_vision_coverage(client):
    """上传纯绿合成图返回 200 且 coverage 接近 1。"""
    resp = client.post(
        "/api/vision/coverage",
        files={"file": ("grass.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert 0 <= body["data"]["coverage"] <= 1
    assert body["data"]["coverage"] > 0.9  # 纯绿图
    assert "greenness" in body["data"]


def _tilted_tower_png():
    """合成一张「塔基斜拍」草地图（左半植被 / 右半裸土，经已知透视变换）→ (PNG 字节, 4 角点)。"""
    import cv2
    import numpy as np

    h, w = 240, 320
    flat = np.zeros((h, w, 3), dtype=np.uint8)
    flat[:, : w // 2] = (0, 165, 0)  # BGR 植被
    flat[:, w // 2 :] = (120, 120, 130)  # 低饱和裸土 → 判为非植被
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = np.array([[30, 40], [300, 12], [318, 210], [14, 190]], dtype=np.float32)
    tilted = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(src, dst), (340, 240))
    ok_flag, buf = cv2.imencode(".png", tilted)
    assert ok_flag
    return buf.tobytes(), dst.tolist()


def test_vision_orthorectify(client):
    """四点标定正射校正：返回校正图、3×3 单应矩阵、重投影误差与校正前后覆盖度对比。"""
    import json

    png, corners = _tilted_tower_png()
    flat = [[0, 0], [319, 0], [319, 239], [0, 239]]  # 正射平面（原始绘图坐标系）
    resp = client.post(
        "/api/vision/orthorectify",
        files={"file": ("tower.png", png, "image/png")},
        data={"points": json.dumps(corners), "dst_points": json.dumps(flat)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["corrected_image"]  # 校正图 base64 非空
    assert len(data["matrix"]) == 3 and len(data["matrix"][0]) == 3
    assert data["reprojection_error"] < 0.5  # 精确对应 → 重投影误差≈0
    assert data["size"] == [320, 240]
    # 校正后恢复「左植被 / 右裸土」→ 覆盖度接近真值 0.5，且比校正前更接近
    assert 0.42 <= data["coverage_after"] <= 0.58
    assert abs(data["coverage_after"] - 0.5) <= abs(data["coverage_before"] - 0.5) + 1e-6
    assert 0 <= data["greenness_after"] <= 1 and 0 <= data["greenness_before"] <= 1

    # 缺省目标点：按源四边形边长生成保持长宽比的正射矩形
    default = client.post(
        "/api/vision/orthorectify",
        files={"file": ("tower.png", png, "image/png")},
        data={"points": json.dumps(corners)},
    ).json()["data"]
    w0, h0 = default["size"]
    assert abs(w0 / h0 - 305 / 199) < 0.05  # 与地面四边形实际长宽比一致


def test_vision_orthorectify_bad_input(client):
    """点数不足 / JSON 非法 → 400（不抛 500）。"""
    png, corners = _tilted_tower_png()
    resp = client.post(
        "/api/vision/orthorectify",
        files={"file": ("tower.png", png, "image/png")},
        data={"points": "[[0,0],[10,0],[10,10]]"},
    )
    assert resp.status_code == 200 and resp.json()["code"] == 400

    resp = client.post(
        "/api/vision/orthorectify",
        files={"file": ("tower.png", png, "image/png")},
        data={"points": "not-json"},
    )
    assert resp.status_code == 200 and resp.json()["code"] == 400

    # 退化四边形（四点共线）也应 400
    resp = client.post(
        "/api/vision/orthorectify",
        files={"file": ("tower.png", png, "image/png")},
        data={"points": "[[0,0],[100,0],[200,0],[300,0]]"},
    )
    assert resp.status_code == 200 and resp.json()["code"] == 400


def _zoned_png_bytes():
    """合成草场图：三条植被带（左/中/右）夹裸地 → 分区自动分割的测试输入。"""
    import cv2
    import numpy as np

    img = np.full((200, 300, 3), (120, 120, 130), dtype=np.uint8)
    for x0, x1 in ((20, 90), (120, 190), (230, 280)):
        img[:, x0:x1] = (0, 165, 0)
    ok_flag, buf = cv2.imencode(".png", img)
    assert ok_flag
    return buf.tobytes()


def test_vision_segment(client):
    """自动分区：三条植被带 → 3 个分区，编号 A/B/C 按位置排序，覆盖度接近 1。"""
    resp = client.post(
        "/api/vision/segment",
        files={"file": ("grass.png", _zoned_png_bytes(), "image/png")},
        data={"n_plots": "3"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["annotated_image"]  # 叠加图 base64 非空
    assert data["n_components"] >= 3
    assert 0.55 < data["mask_ratio"] < 0.70  # 三条植被带（70+70+50px）/ 300 宽 ≈ 0.63
    plots = data["plots"]
    assert len(plots) == 3
    assert [p["code"] for p in plots] == ["A", "B", "C"]
    xs = [p["centroid"][0] for p in plots]
    assert xs == sorted(xs)
    for p in plots:
        assert p["coverage"] > 0.9  # 分区框内几乎全植被
        assert len(p["bbox"]) == 4 and p["area"] > 0


def test_vision_segment_rejects_bad_n_plots(client):
    resp = client.post(
        "/api/vision/segment",
        files={"file": ("grass.png", _zoned_png_bytes(), "image/png")},
        data={"n_plots": "0"},
    )
    assert resp.status_code == 422  # Form 参数校验：n_plots >= 1


def test_vision_carrying_capacity(client):
    """参数调用载畜量接口，手算一致。"""
    resp = client.post(
        "/api/vision/carrying-capacity",
        json={
            "yield_per_area": 1000,
            "area": 100,
            "utilization": 0.5,
            "grazing_days": 100,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["recommended_stocking"] == 250.0
    assert body["data"]["rotation_plan"]


def test_vision_carrying_capacity_validation(client):
    """非法参数（负产量）应被 422 拒绝。"""
    resp = client.post(
        "/api/vision/carrying-capacity",
        json={
            "yield_per_area": -1,
            "area": 100,
            "utilization": 0.5,
            "grazing_days": 100,
        },
    )
    assert resp.status_code == 422


def test_agent_chat(client, monkeypatch):
    """mock 编排器返回决策 JSON。"""
    from app.routers import agent as agent_router
    from app.agent.orchestrator import DecisionJSON

    class _FakeOrchestrator:
        last_tool_calls = ["grassland_coverage"]

        def run(self, question):
            return DecisionJSON(
                alert_level="normal",
                recommended_actions=["正常放牧"],
                recommended_stocking=200.0,
                rotation_plan="轮牧",
                evidence=["草地覆盖度 0.80"],
            )

    monkeypatch.setattr(agent_router, "orchestrator", _FakeOrchestrator())
    resp = client.post("/api/agent/chat", json={"question": "东区草场够吗？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["alert_level"] == "normal"
    assert body["data"]["recommended_stocking"] == 200.0
    assert body["data"]["evidence"] == ["草地覆盖度 0.80"]


def test_agent_chat_validation(client):
    """空问题应被 422 拒绝。"""
    resp = client.post("/api/agent/chat", json={"question": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 阶段 4.2：大屏数据接口
# ---------------------------------------------------------------------------

def test_dashboard_overview(client):
    """启动播种后概览返回有效 KPI（草场逐场返回，无跨草场平均）。"""
    resp = client.get("/api/dashboard/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["sheep_count"] >= 0
    assert "coverage" not in data  # 跨草场没有「一个覆盖度」
    assert data["pastures"] and all("coverage" in p for p in data["pastures"])
    assert data["alert_level"] in {"normal", "warning", "danger"}
    assert data["recommended_stocking"] >= 0
    assert data["temperature"] is None or 0 <= data["temperature"] <= 45
    assert data["humidity"] is None or 0 <= data["humidity"] <= 100


def test_dashboard_overview_empty_db(client):
    """业务表清空后接口返回零值兜底而非报错。"""
    from app.data.db import connect

    conn = connect()
    try:
        for t in ("environment_records", "grassland_metrics", "count_records", "carrying_results"):
            conn.execute(f"DELETE FROM {t}")
        conn.commit()
    finally:
        conn.close()

    resp = client.get("/api/dashboard/overview")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["sheep_count"] == 0
    assert data["fallback"]["coverage"] is None  # 单点兜底口径，无数据即为 None
    assert data["alert_level"] is None  # 无数据不臆断等级
    assert data["temperature"] is None


def test_dashboard_overview_pastures_independent(client):
    """概览 KPI 各草场独立：覆盖度逐场返回；等级取最严重；载畜量取各场之和（可加量）。"""
    ov = client.get("/api/dashboard/overview").json()["data"]
    pastures = client.get("/api/pastures").json()["data"]["pastures"]
    analyzed = [p for p in pastures if p["coverage"] is not None]
    assert analyzed, "播种后应存在已分析草场"

    by_code = {p["code"]: p for p in ov["pastures"]}
    for p in pastures:
        assert by_code[p["code"]]["coverage"] == p["coverage"]  # 与草场清单同口径，逐场独立
        assert by_code[p["code"]]["alert_level"] == p["alert_level"]
    assert ov["recommended_stocking"] == round(
        sum(p["recommended_stocking"] for p in analyzed), 2
    )
    assert ov["pasture_count"] == len(pastures)
    assert ov["analyzed_pasture_count"] == len(analyzed)
    # 告警等级 = 各草场中最严重（不是覆盖度均值判定）
    severity = {"normal": 0, "warning": 1, "danger": 2}
    expected_level = max((p["alert_level"] for p in analyzed), key=lambda lv: severity[lv])
    assert ov["alert_level"] == expected_level
    worst = [p for p in analyzed if p["alert_level"] == expected_level]
    assert ov["alert_pasture"]["code"] in {p["code"] for p in worst}
    assert ov["alert_pasture"]["coverage"] == by_code[ov["alert_pasture"]["code"]]["coverage"]


def test_dashboard_overview_danger_pasture_zero_stocking(client):
    """危险等级草场建议载畜量为 0（禁牧）：理论值非 0，被告警系数折减为 0。"""
    pastures = client.get("/api/pastures").json()["data"]["pastures"]
    danger = [p for p in pastures if p["alert_level"] == "danger"]
    assert danger, "播种数据应包含危险草场（北坡轮休草场）"
    for p in danger:
        assert p["theoretical"] > 0  # 理论载畜量按草场面积算出
        assert p["factor"] == 0.0  # 危险等级禁牧
        assert p["recommended_stocking"] == 0.0


def test_dashboard_overview_danger_forces_zero(client):
    """无草场时回退到单条记录口径：危险覆盖度下建议载畜量必然为 0（禁牧）。"""
    from app.data import repository
    from app.data.db import connect

    conn = connect()
    try:
        for table in ("pastures", "pasture_images", "grassland_metrics"):
            conn.execute(f"DELETE FROM {table}")
        repository.insert_grassland_metric(conn, region="东区", coverage=0.20, greenness=0.05)
        repository.insert_carrying_result(
            conn, params={}, recommended_stocking=666.67, rotation_plan="划分 6 个轮牧小区"
        )
        conn.commit()
    finally:
        conn.close()

    d = client.get("/api/dashboard/overview").json()["data"]
    assert d["alert_level"] == "danger"
    assert d["recommended_stocking"] == 0.0
    assert d["fallback"]["rotation_plan"].startswith("【建议禁牧】")
    assert d["fallback"]["coverage"] == 0.20


def test_dashboard_environment(client):
    resp = client.get("/api/dashboard/environment?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    records = body["data"]["records"]
    assert 0 < len(records) <= 10
    assert {"created_at", "temperature", "humidity"} <= set(records[0])


def test_dashboard_trends(client):
    resp = client.get("/api/dashboard/trends?days=7")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    series = body["data"]["pastures"]
    assert series, "播种后应有草场序列"
    # 各草场独立：每个草场一条按日序列（当日该草场记录均值），无跨草场平均
    for pasture in series:
        assert {"id", "code", "name", "coverage"} <= set(pasture)
        assert len(pasture["coverage"]) == 7
        for point in pasture["coverage"]:
            assert set(point) == {"day", "coverage", "greenness", "n"}
            assert 0 <= point["coverage"] <= 1
            assert point["n"] >= 1
        days = [p["day"] for p in pasture["coverage"]]
        assert days == sorted(days)
    assert len(body["data"]["sheep"]) == 7
    # 瘦身字段：不下发标注图等大字段（响应体积可控）
    assert set(body["data"]["sheep"][0]) == {"created_at", "region", "count", "per_class"}


def test_dashboard_trends_daily_mean_matches_rows(client):
    """趋势的当日覆盖度 = 该草场当日全部记录的平均值（草场之间彼此独立）。"""
    from app.data.db import connect

    series = client.get("/api/dashboard/trends?days=30").json()["data"]["pastures"]
    assert series
    conn = connect()
    try:
        for pasture in series:
            row = conn.execute(
                "SELECT substr(created_at, 1, 10) AS day, AVG(coverage) AS cov, COUNT(*) AS n"
                " FROM grassland_metrics WHERE pasture_id = ?"
                " GROUP BY day ORDER BY day DESC LIMIT 1",
                (pasture["id"],),
            ).fetchone()
            last = pasture["coverage"][-1]
            assert last["day"] == row["day"]
            assert last["n"] == row["n"]
            assert abs(last["coverage"] - round(row["cov"], 4)) < 1e-6
    finally:
        conn.close()


def test_dashboard_decision_flow(client):
    from app.data import repository
    from app.data.db import connect

    conn = connect()
    try:
        repository.insert_decision_log(
            conn,
            question="东区草场够吗？",
            tool_calls=["grassland_coverage"],
            decision_json={
                "alert_level": "warning",
                "recommended_actions": ["轮牧"],
                "recommended_stocking": 100.0,
            },
        )
        conn.commit()
    finally:
        conn.close()

    resp = client.get("/api/dashboard/decision-flow?limit=5")
    assert resp.status_code == 200
    logs = resp.json()["data"]["logs"]
    # 最新一条即刚写入的记录（播种的演示记录在其之前）
    assert logs[-1]["question"] == "东区草场够吗？"
    assert logs[-1]["alert_level"] == "warning"
    assert logs[-1]["tool_calls"] == ["grassland_coverage"]


def test_dashboard_decision_flow_empty(client):
    from app.data.db import connect

    conn = connect()
    try:
        conn.execute("DELETE FROM decision_logs")  # 清掉播种的演示记录
        conn.commit()
    finally:
        conn.close()

    resp = client.get("/api/dashboard/decision-flow")
    assert resp.status_code == 200
    assert resp.json()["data"]["logs"] == []


# ---------------------------------------------------------------------------
# 阶段 4.3：数据大屏静态页
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 阶段 6.4：视频计数与越线统计接口
# ---------------------------------------------------------------------------

def _avi_bytes(frames: int = 12, size=(64, 64), fps: float = 10.0) -> bytes:
    """合成小视频（MJPG/AVI）：白点逐帧移动。"""
    import os
    import tempfile

    path = tempfile.mktemp(suffix=".avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    try:
        for i in range(frames):
            img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            cv2.circle(img, (8 + i * 3, 32), 6, (255, 255, 255), -1)
            writer.write(img)
    finally:
        writer.release()
    with open(path, "rb") as handle:
        data = handle.read()
    os.unlink(path)
    return data


class _MockVideoDetector:
    """固定返回一个检测框（合成视频里没有真羊）。"""

    def detect(self, img, max_det=None):
        return [Detection((20.0, 20.0, 60.0, 60.0), 18, "sheep", 0.9)]


def test_count_video_endpoint_returns_full_payload(client, monkeypatch):
    from app.routers import vision as vision_router

    monkeypatch.setattr(vision_router, "detector", _MockVideoDetector())
    resp = client.post(
        "/api/vision/count-video",
        files={"file": ("v.avi", _avi_bytes(), "video/x-msvideo")},
        data={"stride": "1"},  # 显式逐帧（默认步长见配置 video_stride）
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    data = body["data"]
    for key in (
        "total_frames", "processed_frames", "stride", "fps", "width", "height",
        "max_simultaneous", "unique_count", "inbound", "outbound", "net",
        "per_class", "line", "frames",
    ):
        assert key in data, key
    assert data["total_frames"] == 12
    assert data["max_simultaneous"] == 1  # 固定框 → 单目标
    assert data["unique_count"] == 1
    assert data["per_class"] == {"sheep": 1}  # 每 ID 取最近一次检测类别后按类计数
    assert (data["inbound"], data["outbound"], data["net"]) == (0, 0, 0)  # 静止不越线
    assert data["line"]["start"] and data["line"]["end"]
    previews = [f for f in data["frames"] if f["preview_jpeg"]]
    assert previews and previews[0]["count"] == 1
    assert len(data["frames"]) == 12


def test_count_video_default_stride_from_config(client, monkeypatch):
    """未传 stride 时使用配置步长（默认 2：耗时减半，跟踪稳定）。"""
    from app.config import settings
    from app.routers import vision as vision_router

    monkeypatch.setattr(vision_router, "detector", _MockVideoDetector())
    assert settings.video_stride == 2
    body = client.post(
        "/api/vision/count-video",
        files={"file": ("v.avi", _avi_bytes(), "video/x-msvideo")},
    ).json()
    assert body["code"] == 0
    assert body["data"]["stride"] == 2
    assert body["data"]["processed_frames"] == 6  # 12 帧按步长 2


def test_count_video_stride_reduces_processed_frames(client, monkeypatch):
    from app.routers import vision as vision_router

    monkeypatch.setattr(vision_router, "detector", _MockVideoDetector())
    body = client.post(
        "/api/vision/count-video",
        files={"file": ("v.avi", _avi_bytes(), "video/x-msvideo")},
        data={"stride": "3"},
    ).json()
    assert body["code"] == 0
    assert body["data"]["stride"] == 3
    assert body["data"]["processed_frames"] == 4  # 帧 0/3/6/9
    assert body["data"]["total_frames"] == 12


def test_count_video_rejects_oversize(client, monkeypatch):
    from app.config import settings
    from app.routers import vision as vision_router

    monkeypatch.setattr(vision_router, "detector", _MockVideoDetector())
    monkeypatch.setattr(settings, "video_max_bytes", 100)
    body = client.post(
        "/api/vision/count-video",
        files={"file": ("v.avi", _avi_bytes(), "video/x-msvideo")},
    ).json()
    assert body["code"] == 400
    assert "过大" in body["message"]


def test_count_video_rejects_over_duration(client, monkeypatch):
    from app.config import settings
    from app.routers import vision as vision_router

    monkeypatch.setattr(vision_router, "detector", _MockVideoDetector())
    monkeypatch.setattr(settings, "video_max_seconds", 0.05)  # 12 帧 @10fps = 1.2s
    body = client.post(
        "/api/vision/count-video",
        files={"file": ("v.avi", _avi_bytes(), "video/x-msvideo")},
    ).json()
    assert body["code"] == 400
    assert "时长超限" in body["message"]


def test_count_video_rejects_image_upload(client):
    """PNG 冒充视频 → 400（按文件头拦截，不交给 VideoCapture）。"""
    body = client.post(
        "/api/vision/count-video",
        files={"file": ("fake.mp4", _png_bytes(), "video/mp4")},
    ).json()
    assert body["code"] == 400
    assert "图片" in body["message"]


def test_count_video_rejects_unreadable_bytes(client):
    body = client.post(
        "/api/vision/count-video",
        files={"file": ("broken.mp4", b"not a video at all" * 40, "video/mp4")},
    ).json()
    assert body["code"] == 400
    assert "无法打开视频" in body["message"]


def test_existing_image_count_endpoint_unchanged(client, monkeypatch):
    """保留现有 /api/vision/count 行为不变（阶段 6 不回归）。"""
    from app.routers import vision as vision_router

    monkeypatch.setattr(vision_router, "detector", _MockDetector())
    body = client.post(
        "/api/vision/count",
        files={"file": ("sheep.png", _png_bytes(), "image/png")},
    ).json()
    assert body["code"] == 0
    assert body["data"]["count"] == 2


def test_static_index_served(client):
    """大屏页可访问：ECharts + 四视图导航 + 全部数据接口引用。"""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "echarts" in html
    assert "npmmirror" in html  # 首选 CDN
    assert "/api/dashboard/overview" in html
    # 左侧分组菜单：12 个页面入口（data-view 与前端 PAGES 注册表一致）
    for view in ("overview", "sheep", "grass", "rectify", "zoning", "forecast",
                 "health", "agent", "patrol", "devices", "users", "settings"):
        assert f'data-view="{view}"' in html
    assert 'id="menu"' in html and 'class="menu-item' in html
    assert 'id="pageTitle"' in html and 'id="pageSub"' in html
    # 核心接口引用：计数 / 载畜量 / Agent 思考流；草场覆盖度走多草场编号照片接口
    assert "/api/vision/count" in html
    assert "/api/vision/carrying-capacity" in html
    assert "/api/agent/stream" in html
    assert '"/api/pastures"' in html  # 草场清单 / 新建
    assert "/images" in html  # 编号照片上传与单张分析（归属校验）
    # 背景素材本地引用（不依赖外链）
    assert "assets/bg-pasture.jpg" in html
    # 图表归属各自视图：羊群数量趋势在识别视图、草场趋势在草场分析视图
    sheep_view = html.split('id="view-sheep"')[1].split('id="view-grass"')[0]
    grass_view = html.split('id="view-grass"')[1].split('id="view-rectify"')[0]
    assert 'id="sheepChart"' in sheep_view
    assert 'id="covChart"' in grass_view
    # 历史记录容器（羊群识别视图）
    assert 'id="sheepHistory"' in sheep_view
    # 多草场管理：草场列表 / 分区编号照片 / 编号上传与归属校验 / 一键分析（草场分析视图）
    for anchor in ('id="pastureList"', 'id="plotList"', 'id="pastureSummary"',
                   'id="uploadRow"', 'id="upCode"', 'id="newPastureForm"'):
        assert anchor in grass_view, anchor
    assert "/api/pastures" in html
    assert "/trend?days=30" in html
    assert "一键分析全部照片" in html
    assert "编号即归属证明" in html
    assert "data-analyze" in html
    # 塔基采集预处理已升级为四个能力独立页；草场分析页保留快捷入口（一键跳转）
    for view in ("rectify", "zoning", "forecast", "health", "patrol", "devices", "users", "settings"):
        assert f'id="view-{view}"' in html, view
    assert 'id="quickEntry"' in grass_view
    assert grass_view.count('class="qe-btn"') == 4
    for goto in ("rectify", "zoning", "forecast", "health"):
        assert f'data-goto="{goto}"' in grass_view, goto
    assert "四点标定" in html and "单应变换" in html and "分区自动分割" in html
    for gone in ('id="towerDrop"', 'id="towerCanvas"', 'id="towerRunBtn"', 'id="zoningRunBtn"'):
        assert gone not in grass_view, gone
    # 塔基几何校正独立页：画布 + 一键样例 + 接口（能力从草场分析页迁出）
    rectify_view = html.split('id="view-rectify"')[1].split('id="view-zoning"')[0]
    for anchor in ('id="rectifyCanvas"', 'id="rectifySampleBtn"', 'id="rectifyRunBtn"',
                   'id="rectifyDrop"', 'id="rectifyResult"'):
        assert anchor in rectify_view, anchor
    assert "/api/vision/orthorectify" in html
    assert "TOWER_SAMPLE_QUAD" in html
    # 分区自动分割独立页：滑杆控制 + 接口（能力从草场分析页迁出）
    zoning_view = html.split('id="view-zoning"')[1].split('id="view-forecast"')[0]
    for anchor in ('id="zoningDrop"', 'id="zoningRunBtn"', 'id="zoningPlots"'):
        assert anchor in zoning_view, anchor
    assert "/api/vision/segment" in html
    # 覆盖度时序预测：趋势图带预测虚线 + 置信带（Holt 线性趋势）
    assert "forecast_days=7" in html
    assert "置信上界" in html and '"预测"' in html
    # 多模态融合评分：草场详情展示评分与贡献分解
    assert "多模态草情评分" in html
    # 历史对话（AI 助手视图）与决策流跳转锚点
    agent_view = html.split('id="view-agent"')[1]
    assert 'id="historyList"' in agent_view
    assert "/api/agent/history" in html
    assert "data-log-id" in html
    assert "点击跳转到对应回答" in html
    # 视频计数（阶段 6）：模式切换 + canvas 播放器 + 视频接口与样例素材
    assert 'id="sheepModeRow"' in sheep_view
    assert 'id="videoCanvas"' in sheep_view
    assert 'id="videoPlay"' in sheep_view and 'id="vsFlow"' in sheep_view
    assert "/api/vision/count-video" in html
    assert 'data-video="/samples/video/aerial_flock.mp4"' in sheep_view
    assert 'data-video="/samples/video/sheep_road.mp4"' in sheep_view
    assert "getContext(\"2d\")" in html
    # 巡检告警（阶段 7 A3 + 反馈迭代）：逐场巡检 + 面板 + 实时弹出 + 跳转高亮
    overview_view = html.split('id="view-overview"')[1].split('id="view-sheep"')[0]
    for anchor in ('id="alertList"', 'id="alertCount"', 'id="patrolRunBtn"',
                   'id="alertPanel"', 'id="patrolResult"'):
        assert anchor in overview_view, anchor
    assert 'id="toastHost"' in html
    assert "EventSource(" in html
    assert "/api/dashboard/alerts" in html
    assert "/api/patrol/run" in html
    assert "setInterval(loadAlerts, 5000)" in html  # 轮询兜底
    # 点击实时弹窗 → 定位告警面板：聚光灯压暗其余界面 + 面板聚光 + 该条告警标记
    assert "function focusAlertPanel(" in html and "focusAlertPanel(alertItem.id)" in html
    assert "spotlight" in html and ".card.spot-on" in html
    assert "alert-marker" in html and "已定位到实时告警" in html
    # 逐场巡检结果展示（每个草场的结果都告诉用户）
    assert "function renderPatrolResult(" in html and "PATROL_REASON" in html
    # 各草场独立：hero 逐场状态 + KPI 不再有跨草场覆盖度/温湿度卡片
    assert 'id="heroPastures"' in overview_view and "renderHeroPastures" in html
    assert 'id="kpiPastures"' in overview_view
    for removed in ('id="kpiCoverage"', 'id="kpiTemp"', 'id="kpiHumidity"', "综合覆盖度"):
        assert removed not in html, removed
    # 概览布局：KPI 行（4 卡）与温湿度图表同列，右列为巡检告警 + 决策流（各占 1fr）
    assert 'id="envChart"' in overview_view
    assert ".kpis { flex: none; display: grid; grid-template-columns: repeat(4, 1fr)" in html
    # 思考流为浅色主题（需求：背景不要用深色）
    assert "#1c201c" not in html


def test_static_assets_served(client):
    """背景/纹理素材随静态目录提供（本地引用，不依赖外链）。"""
    for name in ("bg-pasture.jpg", "sheep-flock.jpg"):
        resp = client.get(f"/assets/{name}")
        assert resp.status_code == 200, name
        assert len(resp.content) > 1000, name


def test_static_responses_no_cache(client):
    """静态资源带 Cache-Control: no-cache（用前重新验证）——
    保证替换页面/样例素材后普通刷新即可生效；API 响应不受影响。"""
    for path in ("/", "/assets/bg-pasture.jpg", "/samples/video/aerial_flock.mp4"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert resp.headers.get("cache-control") == "no-cache", path
    assert client.get("/api/health").headers.get("cache-control") != "no-cache"


# ---------------------------------------------------------------------------
# 阶段 4.4：后台管理静态页
# ---------------------------------------------------------------------------

def test_admin_page_removed(client):
    """后台功能已并入单页四视图，/admin.html 不再提供（功能迁移，非删测达标）。"""
    resp = client.get("/admin.html")
    assert resp.status_code == 404
    # 迁移后的能力由 test_static_index_served 覆盖（四视图 + 全部接口引用）


# ---------------------------------------------------------------------------
# 阶段 F1：SSE 思考流 / 样例图挂载 / 标注图压缩
# ---------------------------------------------------------------------------

def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def _seeded_decisions() -> int:
    """播种的演示决策条数（每个演示草场一条）。"""
    from app.data.simulator import demo_pastures

    return len(demo_pastures())


def test_agent_stream_event_sequence(client, monkeypatch):
    """SSE 流：thinking → 工具事件序列（含草场口径）→ answer，且决策入库。"""
    from app.agent.orchestrator import AgentOrchestrator
    from app.data.db import connect
    from app.routers import agent as agent_router

    class _NoKeyLLM:
        enabled = False

    def _fake_dispatch(name, arguments):
        if name == "grassland_coverage":
            return {"coverage": 0.7, "greenness": 0.5}
        if name == "carrying_capacity":
            return {"recommended_stocking": 250.0, "rotation_plan": "轮牧"}
        if name == "pasture_status":
            return {
                "pasture": {
                    "name": "东区示范牧场",
                    "coverage": 0.7,
                    "alert_level": "normal",
                    "analyzed_plot_count": 3,
                    "plot_count": 3,
                }
            }
        return {"count": 3, "per_class": {"sheep": 3}}

    monkeypatch.setattr(
        agent_router,
        "orchestrator",
        AgentOrchestrator(llm=_NoKeyLLM(), dispatch=_fake_dispatch),
    )

    resp = client.post(
        "/api/agent/stream",
        json={"question": "东区草场够吗？亩产500公斤、100亩、放牧30天能放多少只羊？"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    assert events[0]["type"] == "thinking"
    assert events[-1]["type"] == "answer"
    assert [e["name"] for e in events if e["type"] == "tool_call"] == [
        "grassland_coverage",
        "sheep_count",
        "carrying_capacity",
        "pasture_status",
    ]
    assert events[-1]["decision"]["recommended_stocking"] == 250.0
    # 每个事件带毫秒时间戳；工具结果后跟确定性解读
    assert all(isinstance(e.get("ts"), int) for e in events)
    assert len([e for e in events if e["type"] == "analysis"]) == 4

    conn = connect()
    try:
        rows = conn.execute("SELECT COUNT(*) FROM decision_logs").fetchone()[0]
        stored = conn.execute(
            "SELECT events FROM decision_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()["events"]
    finally:
        conn.close()
    assert rows == _seeded_decisions() + 1  # 播种的演示决策 + 本次流式对话
    stored_events = json.loads(stored)
    assert [e["type"] for e in stored_events][0] == "thinking"
    assert stored_events[-1]["type"] == "answer"  # 思考事件随记录保存，供历史回放


def test_agent_stream_error_event(client, monkeypatch):
    """编排器异常时以 error 事件收尾，不抛出 500。"""
    from app.routers import agent as agent_router

    class _Boom:
        last_tool_calls: list = []

        def run(self, question, on_event=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(agent_router, "orchestrator", _Boom())
    resp = client.post("/api/agent/stream", json={"question": "x"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert events[-1]["type"] == "error"
    assert "boom" in events[-1]["message"]


# ---------------------------------------------------------------------------
# 历史对话：服务端保存（含思考过程）+ 概览决策流点击跳转到对应回答
# ---------------------------------------------------------------------------

def test_agent_history_saved_and_replayable(client, monkeypatch):
    """问答落库后：历史列表与单条记录均返回问题、工具调用、决策与思考事件。"""
    from app.agent.orchestrator import AgentOrchestrator
    from app.routers import agent as agent_router

    class _NoKeyLLM:
        enabled = False

    def _fake_dispatch(name, arguments):
        if name == "grassland_coverage":
            return {"coverage": 0.7, "greenness": 0.5}
        return {"count": 3, "per_class": {"sheep": 3}}

    monkeypatch.setattr(
        agent_router,
        "orchestrator",
        AgentOrchestrator(llm=_NoKeyLLM(), dispatch=_fake_dispatch),
    )
    assert client.post("/api/agent/stream", json={"question": "草够吗"}).status_code == 200

    logs = client.get("/api/agent/history?limit=20").json()["data"]["logs"]
    assert len(logs) == _seeded_decisions() + 1  # 播种的演示对话 + 本次问答
    log = logs[0]  # 最新一条在最前
    assert log["question"] == "草够吗"
    assert log["decision"]["alert_level"] in {"normal", "warning", "danger"}
    assert log["events"] and log["events"][-1]["type"] == "answer"

    detail = client.get(f"/api/agent/history/{log['id']}").json()
    assert detail["code"] == 0
    events = detail["data"]["events"]
    assert events[-1]["type"] == "answer"
    assert "覆盖度" in json.dumps(events, ensure_ascii=False)  # 思考过程可回放


def test_agent_history_missing_returns_404(client):
    assert client.get("/api/agent/history/99999").json()["code"] == 404


def test_stream_persists_before_answer_event(client, monkeypatch):
    """answer 事件推送前已完成落库（前端收到 answer 后立即刷新历史必能看到该条）。"""
    import asyncio

    from app.agent.orchestrator import AgentOrchestrator
    from app.data.db import connect
    from app.routers import agent as agent_router

    class _NoKeyLLM:
        enabled = False

    monkeypatch.setattr(
        agent_router,
        "orchestrator",
        AgentOrchestrator(llm=_NoKeyLLM(), dispatch=lambda name, args: {"coverage": 0.7, "greenness": 0.5}),
    )

    response = asyncio.run(
        agent_router.chat_stream_endpoint(agent_router.ChatRequest(question="草够吗"))
    )

    async def first_answer_chunk():
        async for chunk in response.body_iterator:
            if "answer" in str(chunk):
                return chunk
        return None

    assert asyncio.run(first_answer_chunk()) is not None

    conn = connect()
    try:
        rows = conn.execute(
            "SELECT COUNT(*) FROM decision_logs WHERE question = '草够吗'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert rows == 1  # 客户端收到 answer 时记录已落库（无列表缺条竞态）


def test_decision_flow_items_carry_log_id(client):
    """大屏决策流每条带 id：前端以此为锚点跳转到 AI 助手视图的对应回答。"""
    from app.data import repository
    from app.data.db import connect

    conn = connect()
    try:
        repository.insert_decision_log(
            conn,
            question="东区草够吗？",
            tool_calls=["grassland_coverage"],
            decision_json={"alert_level": "warning", "recommended_stocking": 100.0},
        )
        conn.commit()
        expected_id = conn.execute("SELECT MAX(id) FROM decision_logs").fetchone()[0]
    finally:
        conn.close()

    logs = client.get("/api/dashboard/decision-flow?limit=1").json()["data"]["logs"]
    assert logs[0]["id"] == expected_id


def test_agent_stream_validation(client):
    resp = client.post("/api/agent/stream", json={"question": ""})
    assert resp.status_code == 422


def test_samples_static_mount(client):
    """样例图通过 /samples/ 直接可访问（供前端快速样例按钮）。"""
    from pathlib import Path

    samples = Path(__file__).resolve().parent.parent / "data" / "samples"
    names = sorted(
        p.name
        for p in samples.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    assert names, "样例图缺失"
    resp = client.get(f"/samples/{names[0]}")
    assert resp.status_code == 200
    assert len(resp.content) > 0


def test_encode_image_downscales_and_compresses():
    """标注图编码限宽 1600 + JPEG 压缩（大图响应体积可控）。"""
    import base64

    from app.routers.vision import decode_image, encode_image

    big = np.zeros((2000, 3000, 3), dtype=np.uint8)
    data = base64.b64decode(encode_image(big))
    decoded = decode_image(data)
    assert decoded.shape[1] == 1600  # 限宽
    assert data[:2] == b"\xff\xd8"  # JPEG 魔数

    small = np.zeros((100, 100, 3), dtype=np.uint8)
    assert decode_image(base64.b64decode(encode_image(small))).shape[1] == 100  # 小图不放大


def test_encode_image_max_height_for_portrait():
    """竖屏帧按高度上限缩放（视频预览帧瘦身）。"""
    import base64

    from app.routers.vision import decode_image, encode_image

    portrait = np.zeros((1280, 720, 3), dtype=np.uint8)
    decoded = decode_image(base64.b64decode(encode_image(portrait, max_width=560, max_height=560)))
    assert decoded.shape[0] == 560  # 高度受限
    assert decoded.shape[1] == 315  # 等比

    landscape = np.zeros((720, 1280, 3), dtype=np.uint8)
    decoded2 = decode_image(base64.b64decode(encode_image(landscape, max_width=560, max_height=560)))
    assert decoded2.shape[1] == 560  # 横屏仍按宽度限制


# ---------------------------------------------------------------------------
# 阶段 7.3：巡检后台任务的启停（默认关闭无副作用）
# ---------------------------------------------------------------------------

def test_patrol_disabled_by_default_no_side_effects(client):
    """默认关闭：应用照常启动、无巡检任务、alerts 表保持为空。"""
    from app.config import settings
    from app.data.db import connect

    assert settings.patrol_enabled is False
    assert client.get("/api/dashboard/overview").json()["code"] == 0
    conn = connect()
    try:
        assert conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 0
    finally:
        conn.close()


def test_app_runs_first_patrol_round_when_enabled(tmp_path, monkeypatch):
    """开启配置：lifespan 创建巡检任务且首轮立即执行（逐草场判定 → 非正常草场各自落库）。"""
    from app.config import settings
    from app.data.db import connect
    from fastapi.testclient import TestClient

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "patrol_app.db"))
    monkeypatch.setattr(settings, "patrol_enabled", True)
    monkeypatch.setattr(settings, "patrol_interval_seconds", 3600.0)  # 首轮后长眠

    from app.main import app

    with TestClient(app) as c:
        assert c.get("/api/health").json()["code"] == 0
        rows = []
        for _ in range(50):  # 后台首轮是异步的：最多等 2.5s
            conn = connect()
            try:
                rows = conn.execute("SELECT level, title, pasture_id FROM alerts").fetchall()
            finally:
                conn.close()
            if len(rows) >= 2:  # 播种三草场：预警 + 危险各一条
                break
            time.sleep(0.05)

    assert len(rows) == 2  # 逐草场告警：正常草场不告警，预警/危险草场各一条
    assert {r["level"] for r in rows} == {"warning", "danger"}
    assert all(r["pasture_id"] for r in rows)  # 每条告警都标明归属草场
    assert all("P" in r["title"] for r in rows)


# ---------------------------------------------------------------------------
# 阶段 7.4 / 7.7：手动巡检 → 告警历史 → 确认；升级时 LLM 建议
# ---------------------------------------------------------------------------

def test_patrol_run_manual_alert_history_and_ack(client):
    """手动巡检：逐草场判定 → 落库 → 历史可查 → 确认生效 → 未命中 404。"""
    body = client.post("/api/patrol/run").json()
    assert body["code"] == 0
    data = body["data"]
    # 播种三草场各自出结果（含未告警的正常草场），每个草场独立判定
    assert data["summary"]["total"] == 3
    assert data["summary"]["alerted"] == len(data["alerts"]) >= 2
    codes = [r["pasture_code"] for r in data["results"]]
    assert sorted(codes) == ["P1", "P2", "P3"]
    normal = [r for r in data["results"] if r["level"] == "normal"]
    assert normal and normal[0]["should_alert"] is False
    assert normal[0]["coverage"] is not None  # 未告警也给出该场覆盖度
    assert {a["level"] for a in data["alerts"]} == {"warning", "danger"}
    assert all(a["pasture_code"] and a["pasture_code"] in a["title"] for a in data["alerts"])

    alerts = client.get("/api/dashboard/alerts").json()["data"]["alerts"]
    assert len(alerts) == len(data["alerts"])
    assert alerts[0]["id"] == data["alerts"][-1]["id"]
    assert alerts[0]["acknowledged"] is False
    assert alerts[0]["message"] and alerts[0]["metrics"]["coverage"] is not None
    assert alerts[0]["pasture_code"]  # 告警面板可按草场区分

    ack = client.post(f"/api/dashboard/alerts/{alerts[0]['id']}/ack").json()
    assert ack["code"] == 0 and ack["data"]["acknowledged"] is True
    assert client.get("/api/dashboard/alerts").json()["data"]["alerts"][0]["acknowledged"] is True

    assert client.post("/api/dashboard/alerts/99999/ack").json()["code"] == 404


def test_patrol_run_respects_cooldown(client):
    """同级且在冷却期内：逐场抑制（每个草场各自冷却），不新增告警。"""
    first = client.post("/api/patrol/run").json()["data"]
    second = client.post("/api/patrol/run").json()["data"]
    assert second["summary"]["alerted"] == 0
    assert second["summary"]["suppressed"] == len(first["alerts"])
    assert all(r["reason"] == "cooldown_suppressed" for r in second["results"]
               if r["level"] != "normal")
    assert len(client.get("/api/dashboard/alerts").json()["data"]["alerts"]) == len(first["alerts"])


def test_alerts_history_empty_by_default(client):
    assert client.get("/api/dashboard/alerts").json()["data"]["alerts"] == []


def test_patrol_advice_only_on_escalation_over_api(client, monkeypatch):
    """升级路径：LLM 建议写入告警文案；首次告警不调用 LLM。"""
    from app.agent import patrol
    from app.data import repository
    from app.data.db import connect

    calls: list[str] = []

    def fake_advice(metrics, decision):
        calls.append(decision.reason)
        return "建议：立即禁牧并安排补播。"

    monkeypatch.setattr(patrol, "patrol_advice", fake_advice)

    first = client.post("/api/patrol/run").json()["data"]
    assert first["alerts"] and calls == []  # 首次告警：不调 LLM

    # 制造升级：把全部草场的分区覆盖度压到危险区间
    conn = connect()
    try:
        for pasture in repository.list_pastures(conn):
            for image in repository.list_pasture_images(conn, pasture["id"]):
                repository.insert_grassland_metric(
                    conn, region=pasture["code"], coverage=0.10, greenness=0.02,
                    pasture_id=pasture["id"], plot_code=image["plot_code"],
                    image_code=image["image_code"],
                )
        conn.commit()
    finally:
        conn.close()

    escalated = client.post("/api/patrol/run").json()["data"]
    # 三草场全压到危险：原本正常的 P1 走「首次告警」、原预警的 P2「升级」、原危险的 P3 同级抑制
    assert {a["level"] for a in escalated["alerts"]} == {"danger"}
    assert {a["pasture_code"] for a in escalated["alerts"]} == {"P1", "P2"}
    assert calls == ["escalated"]  # 仅升级路径调用 LLM（P1 是首次、P3 被冷却抑制）
    advised = [a for a in escalated["alerts"] if "AI 建议：建议：立即禁牧并安排补播。" in a["message"]]
    assert len(advised) == 1 and advised[0]["pasture_code"] == "P2"
