"""草场管理路由：多草场 CRUD、分区编号照片登记、逐张/批量覆盖度分析、草场趋势。

一个草场 = 多个分区照片的集合；照片编号（草场-分区-序号，如 P1-A-01）自证归属。
装配逻辑在 app.data.pasture_service，聚合口径在 app.vision.grassland（确定性）。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, File, Form, Query, UploadFile

from app.data import repository
from app.data.db import connect
from app.data.forecast import holt_linear
from app.data.pasture_service import analyze_local_image, pasture_summary, plots_detail
from app.routers.vision import decode_image
from app.schemas import PastureCreate, err, ok
from app.vision.grassland import coverage

router = APIRouter(prefix="/api/pastures", tags=["pasture"])


@router.get("")
def list_pastures_endpoint():
    """草场清单（含本草场覆盖度/告警等级/照片与分区数/载畜量建议）。"""
    try:
        conn = connect()
        try:
            rows = repository.list_pastures(conn)
            return ok({"pastures": [pasture_summary(conn, r) for r in rows]})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.post("")
def create_pasture_endpoint(req: PastureCreate):
    """新建草场（编号唯一；照片通过 /{id}/images 登记）。"""
    try:
        conn = connect()
        try:
            if repository.get_pasture_by_code(conn, req.code):
                return err(400, f"草场编号已存在：{req.code}")
            pasture_id = repository.insert_pasture(
                conn,
                code=req.code,
                name=req.name,
                region=req.region,
                area_mu=req.area_mu,
                note=req.note,
            )
            conn.commit()
            return ok(pasture_summary(conn, repository.get_pasture(conn, pasture_id)))
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.get("/{pasture_id}")
def pasture_detail_endpoint(pasture_id: int):
    """草场详情：摘要 + 分区明细（编号照片、各分区最新覆盖度与判定）。"""
    try:
        conn = connect()
        try:
            pasture = repository.get_pasture(conn, pasture_id)
            if pasture is None:
                return err(404, f"草场不存在：{pasture_id}")
            return ok({"pasture": pasture_summary(conn, pasture), "plots": plots_detail(conn, pasture)})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.delete("/{pasture_id}")
def delete_pasture_endpoint(pasture_id: int):
    """删除草场（其照片与覆盖度记录一并移除）。"""
    try:
        conn = connect()
        try:
            pasture = repository.get_pasture(conn, pasture_id)
            if pasture is None:
                return err(404, f"草场不存在：{pasture_id}")
            repository.delete_pasture(conn, pasture_id)
            conn.commit()
            return ok({"deleted": pasture_id, "code": pasture["code"]})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.post("/{pasture_id}/images")
async def upload_pasture_image_endpoint(
    pasture_id: int,
    file: UploadFile = File(...),
    image_code: str = Form(...),
    plot_code: str | None = Form(None),
):
    """按编号上传一张草场照片 → 覆盖度分析 → 登记为草场分区照片并入库。

    编号约定「草场编号-分区-序号」（如 P1-A-01），编号即归属证明：
    编号必须以该草场编号开头；分区缺省时从编号中解析，两处不一致则报错。
    """
    try:
        conn = connect()
        try:
            pasture = repository.get_pasture(conn, pasture_id)
            if pasture is None:
                return err(404, f"草场不存在：{pasture_id}")

            code = (image_code or "").strip()
            if not code.startswith(f"{pasture['code']}-"):
                return err(
                    400,
                    f"图片编号需以草场编号开头（如 {pasture['code']}-A-01），用于标明照片归属",
                )
            segments = code.split("-")
            derived_plot = segments[1] if len(segments) >= 3 and segments[1] else None
            plot = (plot_code or "").strip() or derived_plot
            if plot is None:
                return err(400, "无法从编号解析分区，请提供分区或使用 草场-分区-序号 格式")
            if derived_plot and plot_code and plot != derived_plot:
                return err(400, f"分区与编号不一致：编号中为 {derived_plot}，提交为 {plot_code}")

            try:
                img = decode_image(await file.read())
            except ValueError as e:
                return err(400, str(e))

            result = coverage(img)
            repository.insert_pasture_image(
                conn, pasture_id=pasture_id, plot_code=plot, image_code=code, image_path=None
            )
            repository.insert_grassland_metric(
                conn,
                region=pasture["code"],
                coverage=result["coverage"],
                greenness=result["greenness"],
                pasture_id=pasture_id,
                plot_code=plot,
                image_code=code,
                image_path=None,
            )
            conn.commit()
            return ok(
                {
                    "image": {
                        "image_code": code,
                        "plot_code": plot,
                        "coverage": round(result["coverage"], 4),
                        "greenness": round(result["greenness"], 4),
                    },
                    "pasture": pasture_summary(conn, pasture),
                }
            )
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.post("/{pasture_id}/images/{image_code}/analyze")
def analyze_pasture_image_endpoint(pasture_id: int, image_code: str):
    """重新分析该草场已登记的某一张编号照片（单张分析/复算）。"""
    try:
        conn = connect()
        try:
            pasture = repository.get_pasture(conn, pasture_id)
            if pasture is None:
                return err(404, f"草场不存在：{pasture_id}")
            image = next(
                (
                    i
                    for i in repository.list_pasture_images(conn, pasture_id)
                    if i["image_code"] == image_code
                ),
                None,
            )
            if image is None:
                return err(404, f"该草场未登记照片编号：{image_code}")

            result = analyze_local_image(
                conn, pasture, image["image_code"], image["plot_code"], image["image_path"]
            )
            if result.get("error"):
                return err(400, result["error"])
            conn.commit()
            return ok({"image": result, "pasture": pasture_summary(conn, pasture)})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.post("/{pasture_id}/analyze")
def analyze_pasture_endpoint(pasture_id: int):
    """一键分析该草场全部已登记照片（照片集合 → 草场状况）。"""
    try:
        conn = connect()
        try:
            pasture = repository.get_pasture(conn, pasture_id)
            if pasture is None:
                return err(404, f"草场不存在：{pasture_id}")
            images = repository.list_pasture_images(conn, pasture_id)
            if not images:
                return err(400, "该草场尚未登记任何照片")

            results, failed = [], 0
            for image in images:
                item = analyze_local_image(
                    conn, pasture, image["image_code"], image["plot_code"], image["image_path"]
                )
                if item.get("error"):
                    failed += 1
                results.append(item)
            conn.commit()
            return ok(
                {
                    "pasture": pasture_summary(conn, pasture),
                    "results": results,
                    "analyzed": len(results) - failed,
                    "failed": failed,
                }
            )
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.get("/{pasture_id}/trend")
def pasture_trend_endpoint(
    pasture_id: int,
    days: int = Query(30, ge=1, le=365),
    forecast_days: int = Query(0, ge=0, le=30, description="预测天数（>0 时用 Holt 线性趋势外推并附置信带）"),
):
    """该草场按日聚合的覆盖度趋势（当日各条记录均值，升序）+ 可选时序预测。"""
    try:
        conn = connect()
        try:
            pasture = repository.get_pasture(conn, pasture_id)
            if pasture is None:
                return err(404, f"草场不存在：{pasture_id}")
            rows = repository.daily_coverage_series(conn, days=days, pasture_id=pasture_id)
            series = [
                {
                    "day": r["day"],
                    "coverage": round(r["coverage"], 4) if r["coverage"] is not None else None,
                    "greenness": round(r["greenness"], 4) if r["greenness"] is not None else None,
                    "n": r["n"],
                }
                for r in rows
            ]

            # Holt 线性趋势外推（短期趋势预警辅助；置信带随步长加宽）
            forecast = None
            values = [s["coverage"] for s in series if s["coverage"] is not None]
            if forecast_days > 0 and len(values) >= 3:
                result = holt_linear(values, horizon=forecast_days)
                last_day = datetime.strptime(series[-1]["day"], "%Y-%m-%d")
                forecast = {
                    "days": [
                        (last_day + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                        for i in range(forecast_days)
                    ],
                    "values": result["forecast"],
                    "band": result["band"],
                    "residual_std": result["residual_std"],
                    "trend": result["trend"],
                }
            return ok(
                {
                    "pasture": {"id": pasture["id"], "code": pasture["code"], "name": pasture["name"]},
                    "days": days,
                    "forecast_days": forecast_days,
                    "series": series,
                    "forecast": forecast,
                }
            )
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))
