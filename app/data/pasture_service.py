"""草场装配服务：草场摘要、分区明细、单张照片分析。

「照片集合 → 草场状况」的归并口径统一在 app.vision.grassland（确定性、可单测），
本模块负责把仓储数据装配成接口/大屏所需的形状。
"""
from __future__ import annotations

from pathlib import Path

import cv2

from app.config import BASE_DIR
from app.data import repository
from app.vision.grassland import (
    alert_level_from_coverage,
    coverage,
    grass_health_score,
    pasture_aggregate,
    pasture_recommendation,
    plot_coverage_map,
)

SAMPLES_DIR = BASE_DIR / "data" / "samples"


def local_image_path(image_path: str | None) -> Path | None:
    """静态资源路径（/samples/...）→ 本地文件路径；绝对/相对路径原样解析。"""
    if not image_path:
        return None
    if image_path.startswith("/samples/"):
        return SAMPLES_DIR / image_path[len("/samples/") :]
    path = Path(image_path)
    return path if path.is_absolute() else BASE_DIR / path


def pasture_summary(conn, pasture) -> dict:
    """草场摘要：基础信息 + 照片/分区数 + 本草场覆盖度（各分区照片聚合）+ 载畜量建议。"""
    metrics = repository.latest_image_metrics(conn, pasture["id"])
    plot_values = plot_coverage_map(metrics)
    agg = pasture_aggregate(list(plot_values.values()))
    rec = pasture_recommendation(agg["coverage"], float(pasture["area_mu"] or 0))

    images = repository.list_pasture_images(conn, pasture["id"])
    plot_codes = sorted({i["plot_code"] for i in images} | set(plot_values))
    worst_plot, worst_coverage = (None, None)
    if plot_values:
        worst_plot, worst_coverage = min(plot_values.items(), key=lambda kv: kv[1])
    greenness_values = [m["greenness"] for m in metrics if m["greenness"] is not None]
    greenness = round(sum(greenness_values) / len(greenness_values), 4) if greenness_values else None

    # 多模态融合草情评分：覆盖度 + 绿度 + 最新环境温湿度（缺项自动归一化权重）
    env = repository.list_environment_records(conn, limit=1)
    health = grass_health_score(
        agg["coverage"],
        greenness,
        env[0]["temperature"] if env else None,
        env[0]["humidity"] if env else None,
    )

    return {
        "id": pasture["id"],
        "code": pasture["code"],
        "name": pasture["name"],
        "region": pasture["region"],
        "area_mu": pasture["area_mu"],
        "note": pasture["note"],
        "image_count": len(images),
        "plot_count": len(plot_codes),
        "analyzed_plot_count": len(plot_values),
        "coverage": agg["coverage"],
        "greenness": greenness,
        "alert_level": agg["alert_level"],
        "min_coverage": agg["min_coverage"],
        "max_coverage": agg["max_coverage"],
        "worst_plot": worst_plot,
        "worst_coverage": round(worst_coverage, 4) if worst_coverage is not None else None,
        "theoretical": rec["theoretical"],
        "factor": rec["factor"],
        "recommended_stocking": rec["recommended_stocking"],
        "health": health,
        "updated_at": max((m["created_at"] for m in metrics), default=None),
    }


def plots_detail(conn, pasture) -> list[dict]:
    """分区明细：每个分区下的编号照片及其最新分析值（分区值 = 各图片最新值均值）。"""
    images = repository.list_pasture_images(conn, pasture["id"])
    metrics = repository.latest_image_metrics(conn, pasture["id"])
    by_image = {m["image_code"]: m for m in metrics if m["image_code"]}
    plot_level = {m["plot_code"]: m for m in metrics if not m["image_code"]}

    plots: dict[str, dict] = {}
    for image in images:
        plot = plots.setdefault(image["plot_code"], {"plot_code": image["plot_code"], "images": []})
        metric = by_image.get(image["image_code"])
        plot["images"].append(
            {
                "image_code": image["image_code"],
                "image_path": image["image_path"],
                "coverage": metric["coverage"] if metric else None,
                "greenness": metric["greenness"] if metric else None,
                "analyzed_at": metric["created_at"] if metric else None,
            }
        )

    # 有分析记录但未登记照片的分区（例如仅按编号上传分析过）
    for metric in metrics:
        if metric["plot_code"] in plots:
            continue
        plots[metric["plot_code"]] = {
            "plot_code": metric["plot_code"],
            "images": [
                {
                    "image_code": metric["image_code"],
                    "image_path": metric["image_path"],
                    "coverage": metric["coverage"],
                    "greenness": metric["greenness"],
                    "analyzed_at": metric["created_at"],
                }
            ],
        }

    result = []
    for code in sorted(plots):
        plot = plots[code]
        values = [i["coverage"] for i in plot["images"] if i["coverage"] is not None]
        if not values and code in plot_level and plot_level[code]["coverage"] is not None:
            values = [plot_level[code]["coverage"]]
        value = round(sum(values) / len(values), 4) if values else None
        plot["coverage"] = value
        plot["alert_level"] = alert_level_from_coverage(value)
        plot["analyzed_at"] = max(
            (i["analyzed_at"] for i in plot["images"] if i["analyzed_at"]), default=None
        )
        result.append(plot)
    return result


def analyze_local_image(
    conn, pasture, image_code: str, plot_code: str, image_path: str | None
) -> dict:
    """分析一张已登记的草场照片并入库；文件缺失时返回错误项（不中断批量分析）。"""
    local = local_image_path(image_path)
    img = cv2.imread(str(local)) if local else None
    if img is None:
        return {"image_code": image_code, "error": f"图片文件缺失或无法读取：{image_path or '（未登记路径）'}"}
    result = coverage(img)
    repository.insert_grassland_metric(
        conn,
        region=pasture["code"],
        coverage=result["coverage"],
        greenness=result["greenness"],
        pasture_id=pasture["id"],
        plot_code=plot_code,
        image_code=image_code,
        image_path=image_path,
    )
    return {
        "image_code": image_code,
        "plot_code": plot_code,
        "image_path": image_path,
        "coverage": round(result["coverage"], 4),
        "greenness": round(result["greenness"], 4),
    }
