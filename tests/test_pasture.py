"""多草场管理（反馈迭代）：聚合口径、仓储 CRUD、编号照片 API。

核心口径（确定性与可复现）：
- 分区值 = 该分区各照片最新分析值的均值；
- 草场覆盖度 = 该草场各分区值的均值；告警等级按统一阈值；
- **草场之间彼此独立**：跨草场不做覆盖度平均，只做状态汇总（最严重等级 / 载畜量求和）；
- 建议载畜量 = 理论载畜量（按草场面积）× 告警系数（正常 1.0 / 预警 0.6 / 危险 0）。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.data import repository
from app.vision.grassland import (
    ALERT_CAPACITY_FACTOR,
    LEVEL_SEVERITY,
    carrying_capacity,
    coverage_linked_yield,
    pasture_aggregate,
    pasture_recommendation,
    plot_coverage_map,
    worst_level,
)

MANIFEST = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "samples" / "grassland" / "manifest.json")
    .read_text(encoding="utf-8")
)
MANIFEST_PASTURES = {p["code"]: p for p in MANIFEST["pastures"]}


def _row(plot_code, image_code, coverage, rid):
    return {"plot_code": plot_code, "image_code": image_code, "coverage": coverage, "id": rid}


def _green_png_bytes() -> bytes:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:] = (0, 160, 0)
    ok_flag, buf = cv2.imencode(".png", img)
    assert ok_flag
    return buf.tobytes()


# ---------------------------------------------------------------------------
# 聚合与建议（纯函数）
# ---------------------------------------------------------------------------

def test_pasture_aggregate_mean_and_alert_level():
    agg = pasture_aggregate([0.80, 0.70, 0.90])
    assert agg["coverage"] == 0.8
    assert agg["alert_level"] == "normal"
    assert agg["plot_count"] == 3
    assert agg["min_coverage"] == 0.7 and agg["max_coverage"] == 0.9

    assert pasture_aggregate([0.50, 0.40])["alert_level"] == "warning"
    assert pasture_aggregate([0.30, 0.10])["alert_level"] == "danger"


def test_pasture_aggregate_empty_means_unanalyzed():
    """无已分析分区：coverage=None（未分析），不臆断为正常或危险。"""
    agg = pasture_aggregate([])
    assert agg["coverage"] is None
    assert agg["alert_level"] == "normal"
    assert agg["plot_count"] == 0


def test_worst_level_rolls_up_without_averaging():
    """跨草场只做状态汇总：取最严重等级；无有效等级返回 None（不臆断）。"""
    assert LEVEL_SEVERITY == {"normal": 0, "warning": 1, "danger": 2}
    assert worst_level(["normal", "warning", "danger"]) == "danger"
    assert worst_level(["normal", "warning"]) == "warning"
    assert worst_level(["normal", "normal"]) == "normal"
    assert worst_level([]) is None
    assert worst_level([None, "", "unknown"]) is None


def test_plot_coverage_map_image_level_latest_and_plot_mean():
    """分区值 = 该分区各照片最新值均值；同图旧记录被最新记录覆盖；分区级记录兜底。"""
    values = plot_coverage_map(
        [
            _row("A", "P1-A-01", 0.60, 1),
            _row("A", "P1-A-01", 0.80, 5),  # 同一张照片的更新分析 → 取 0.80
            _row("A", "P1-A-02", 0.60, 3),
            _row("B", "P1-B-01", 0.40, 4),
            _row("C", None, 0.30, 2),  # 无图片编号 → 分区级兜底
            _row(None, None, 0.99, 6),  # 未关联分区 → 不参与
        ]
    )
    assert values == {"A": 0.7, "B": 0.4, "C": 0.3}


def test_plot_coverage_map_plot_level_does_not_override_images():
    values = plot_coverage_map([_row("A", "P1-A-01", 0.80, 1), _row("A", None, 0.10, 9)])
    assert values["A"] == 0.8


def test_pasture_recommendation_scales_with_area_and_alert():
    theoretical = carrying_capacity(coverage_linked_yield(0.80), 120.0, 0.5, 30.0)[
        "recommended_stocking"
    ]

    normal = pasture_recommendation(0.80, 120.0)
    assert normal["alert_level"] == "normal"
    assert normal["factor"] == ALERT_CAPACITY_FACTOR["normal"] == 1.0
    assert normal["theoretical"] == theoretical
    assert normal["recommended_stocking"] == round(theoretical, 2)

    warning = pasture_recommendation(0.45, 100.0)
    assert warning["alert_level"] == "warning"
    assert warning["factor"] == 0.6
    assert warning["recommended_stocking"] == round(warning["theoretical"] * 0.6, 2)

    danger = pasture_recommendation(0.20, 60.0)
    assert danger["theoretical"] > 0  # 理论值仍然算出
    assert danger["factor"] == 0.0
    assert danger["recommended_stocking"] == 0.0  # 禁牧


def test_pasture_recommendation_unanalyzed_or_zero_area():
    unanalyzed = pasture_recommendation(None, 100.0)
    assert unanalyzed["theoretical"] == 0.0 and unanalyzed["recommended_stocking"] == 0.0
    assert pasture_recommendation(0.8, 0.0)["theoretical"] == 0.0


# ---------------------------------------------------------------------------
# 仓储：草场 CRUD、照片登记、级联删除
# ---------------------------------------------------------------------------

def _db(tmp_path):
    from app.data.db import connect, init_db

    db = tmp_path / "pasture.db"
    init_db(db)
    return connect(db)


def test_pasture_repository_crud(tmp_path):
    conn = _db(tmp_path)
    try:
        pasture_id = repository.insert_pasture(
            conn, code="P9", name="测试草场", region="南区", area_mu=50.0, note="单元测试"
        )
        assert repository.get_pasture(conn, pasture_id)["name"] == "测试草场"
        assert repository.get_pasture_by_code(conn, "P9")["id"] == pasture_id
        assert [p["code"] for p in repository.list_pastures(conn)] == ["P9"]
        assert repository.get_pasture(conn, 12345) is None
    finally:
        conn.close()


def test_pasture_image_upsert_and_delete_cascade(tmp_path):
    conn = _db(tmp_path)
    try:
        pasture_id = repository.insert_pasture(conn, code="P9", name="测试草场", area_mu=10.0)
        repository.insert_pasture_image(
            conn, pasture_id=pasture_id, plot_code="A", image_code="P9-A-01", image_path="/x.jpg"
        )
        # 同编号重新登记 → 更新而非重复
        repository.insert_pasture_image(
            conn, pasture_id=pasture_id, plot_code="B", image_code="P9-A-01", image_path="/y.jpg"
        )
        images = repository.list_pasture_images(conn, pasture_id)
        assert len(images) == 1
        assert images[0]["plot_code"] == "B" and images[0]["image_path"] == "/y.jpg"

        repository.insert_grassland_metric(
            conn, region="P9", coverage=0.5, greenness=0.1, pasture_id=pasture_id,
            plot_code="B", image_code="P9-A-01", image_path="/y.jpg",
        )
        conn.commit()

        repository.delete_pasture(conn, pasture_id)
        conn.commit()
        assert repository.get_pasture(conn, pasture_id) is None
        assert repository.list_pasture_images(conn, pasture_id) == []
        assert repository.list_metrics_for_pasture(conn, pasture_id) == []
    finally:
        conn.close()


def test_daily_coverage_series_groups_by_day(tmp_path):
    conn = _db(tmp_path)
    try:
        pasture_id = repository.insert_pasture(conn, code="P9", name="九号", area_mu=10.0)
        for coverage, ts in [(0.4, "2026-09-01T08:00:00.000Z"), (0.6, "2026-09-01T20:00:00.000Z"),
                             (0.2, "2026-09-02T08:00:00.000Z")]:
            repository.insert_grassland_metric(
                conn, region="P9", coverage=coverage, greenness=0.1, created_at=ts,
                pasture_id=pasture_id, plot_code="A", image_code="P9-A-01",
            )
        # 未关联草场的记录不计入
        repository.insert_grassland_metric(conn, region="东区", coverage=0.9, greenness=0.2,
                                           created_at="2026-09-01T09:00:00.000Z")
        conn.commit()

        series = repository.daily_coverage_series(conn, days=30, pasture_id=pasture_id)
        assert [r["day"] for r in series] == ["2026-09-01", "2026-09-02"]
        assert abs(series[0]["coverage"] - 0.5) < 1e-9  # (0.4 + 0.6) / 2
        assert series[0]["n"] == 2

        # 草场彼此独立：不存在跨草场合并的覆盖度序列（未关联草场的记录也不参与）
        with pytest.raises(TypeError):
            repository.daily_coverage_series(conn, days=30)  # type: ignore[call-arg]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API：播种、编号照片、批量分析、趋势、增删
# ---------------------------------------------------------------------------

def test_pastures_seeded_with_numbered_photos(client):
    """播种的演示草场：带编号照片、分区与综合覆盖度，且照片编号自证归属。"""
    pastures = client.get("/api/pastures").json()["data"]["pastures"]
    assert len(pastures) >= 2
    assert {"P1", "P2", "P3"} <= {p["code"] for p in pastures}

    for pasture in pastures:
        assert pasture["image_count"] > 0
        assert pasture["plot_count"] > 0
        assert pasture["coverage"] is not None
        assert pasture["alert_level"] in {"normal", "warning", "danger"}

    p1 = next(p for p in pastures if p["code"] == "P1")
    detail = client.get(f"/api/pastures/{p1['id']}").json()["data"]
    assert detail["pasture"]["code"] == "P1"
    assert len(detail["plots"]) == p1["plot_count"]
    for plot in detail["plots"]:
        assert plot["coverage"] is not None
        assert plot["alert_level"] in {"normal", "warning", "danger"}
        for image in plot["images"]:
            assert image["image_code"].startswith("P1-")
            assert image["image_code"].split("-")[1] == plot["plot_code"]
            assert image["image_path"].startswith("/samples/grassland/")
            assert image["coverage"] is not None


def test_pastures_span_three_alert_levels(client):
    """演示样例三个草场分别覆盖 正常/预警/危险 三档（与样例清单 expected 一致）。"""
    pastures = {
        p["code"]: p for p in client.get("/api/pastures").json()["data"]["pastures"]
    }
    for code, entry in MANIFEST_PASTURES.items():
        assert pastures[code]["alert_level"] == entry["expected"]["level"], code


def test_analyze_pasture_aggregates_photo_set(client):
    """一键分析：草场状况由编号照片集合聚合而来（分区值→草场值，与清单实测一致）。"""
    pastures = {p["code"]: p for p in client.get("/api/pastures").json()["data"]["pastures"]}
    p1 = pastures["P1"]

    resp = client.post(f"/api/pastures/{p1['id']}/analyze").json()
    assert resp["code"] == 0
    data = resp["data"]
    assert data["analyzed"] == p1["image_count"]
    assert data["failed"] == 0
    assert all(r["coverage"] is not None for r in data["results"])

    # 期望值 = 各分区（各照片均值）的均值
    plots: dict[str, list[float]] = {}
    for image in MANIFEST_PASTURES["P1"]["images"]:
        plots.setdefault(image["code"].split("-")[1], []).append(image["coverage"])
    expected = round(sum(sum(v) / len(v) for v in plots.values()) / len(plots), 4)
    assert abs(data["pasture"]["coverage"] - expected) < 1e-3
    assert data["pasture"]["alert_level"] == MANIFEST_PASTURES["P1"]["expected"]["level"]

    # 分析结果入库：草场趋势最后一天包含当天记录
    trend = client.get(f"/api/pastures/{p1['id']}/trend?days=30").json()["data"]
    assert trend["series"]
    assert trend["series"][-1]["day"] == max(s["day"] for s in trend["series"])
    assert trend["series"][-1]["n"] >= p1["image_count"]


def test_upload_pasture_image_validates_code_ownership(client):
    """编号即归属证明：编号必须以草场编号开头，且与提交的分区一致。"""
    p1 = next(
        p for p in client.get("/api/pastures").json()["data"]["pastures"] if p["code"] == "P1"
    )

    bad_prefix = client.post(
        f"/api/pastures/{p1['id']}/images",
        files={"file": ("x.png", _green_png_bytes(), "image/png")},
        data={"image_code": "X9-A-01"},
    ).json()
    assert bad_prefix["code"] == 400 and "编号" in bad_prefix["message"]

    mismatch = client.post(
        f"/api/pastures/{p1['id']}/images",
        files={"file": ("x.png", _green_png_bytes(), "image/png")},
        data={"image_code": "P1-A-09", "plot_code": "B"},
    ).json()
    assert mismatch["code"] == 400 and "不一致" in mismatch["message"]

    ok_resp = client.post(
        f"/api/pastures/{p1['id']}/images",
        files={"file": ("x.png", _green_png_bytes(), "image/png")},
        data={"image_code": "P1-A-09"},
    ).json()
    assert ok_resp["code"] == 0
    assert ok_resp["data"]["image"]["plot_code"] == "A"  # 从编号解析分区
    assert ok_resp["data"]["image"]["coverage"] > 0.9  # 纯绿合成图
    assert ok_resp["data"]["pasture"]["image_count"] == p1["image_count"] + 1


def test_create_and_delete_pasture(client):
    created = client.post(
        "/api/pastures",
        json={"code": "P9", "name": "九号试验草场", "region": "南区", "area_mu": 45.0},
    ).json()
    assert created["code"] == 0
    pasture_id = created["data"]["id"]
    assert created["data"]["coverage"] is None  # 未登记照片 → 未分析
    assert created["data"]["recommended_stocking"] == 0.0

    duplicate = client.post("/api/pastures", json={"code": "P9", "name": "重复"}).json()
    assert duplicate["code"] == 400 and "已存在" in duplicate["message"]

    assert client.get(f"/api/pastures/{pasture_id}").json()["data"]["plots"] == []
    deleted = client.delete(f"/api/pastures/{pasture_id}").json()
    assert deleted["code"] == 0
    assert client.get(f"/api/pastures/{pasture_id}").json()["code"] == 404


def test_pasture_endpoints_404_for_unknown_id(client):
    assert client.get("/api/pastures/99999").json()["code"] == 404
    assert client.post("/api/pastures/99999/analyze").json()["code"] == 404
    assert client.get("/api/pastures/99999/trend").json()["code"] == 404
    assert client.delete("/api/pastures/99999").json()["code"] == 404


def test_analyze_single_registered_image(client):
    """单张分析端点：按编号复算该照片，结果与清单实测一致。"""
    p3 = next(
        p for p in client.get("/api/pastures").json()["data"]["pastures"] if p["code"] == "P3"
    )
    entry = MANIFEST_PASTURES["P3"]["images"][0]
    resp = client.post(f"/api/pastures/{p3['id']}/images/{entry['code']}/analyze").json()
    assert resp["code"] == 0
    image = resp["data"]["image"]
    assert image["image_code"] == entry["code"]
    assert abs(image["coverage"] - entry["coverage"]) < 0.01
    assert resp["data"]["pasture"]["coverage"] is not None

    missing = client.post(f"/api/pastures/{p3['id']}/images/P3-Z-99/analyze").json()
    assert missing["code"] == 404


def test_pasture_summary_includes_health_score(client):
    """草场清单/详情含多模态草情评分：0–100、含贡献分解与等级。"""
    pastures = client.get("/api/pastures").json()["data"]["pastures"]
    assert pastures
    for p in pastures:
        health = p.get("health")
        assert health is not None
        if p["coverage"] is None:
            assert health["score"] is None
            continue
        assert 0 <= health["score"] <= 100
        assert "coverage" in health["contributions"]
        assert health["level"] == p["alert_level"]  # 等级与全站告警口径一致


def test_pasture_trend_daily_series(client):
    p2 = next(
        p for p in client.get("/api/pastures").json()["data"]["pastures"] if p["code"] == "P2"
    )
    body = client.get(f"/api/pastures/{p2['id']}/trend?days=14").json()
    assert body["code"] == 0
    series = body["data"]["series"]
    assert len(series) == 14
    assert [s["day"] for s in series] == sorted(s["day"] for s in series)
    assert all({"day", "coverage", "greenness", "n"} == set(s) for s in series)
    assert all(0 <= s["coverage"] <= 1 for s in series)
    assert body["data"]["forecast"] is None  # 默认不预测


def test_pasture_trend_with_forecast(client):
    """Holt 趋势预测：7 个预测日紧接最后实测日，值在 [0,1]、置信带随步长加宽。"""
    p2 = next(
        p for p in client.get("/api/pastures").json()["data"]["pastures"] if p["code"] == "P2"
    )
    body = client.get(f"/api/pastures/{p2['id']}/trend?days=30&forecast_days=7").json()
    assert body["code"] == 0
    data = body["data"]
    fc = data["forecast"]
    assert fc is not None
    assert len(fc["days"]) == 7 and len(fc["values"]) == 7 and len(fc["band"]) == 7
    assert all(0 <= v <= 1 for v in fc["values"])
    assert fc["days"][0] > data["series"][-1]["day"]  # 预测日紧接最后实测日
    assert fc["band"] == sorted(fc["band"])  # 越远期越不确定
    assert isinstance(fc["trend"], float) and isinstance(fc["residual_std"], float)
