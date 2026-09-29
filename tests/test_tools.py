"""阶段 3.2：工具注册表测试 —— schema 合法、各工具返回正确类型。"""
from __future__ import annotations

import cv2
import numpy as np

from app.agent import tools
from app.agent.tools import (
    TOOLS,
    dispatch_tool,
    tool_carrying_capacity,
    tool_grassland_coverage,
    tool_schemas,
    tool_sheep_count,
    tool_query_history,
)
from app.data.db import init_db
from app.vision.detector import Detection


def _green_image(tmp_path, name="grass.png"):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:] = (0, 160, 0)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return str(p)


# ---------------------------------------------------------------------------
# schema 合法性
# ---------------------------------------------------------------------------

def test_tool_schemas_structure():
    schemas = tool_schemas()
    assert len(schemas) == len(TOOLS)
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"]
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params.get("properties", {}), dict)


def test_tool_schemas_valid_json_schema_types():
    valid_types = {"object", "string", "number", "integer", "boolean", "array"}
    for s in tool_schemas():
        for prop in s["function"]["parameters"].get("properties", {}).values():
            assert prop["type"] in valid_types


def test_tool_schemas_required_keys_exist():
    cap = TOOLS["carrying_capacity"]
    for req in cap.parameters.get("required", []):
        assert req in cap.parameters["properties"]
    # 只要求用户明确给出的参数；利用率/放牧天数缺省（防止模型臆造参数）
    assert cap.parameters["required"] == ["yield_per_area", "area"]
    assert "缺省" in cap.parameters["properties"]["utilization"]["description"]
    assert "缺省" in cap.parameters["properties"]["grazing_days"]["description"]


# ---------------------------------------------------------------------------
# 工具返回类型
# ---------------------------------------------------------------------------

def test_tool_grassland_coverage(tmp_path):
    result = tool_grassland_coverage(_green_image(tmp_path))
    assert isinstance(result["coverage"], float)
    assert isinstance(result["greenness"], float)
    assert 0 <= result["coverage"] <= 1


def test_tool_carrying_capacity():
    result = tool_carrying_capacity(
        yield_per_area=1000.0, area=100.0, utilization=0.5, grazing_days=100.0
    )
    assert isinstance(result["recommended_stocking"], (int, float))
    assert isinstance(result["rotation_plan"], str)


def test_tool_sheep_count_mocked(tmp_path, monkeypatch):
    class _MockDetector:
        def detect(self, img):
            return [
                Detection((0, 0, 5, 5), 0, "sheep", 0.9),
                Detection((6, 6, 10, 10), 0, "sheep", 0.8),
            ]

    monkeypatch.setattr(tools, "detector", _MockDetector())
    result = tool_sheep_count(_green_image(tmp_path))
    assert result["count"] == 2
    assert result["per_class"] == {"sheep": 2}


def test_tool_query_history(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    init_db()

    from app.data.db import connect
    from app.data import repository

    conn = connect()
    repository.insert_grassland_metric(conn, region="东区", coverage=0.7, greenness=0.5)
    conn.commit()
    conn.close()

    result = tool_query_history("coverage", limit=5)
    assert result["entity"] == "coverage"
    assert result["count"] == 1
    assert result["records"][0]["coverage"] == 0.7


# ---------------------------------------------------------------------------
# 草场口径（多草场：编号照片集合 → 草场状况）
# ---------------------------------------------------------------------------

def _seed_two_pastures(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    init_db()

    from app.data import repository
    from app.data.db import connect

    conn = connect()
    for code, name, coverage, area in (
        ("P1", "东区示范牧场", 0.8, 120.0),
        ("P2", "西区生态牧场", 0.4, 90.0),
    ):
        pasture_id = repository.insert_pasture(conn, code=code, name=name, area_mu=area)
        repository.insert_pasture_image(
            conn, pasture_id=pasture_id, plot_code="A",
            image_code=f"{code}-A-01", image_path="/samples/grassland/x.jpg",
        )
        repository.insert_grassland_metric(
            conn, region=code, coverage=coverage, greenness=0.2, pasture_id=pasture_id,
            plot_code="A", image_code=f"{code}-A-01",
        )
    conn.commit()
    conn.close()


def test_tool_pasture_status_all_and_single(tmp_path, monkeypatch):
    """草场状况工具：缺省返回全部草场概览；按编号/名称/整句问题定位单个草场。"""
    _seed_two_pastures(tmp_path, monkeypatch)

    overview = tools.tool_pasture_status()
    assert overview["count"] == 2
    assert [p["code"] for p in overview["pastures"]] == ["P1", "P2"]
    assert overview["pastures"][0]["coverage"] == 0.8

    single = tools.tool_pasture_status("P1")
    assert single["pasture"]["name"] == "东区示范牧场"
    assert single["pasture"]["plots"][0]["image_codes"] == ["P1-A-01"]

    by_name = tools.tool_pasture_status("西区生态牧场现在适合放牧吗？")  # 整句问题
    assert by_name["pasture"]["code"] == "P2"

    unmatched = tools.tool_pasture_status("P9")
    assert unmatched["matched"] is False and "未匹配" in unmatched["note"]


def test_tool_grassland_coverage_lists_pastures_independently(tmp_path, monkeypatch):
    """未指定图像时返回**各草场独立**覆盖度（不做跨草场平均）；指定图像仍走图像分析。"""
    _seed_two_pastures(tmp_path, monkeypatch)

    result = tools.tool_grassland_coverage()
    assert result["source"] == "pastures"
    assert "coverage" not in result  # 跨草场没有「一个覆盖度」
    assert result["pasture_count"] == 2
    assert result["total_area_mu"] == 210.0
    by_code = {p["code"]: p for p in result["pastures"]}
    assert set(by_code) == {"P1", "P2"}
    assert by_code["P1"]["coverage"] == 0.8
    assert by_code["P2"]["coverage"] == 0.4
    assert by_code["P2"]["alert_level"] == "warning"
    # 可加量（载畜量）才求和
    assert result["recommended_stocking_sum"] == round(
        sum(p["recommended_stocking"] for p in result["pastures"]), 2
    )

    image_result = tools.tool_grassland_coverage(_green_image(tmp_path, "explicit.png"))
    assert "source" not in image_result  # 指定图像 → 单图分析
    assert image_result["coverage"] > 0.9


def test_tool_grassland_coverage_falls_back_without_pastures(tmp_path, monkeypatch):
    """库中无草场数据时退回样例图分析（兼容未建立草场的库）。"""
    from app.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    init_db()

    result = tools.tool_grassland_coverage()
    assert 0 <= result["coverage"] <= 1
    assert result.get("source") is None


# ---------------------------------------------------------------------------
# 分发
# ---------------------------------------------------------------------------

def test_dispatch_unknown_tool_returns_error():
    result = dispatch_tool("not_a_tool", {})
    assert "error" in result


def test_dispatch_carrying_capacity():
    result = dispatch_tool(
        "carrying_capacity",
        {"yield_per_area": 400, "area": 10, "utilization": 1.0, "grazing_days": 10},
    )
    assert result["recommended_stocking"] == 200.0
