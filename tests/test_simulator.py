"""阶段 4.1：模拟数据生成器测试 —— 字段/范围/条数/确定性/可入库。"""
from __future__ import annotations

from datetime import datetime, timezone

from app.data import repository, simulator
from app.data.db import connect, init_db


# ---------------------------------------------------------------------------
# 环境序列
# ---------------------------------------------------------------------------

def test_generate_environment_fields_and_ranges():
    rows = simulator.generate_environment(hours=24, interval_hours=1, seed=42)
    assert len(rows) == 24
    for r in rows:
        assert set(r) == {"ts", "temperature", "humidity"}
        assert 0 <= r["temperature"] <= 45
        assert 0 <= r["humidity"] <= 100
        assert r["ts"].endswith("Z")  # ISO8601 UTC


def test_generate_environment_diurnal_shape():
    """温度峰值在午后、谷值在凌晨（昼夜曲线合理）。"""
    rows = simulator.generate_environment(hours=24, interval_hours=1, seed=42)
    temps = [r["temperature"] for r in rows]
    peak_hour = int(rows[temps.index(max(temps))]["ts"][11:13])
    trough_hour = int(rows[temps.index(min(temps))]["ts"][11:13])
    assert 10 <= peak_hour <= 17
    assert trough_hour <= 6


def test_environment_default_hourly_density():
    """默认 1 小时一点：48 点覆盖近 48 小时，供大屏精细时序图。"""
    rows = simulator.generate_environment(seed=42)
    assert len(rows) == 48
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    d0 = datetime.strptime(rows[0]["ts"], fmt)
    d1 = datetime.strptime(rows[1]["ts"], fmt)
    assert (d1 - d0).total_seconds() == 3600


# ---------------------------------------------------------------------------
# 覆盖度 / 羊数序列
# ---------------------------------------------------------------------------

def test_generate_coverage_series():
    rows = simulator.generate_coverage_series(days=30, seed=42)
    assert len(rows) == 30
    for r in rows:
        assert set(r) == {"ts", "coverage", "greenness"}
        assert 0 <= r["coverage"] <= 1
        assert 0 <= r["greenness"] <= 1


def test_generate_sheep_series():
    rows = simulator.generate_sheep_series(days=30, seed=42)
    assert len(rows) == 30
    for r in rows:
        assert isinstance(r["count"], int)
        assert r["count"] >= 0


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------

def test_series_deterministic_with_same_seed():
    end = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert simulator.generate_environment(hours=10, seed=7, end=end) == (
        simulator.generate_environment(hours=10, seed=7, end=end)
    )
    assert simulator.generate_coverage_series(days=10, seed=7, end=end) == (
        simulator.generate_coverage_series(days=10, seed=7, end=end)
    )
    assert simulator.generate_sheep_series(days=10, seed=7, end=end) == (
        simulator.generate_sheep_series(days=10, seed=7, end=end)
    )


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------

def test_populate_writes_rows(tmp_path):
    db = tmp_path / "sim.db"
    init_db(db)
    conn = connect(db)
    try:
        result = simulator.populate(conn, days=5, hours=6, seed=42)
        assert result == {"environment": 6, "coverage": 5, "sheep": 5}
        assert len(repository.list_environment_records(conn)) == 6
        assert len(repository.list_grassland_metrics(conn)) == 5
        assert len(repository.list_count_records(conn)) == 5
        # 历史时间戳已写入，升序查询返回时间有序序列
        env = repository.list_environment_records(conn, limit=6, ascending=True)
        ts_list = [r["created_at"] for r in env]
        assert ts_list == sorted(ts_list)
        assert env[0]["region"] == "东区"
    finally:
        conn.close()


def test_populate_if_empty_idempotent(tmp_path):
    db = tmp_path / "sim.db"
    init_db(db)
    conn = connect(db)
    try:
        assert simulator.populate_if_empty(conn, days=3, hours=4) is not None
        assert simulator.populate_if_empty(conn, days=3, hours=4) is None  # 已有数据不再播种
        assert len(repository.list_environment_records(conn)) == 4  # 未重复写入
    finally:
        conn.close()


def test_populate_if_empty_partial_fill(tmp_path):
    """已有真实数据的表不重复播种，空表补齐（如已有上传计数记录）。"""
    db = tmp_path / "sim.db"
    init_db(db)
    conn = connect(db)
    try:
        repository.insert_count_record(
            conn, region=None, count=9, per_class={"sheep": 9}, annotated_image=None
        )
        conn.commit()

        result = simulator.populate_if_empty(conn, days=3, hours=4)
        # sheep 表已有数据不播；草场与其编号照片按样例清单播种，并补演示决策记录
        assert set(result) == {"environment", "pastures", "pasture_metrics", "decisions"}
        assert result["environment"] == 4
        assert result["pastures"] == len(simulator.demo_pastures()) >= 2
        assert result["pasture_metrics"] == 3 * sum(
            len(p["images"]) for p in simulator.demo_pastures()
        )
        assert result["decisions"] == len(simulator.demo_pastures())
        assert len(repository.list_count_records(conn)) == 1  # 真实记录保留
        assert len(repository.list_environment_records(conn)) == 4
        assert len(repository.list_grassland_metrics(conn)) == result["pasture_metrics"]
    finally:
        conn.close()


def test_populate_seeds_demo_decisions_with_events(tmp_path):
    """演示决策记录：带问题/工具调用/决策与完整思考事件（历史对话可回放）。"""
    import json

    db = tmp_path / "sim.db"
    init_db(db)
    conn = connect(db)
    try:
        simulator.populate_if_empty(conn, days=3, hours=4)
        logs = repository.list_decision_logs(conn, limit=10)
        assert len(logs) == len(simulator.demo_pastures()) >= 2

        levels = set()
        for row in logs:
            assert row["question"].endswith("草情如何？能放多少羊？")
            assert json.loads(row["tool_calls"]) == [
                "grassland_coverage", "sheep_count", "carrying_capacity",
            ]
            decision = json.loads(row["decision_json"])
            levels.add(decision["alert_level"])
            assert decision["recommended_actions"]

            events = json.loads(row["events"])
            assert events[0]["type"] == "thinking"
            assert events[-1]["type"] == "answer"
            assert {"tool_call", "tool_result", "analysis"} <= {e["type"] for e in events}
            assert all(isinstance(e.get("ts"), int) for e in events)
            # 与真实 SSE 流同序：每个工具都是 调用 → 结果 → 确定性解读
            types = [e["type"] for e in events]
            for i, etype in enumerate(types):
                if etype == "analysis":
                    assert types[i - 1] == "tool_result"
                if etype == "tool_result":
                    assert types[i - 1] == "tool_call"
                if etype == "tool_call":
                    assert types[i + 1] == "tool_result"

        assert levels == {"normal", "warning", "danger"}  # 三个草场覆盖三档
        assert simulator.populate_if_empty(conn, days=3, hours=4) is None  # 幂等
    finally:
        conn.close()


def test_populate_seeds_pastures_with_numbered_photos(tmp_path):
    """草场播种：编号照片（草场-分区-序号）登记完整，覆盖度记录按草场/分区/图片编号归属。"""
    db = tmp_path / "sim.db"
    init_db(db)
    conn = connect(db)
    try:
        simulator.populate_if_empty(conn, days=3, hours=4)
        pastures = repository.list_pastures(conn)
        assert [p["code"] for p in pastures] == ["P1", "P2", "P3"]

        for pasture in pastures:
            images = repository.list_pasture_images(conn, pasture["id"])
            assert images, pasture["code"]
            for image in images:
                assert image["image_code"].startswith(pasture["code"] + "-")
                assert image["plot_code"]
                assert image["image_path"].startswith("/samples/grassland/")

        metrics = repository.list_metrics_for_pasture(conn, pastures[0]["id"], limit=100)
        assert metrics
        assert {m["region"] for m in metrics} == {"P1"}
        assert all(m["plot_code"] and m["image_code"] for m in metrics)
        # 同日只播种一次（幂等）
        assert simulator.populate_if_empty(conn, days=3, hours=4) is None
    finally:
        conn.close()


def test_generate_pasture_coverage_series_deterministic():
    """草场覆盖度序列固定种子可复现，且围绕草场基准草情波动。"""
    from datetime import datetime, timezone

    pastures = simulator.demo_pastures()
    assert pastures
    spec = pastures[0]
    # 传入固定 end：否则两次调用取「当前时间」，时间戳会有微秒级差异（测试会抖动）
    anchor = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
    a = simulator.generate_pasture_coverage_series(spec, days=5, seed=42, end=anchor)
    b = simulator.generate_pasture_coverage_series(spec, days=5, seed=42, end=anchor)
    assert a == b
    assert len(a) == 5 * len(spec["images"])
    values = [p["coverage"] for p in a]
    assert all(0 <= v <= 1 for v in values)
    assert abs(sum(values) / len(values) - spec["base"]) < 0.1
