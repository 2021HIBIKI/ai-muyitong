"""阶段 0.3：存储模块测试 —— 建表幂等、CRUD 正确。"""
from __future__ import annotations

import pytest

from app.data import db
from app.data import repository


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    c = db.connect(db_path)
    yield c
    c.close()


def test_init_db_idempotent(tmp_path):
    """建表幂等：连续初始化两次不抛错。"""
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    db.init_db(db_path)


def test_init_db_migrates_legacy_grassland_metrics(tmp_path):
    """旧库（覆盖度表无草场关联列）执行 init_db 后自动补齐新列且保留既有数据。"""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.execute(
        "CREATE TABLE grassland_metrics ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
        " region TEXT, coverage REAL, greenness REAL)"
    )
    legacy.execute(
        "INSERT INTO grassland_metrics (region, coverage, greenness) VALUES ('东区', 0.42, 0.1)"
    )
    legacy.commit()
    legacy.close()

    db.init_db(db_path)
    conn = db.connect(db_path)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(grassland_metrics)")}
        assert {"pasture_id", "plot_code", "image_code", "image_path"} <= columns
        row = conn.execute("SELECT * FROM grassland_metrics").fetchone()
        assert row["coverage"] == 0.42 and row["pasture_id"] is None  # 老数据保留

        tables = {
            r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"pastures", "pasture_images"} <= tables
    finally:
        conn.close()

    db.init_db(db_path)  # 迁移幂等


def test_init_db_migrates_legacy_alerts_with_pasture_id(tmp_path):
    """旧库（告警表无草场归属列）执行 init_db 后自动补齐 pasture_id 且保留既有告警。"""
    import sqlite3

    db_path = tmp_path / "legacy_alerts.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.execute(
        "CREATE TABLE alerts ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),"
        " level TEXT NOT NULL, title TEXT NOT NULL, message TEXT,"
        " metrics_json TEXT, acknowledged INTEGER NOT NULL DEFAULT 0)"
    )
    legacy.execute(
        "INSERT INTO alerts (level, title, message) VALUES ('warning', '草场预警告警', '历史告警')"
    )
    legacy.commit()
    legacy.close()

    db.init_db(db_path)
    conn = db.connect(db_path)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(alerts)")}
        assert "pasture_id" in columns
        row = conn.execute("SELECT * FROM alerts").fetchone()
        assert row["title"] == "草场预警告警" and row["pasture_id"] is None  # 老数据保留

        # 逐场查询与写入在迁移后可用
        assert repository.list_alerts(conn, pasture_id=1) == []
        repository.insert_alert(
            conn, level="danger", title="P3 北坡轮休草场 · 危险告警", pasture_id=3
        )
        conn.commit()
        assert len(repository.list_alerts(conn, pasture_id=3)) == 1
        assert len(repository.list_alerts(conn, limit=10)) == 2
    finally:
        conn.close()

    db.init_db(db_path)  # 迁移幂等


def test_insert_and_query_count(conn):
    rid = repository.insert_count_record(
        conn, region="东区", count=42, per_class={"sheep": 42}, annotated_image="b64img"
    )
    assert rid is not None
    rows = repository.list_count_records(conn)
    assert len(rows) == 1
    assert rows[0]["count"] == 42
    assert rows[0]["region"] == "东区"
    assert rows[0]["created_at"]  # ISO8601 时间列非空


def test_insert_all_tables(conn):
    repository.insert_count_record(conn, region="东区", count=1, per_class={}, annotated_image="")
    repository.insert_grassland_metric(conn, region="东区", coverage=0.7, greenness=0.5)
    repository.insert_carrying_result(
        conn, params={"yield": 100}, recommended_stocking=10, rotation_plan="A区"
    )
    repository.insert_decision_log(
        conn, question="草够吗", tool_calls=["grassland_coverage"], decision_json={"alert_level": "normal"}
    )
    assert len(repository.list_count_records(conn)) == 1
    assert len(repository.list_grassland_metrics(conn)) == 1
    assert len(repository.list_carrying_results(conn)) == 1
    assert len(repository.list_decision_logs(conn)) == 1


def test_per_class_roundtrip(conn):
    """per_class 字典以 JSON 存储，可正确解析。"""
    repository.insert_count_record(
        conn, region="西区", count=3, per_class={"sheep": 2, "cow": 1}, annotated_image=""
    )
    row = repository.list_count_records(conn)[0]
    import json

    assert json.loads(row["per_class"]) == {"sheep": 2, "cow": 1}


def test_ascending_returns_latest_n(conn):
    """ascending=True 语义：返回「最近 limit 条」并升序（趋势图取数正确性回归）。"""
    for i in range(5):
        repository.insert_grassland_metric(conn, region="东区", coverage=0.1 * (i + 1), greenness=0.1)
    conn.commit()

    rows = repository.list_grassland_metrics(conn, limit=3, ascending=True)
    assert len(rows) == 3
    assert [round(r["coverage"], 2) for r in rows] == [0.3, 0.4, 0.5]  # 最近 3 条，时间升序
    # 与降序首条一致（都是最新记录）
    latest = repository.list_grassland_metrics(conn, limit=1)[0]
    assert latest["id"] == rows[-1]["id"]


def test_decision_log_events_roundtrip(conn):
    """决策日志保存思考事件，可按 id 取回（历史对话回放思考过程）。"""
    import json

    events = [
        {"type": "thinking", "text": "分析问题…", "ts": 5},
        {"type": "answer", "decision": {"alert_level": "normal"}, "ts": 1200},
    ]
    log_id = repository.insert_decision_log(
        conn,
        question="草够吗？",
        tool_calls=["grassland_coverage"],
        decision_json={"alert_level": "normal"},
        events=events,
    )
    conn.commit()

    row = repository.get_decision_log(conn, log_id)
    assert row["question"] == "草够吗？"
    assert json.loads(row["events"]) == events
    # 未保存事件的记录（同步 /chat 路径）events 为空
    plain_id = repository.insert_decision_log(
        conn, question="x", tool_calls=[], decision_json={"alert_level": "normal"}
    )
    conn.commit()
    assert repository.get_decision_log(conn, plain_id)["events"] is None
    assert repository.get_decision_log(conn, 99999) is None


def test_alerts_crud(conn):
    """阶段 7.1 告警表：插入 / 列表（默认降序、可升序）/ 按 id 取 / 确认。"""
    import json

    first = repository.insert_alert(
        conn,
        level="warning",
        title="覆盖度预警",
        message="覆盖度 45.0% 处于预警区间",
        metrics={"coverage": 0.45, "source": "pastures"},
    )
    second = repository.insert_alert(
        conn, level="danger", title="覆盖度危险", message="覆盖度 20.0% 处于危险区间"
    )
    conn.commit()

    rows = repository.list_alerts(conn, limit=10)
    assert [r["id"] for r in rows] == [second, first]  # 最新在前
    assert json.loads(rows[1]["metrics_json"]) == {"coverage": 0.45, "source": "pastures"}
    assert rows[0]["acknowledged"] == 0

    ascending = repository.list_alerts(conn, limit=10, ascending=True)
    assert [r["id"] for r in ascending] == [first, second]

    assert repository.acknowledge_alert(conn, first) is True
    conn.commit()
    assert repository.get_alert(conn, first)["acknowledged"] == 1
    assert repository.get_alert(conn, 99999) is None
    assert repository.acknowledge_alert(conn, 99999) is False
