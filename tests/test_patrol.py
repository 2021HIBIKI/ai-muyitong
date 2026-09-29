"""阶段 7.2：巡检评估纯函数与巡检轮次（确定性，不依赖网络/权重/GPU）。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from app.agent import patrol
from app.data import repository
from app.data.db import connect, init_db

NOW = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


def _metrics(coverage, source="pasture", greenness=0.2):
    return {"coverage": coverage, "greenness": greenness, "source": source}


def _pasture_metrics(code, coverage, pid, name=None):
    """注入用的单草场巡检指标（编号/名称/归属齐全）。"""
    return {
        "pasture_id": pid,
        "pasture_code": code,
        "pasture_name": name or f"{code} 示范草场",
        "coverage": coverage,
        "greenness": 0.2,
        "alert_level": None,
        "plot_count": 3,
        "analyzed_plot_count": 3,
        "image_count": 3,
        "source": "pasture",
    }


def _decide(coverage, prev_level, cooldown=300.0, last_alert_at=None):
    return patrol.evaluate(
        _metrics(coverage),
        prev_level=prev_level,
        last_alert_at=last_alert_at,
        now=NOW,
        cooldown_seconds=cooldown,
    )


# ---------------------------------------------------------------------------
# 评估规则
# ---------------------------------------------------------------------------

def test_first_non_normal_alerts_and_normal_does_not():
    """首次巡检：非 normal 必发；normal 不发。"""
    first = _decide(0.45, prev_level=None)
    assert first.should_alert and first.reason == "first"
    assert first.level == "warning" and "预警区间" in first.message

    quiet = _decide(0.80, prev_level=None)
    assert not quiet.should_alert and quiet.reason == "normal"
    assert quiet.level == "normal"


def test_escalation_always_alerts_even_in_cooldown():
    """升级必发（不受冷却限制）。"""
    decision = _decide(0.20, prev_level="warning", last_alert_at=NOW - timedelta(seconds=5))
    assert decision.should_alert and decision.escalated and decision.reason == "escalated"
    assert decision.level == "danger"
    assert "升级" in decision.message


def test_same_level_suppressed_within_cooldown_then_reminds():
    """同级：冷却期内抑制，冷却期外补发提醒。"""
    suppressed = _decide(0.45, prev_level="warning", last_alert_at=NOW - timedelta(seconds=30))
    assert not suppressed.should_alert and suppressed.reason == "cooldown_suppressed"
    assert "冷却期" in suppressed.message

    reminder = _decide(0.45, prev_level="warning", last_alert_at=NOW - timedelta(seconds=600))
    assert reminder.should_alert and reminder.reason == "repeat_reminder"

    # 同级 normal 永不告警
    quiet = _decide(0.90, prev_level="normal", last_alert_at=None)
    assert not quiet.should_alert and quiet.reason == "normal"


def test_recovery_sends_notice():
    """降级发一条恢复通知。"""
    decision = _decide(0.75, prev_level="danger", last_alert_at=NOW - timedelta(seconds=10))
    assert decision.should_alert and decision.recovered and decision.reason == "recovered"
    assert decision.level == "normal"
    assert "恢复" in decision.title or "恢复" in decision.message


def test_message_includes_multimodal_health_score():
    """巡检文案追加多模态草情评分（有评分时）；无评分时保持原样。"""
    with_health = patrol.evaluate(
        {"coverage": 0.45, "source": "pasture",
         "health": {"score": 72.5, "contributions": {}}},
        prev_level=None, last_alert_at=None, now=NOW, cooldown_seconds=300.0,
    )
    assert "多模态草情评分 72/100" in with_health.message

    without = patrol.evaluate(
        {"coverage": 0.45, "source": "pasture"},
        prev_level=None, last_alert_at=None, now=NOW, cooldown_seconds=300.0,
    )
    assert "多模态草情评分" not in without.message


def test_threshold_shared_with_orchestrator():
    """阈值口径复用 alert_level_from_coverage（不重写）：0.60 / 0.35 边界一致。"""
    assert _decide(0.60, prev_level=None).level == "normal"
    assert _decide(0.35, prev_level=None).level == "warning"
    assert _decide(0.34, prev_level=None).level == "danger"


def test_missing_coverage_means_normal_and_message_notes_no_data():
    decision = patrol.evaluate(
        {"coverage": None, "source": None},
        prev_level=None,
        last_alert_at=None,
        now=NOW,
        cooldown_seconds=300.0,
    )
    assert decision.level == "normal" and not decision.should_alert
    assert "暂无覆盖度数据" in decision.message


# ---------------------------------------------------------------------------
# 巡检轮次（DB 落库 + 上一等级取自 DB）
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "patrol.db"))
    monkeypatch.setattr(settings, "patrol_image_path", "")
    init_db()
    connection = connect()
    yield connection
    connection.close()


def test_run_patrol_inserts_alert_and_cooldown_blocks_repeat(conn):
    """落库：首次预警写入；同级再跑（冷却期内）不写入。"""
    first = patrol.run_patrol(conn, now=NOW, metrics=_metrics(0.45))
    assert first["summary"] == {
        "total": 1, "alerted": 1, "suppressed": 0, "normal": 0, "recovered": 0,
    }
    assert first["results"][0]["should_alert"] and first["results"][0]["reason"] == "first"
    assert first["alerts"][0]["level"] == "warning"
    assert json.loads(
        repository.list_alerts(conn, limit=1)[0]["metrics_json"]
    )["coverage"] == 0.45

    second = patrol.run_patrol(conn, now=NOW + timedelta(seconds=60), metrics=_metrics(0.45))
    assert not second["results"][0]["should_alert"]
    assert second["results"][0]["reason"] == "cooldown_suppressed"
    assert second["alerts"] == [] and second["summary"]["suppressed"] == 1
    assert len(repository.list_alerts(conn, limit=10)) == 1  # 未新增


def test_run_patrol_escalation_after_cooldown(conn):
    """上一等级来自 DB：升级即便紧接上次告警也会再发。"""
    patrol.run_patrol(conn, now=NOW, metrics=_metrics(0.45))
    escalated = patrol.run_patrol(
        conn, now=NOW + timedelta(seconds=10), metrics=_metrics(0.20)
    )
    assert escalated["results"][0]["escalated"]
    assert escalated["results"][0]["reason"] == "escalated"
    assert len(repository.list_alerts(conn, limit=10)) == 2


def test_run_patrol_alerts_each_pasture_independently(conn):
    """一轮巡检逐草场独立判定：各自等级/告警，未告警的草场也在结果里。"""
    payload = patrol.run_patrol(
        conn,
        now=NOW,
        metrics=[
            _pasture_metrics("P1", 0.80, 1, "东区示范牧场"),
            _pasture_metrics("P2", 0.50, 2, "西区生态牧场"),
            _pasture_metrics("P3", 0.20, 3, "北坡轮休草场"),
        ],
    )
    assert payload["summary"] == {
        "total": 3, "alerted": 2, "suppressed": 0, "normal": 1, "recovered": 0,
    }
    by_code = {r["pasture_code"]: r for r in payload["results"]}
    assert set(by_code) == {"P1", "P2", "P3"}  # 每个草场都有结果（含未告警）
    assert by_code["P1"]["level"] == "normal" and not by_code["P1"]["should_alert"]
    assert by_code["P2"]["level"] == "warning" and by_code["P2"]["should_alert"]
    assert by_code["P3"]["level"] == "danger"
    assert by_code["P3"]["coverage"] == 0.20  # 逐场各自的覆盖度，无跨草场平均

    assert {a["level"] for a in payload["alerts"]} == {"warning", "danger"}
    for alert in payload["alerts"]:
        assert alert["pasture_id"] and alert["pasture_code"]
        assert alert["pasture_code"] in alert["title"]  # 告警标题标明是哪个草场
        assert "本草场编号照片聚合" in alert["message"]  # 来源为本草场照片（非跨场聚合）
    assert by_code["P3"]["alert"]["level"] == "danger"


def test_run_patrol_cooldown_and_escalation_are_per_pasture(conn):
    """冷却与升级按草场独立：一个草场被抑制不影响其他草场告警。"""
    patrol.run_patrol(
        conn, now=NOW, metrics=[_pasture_metrics("P2", 0.50, 2), _pasture_metrics("P3", 0.45, 3)]
    )
    # P2 同级且在冷却期 → 抑制；同轮 P3 升级 → 必发（不受 P2 影响）
    second = patrol.run_patrol(
        conn,
        now=NOW + timedelta(seconds=30),
        metrics=[_pasture_metrics("P2", 0.50, 2), _pasture_metrics("P3", 0.20, 3)],
    )
    by_code = {r["pasture_code"]: r for r in second["results"]}
    assert by_code["P2"]["reason"] == "cooldown_suppressed" and not by_code["P2"]["should_alert"]
    assert by_code["P3"]["escalated"] and by_code["P3"]["alert"]["level"] == "danger"
    assert len(repository.list_alerts(conn, pasture_id=2)) == 1
    assert len(repository.list_alerts(conn, pasture_id=3)) == 2


def test_run_patrol_advice_only_on_escalation(conn):
    """LLM 建议只在升级时调用（同级不调，省配额）。"""
    calls: list[dict] = []

    def advice(metrics, decision):
        calls.append(metrics)
        return "建议：先禁牧观察一周。"

    patrol.run_patrol(conn, now=NOW, metrics=_metrics(0.45), advice_fn=advice)
    assert calls == []  # 首次预警不调 LLM

    patrol.run_patrol(conn, now=NOW + timedelta(seconds=10), metrics=_metrics(0.20), advice_fn=advice)
    assert len(calls) == 1  # 升级才调
    latest = repository.list_alerts(conn, limit=1)[0]
    assert "AI 建议：建议：先禁牧观察一周。" in latest["message"]


# ---------------------------------------------------------------------------
# 指标采集
# ---------------------------------------------------------------------------

def test_collect_metrics_returns_each_pasture_independently(conn):
    """采集按草场独立：每个草场一条（本草场分区照片聚合），没有跨草场平均。"""
    p1 = repository.insert_pasture(conn, code="P1", name="东区示范牧场", area_mu=120.0)
    p2 = repository.insert_pasture(conn, code="P2", name="西区生态牧场", area_mu=90.0)
    for plot, coverage in (("A", 0.80), ("B", 0.60)):
        repository.insert_grassland_metric(
            conn, region="P1", coverage=coverage, greenness=0.2, pasture_id=p1,
            plot_code=plot, image_code=f"P1-{plot}-01",
        )
    repository.insert_grassland_metric(
        conn, region="P2", coverage=0.40, greenness=0.1, pasture_id=p2,
        plot_code="A", image_code="P2-A-01",
    )
    conn.commit()

    items = patrol.collect_metrics(conn)
    assert [i["pasture_code"] for i in items] == ["P1", "P2"]
    by_code = {i["pasture_code"]: i for i in items}
    assert by_code["P1"]["coverage"] == 0.7  # 本草场 A/B 分区均值（0.80 + 0.60）/ 2
    assert by_code["P2"]["coverage"] == 0.4  # 各自独立，不受 P1 影响
    assert by_code["P1"]["pasture_id"] == p1 and by_code["P1"]["source"] == "pasture"
    assert by_code["P1"]["pasture_name"] == "东区示范牧场"
    assert by_code["P2"]["analyzed_plot_count"] == 1 and by_code["P2"]["plot_count"] == 1
    # 任何一条都不含「跨草场平均」的痕迹：草场各自返回自己的覆盖度
    assert [i["coverage"] for i in items] == [0.7, 0.4]


def test_collect_metrics_falls_back_to_latest_metric(conn):
    """库中尚无草场：退回最新覆盖度记录（单点位，仍非跨草场平均）。"""
    repository.insert_grassland_metric(conn, region="东区", coverage=0.33, greenness=0.1)
    conn.commit()
    items = patrol.collect_metrics(conn)
    assert len(items) == 1
    assert items[0]["source"] == "metric" and items[0]["coverage"] == 0.33


def test_collect_metrics_without_data(conn):
    items = patrol.collect_metrics(conn)
    assert len(items) == 1
    assert items[0]["coverage"] is None and items[0]["source"] is None


def test_collect_metrics_from_patrol_image(conn, tmp_path, monkeypatch):
    """库中无草场且配置了巡检点位图片时按图片实测覆盖度（模拟固定摄像头）。"""
    from app.config import settings

    dark = np.zeros((60, 60, 3), dtype=np.uint8)  # 近乎无植被
    path = tmp_path / "patrol.jpg"
    cv2.imwrite(str(path), dark)
    monkeypatch.setattr(settings, "patrol_image_path", str(path))

    items = patrol.collect_metrics(conn)
    assert len(items) == 1
    assert items[0]["source"] == "image"
    assert items[0]["coverage"] < 0.05


def test_collect_metrics_unreadable_image_raises(conn, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "patrol_image_path", "data/samples/not-exist.jpg")
    with pytest.raises(ValueError):
        patrol.collect_metrics(conn)


def test_patrol_image_is_fallback_only_when_no_pastures(conn, tmp_path, monkeypatch):
    """有草场时按草场逐个巡检（点位图片只作无草场时的兜底）。"""
    from app.config import settings

    dark = np.zeros((60, 60, 3), dtype=np.uint8)
    path = tmp_path / "patrol.jpg"
    cv2.imwrite(str(path), dark)
    monkeypatch.setattr(settings, "patrol_image_path", str(path))

    pasture_id = repository.insert_pasture(conn, code="P1", name="东区示范牧场", area_mu=120.0)
    repository.insert_grassland_metric(
        conn, region="P1", coverage=0.72, greenness=0.3, pasture_id=pasture_id,
        plot_code="A", image_code="P1-A-01",
    )
    conn.commit()

    items = patrol.collect_metrics(conn)
    assert [i["pasture_code"] for i in items] == ["P1"]
    assert items[0]["coverage"] == 0.72  # 用草场照片聚合，而非点位图片实测值


# ---------------------------------------------------------------------------
# 载荷
# ---------------------------------------------------------------------------

def test_alert_payload_shape(conn):
    patrol.run_patrol(conn, now=NOW, metrics=_pasture_metrics("P3", 0.20, 3, "北坡轮休草场"))
    payload = patrol.alert_payload(repository.list_alerts(conn, limit=1)[0])
    assert set(payload) == {
        "id", "created_at", "level", "title", "message", "metrics", "acknowledged",
        "pasture_id", "pasture_code", "pasture_name",
    }
    assert payload["level"] == "danger" and payload["acknowledged"] is False
    assert payload["metrics"]["coverage"] == 0.20
    assert payload["pasture_id"] == 3 and payload["pasture_code"] == "P3"
    assert payload["pasture_name"] == "北坡轮休草场"
    assert payload["title"] == "P3 北坡轮休草场 · 危险告警"


def test_list_alerts_filters_by_pasture(conn):
    """告警按草场可查：逐场冷却/升级判定依赖本场历史。"""
    patrol.run_patrol(
        conn, now=NOW, metrics=[_pasture_metrics("P1", 0.50, 1), _pasture_metrics("P2", 0.20, 2)]
    )
    assert len(repository.list_alerts(conn, pasture_id=1)) == 1
    assert repository.list_alerts(conn, pasture_id=1)[0]["level"] == "warning"
    assert repository.list_alerts(conn, pasture_id=2)[0]["level"] == "danger"
    assert repository.list_alerts(conn, pasture_id=99) == []


def test_parse_ts_handles_iso_and_bad_input():
    parsed = patrol.parse_ts("2026-09-17T12:00:00.000Z")
    assert parsed == datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
    assert patrol.parse_ts(None) is None
    assert patrol.parse_ts("not-a-time") is None


# ---------------------------------------------------------------------------
# 阶段 7.3：调度器与环境启停
# ---------------------------------------------------------------------------

def test_patrol_loop_runs_rounds_and_cancels_cleanly(tmp_path, monkeypatch):
    """调度器：跑 ≥2 轮后取消 —— 干净退出、无遗留 asyncio 任务。"""
    import asyncio
    import contextlib

    from app.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "loop.db"))
    init_db()

    rounds = {"n": 0}
    real_round = patrol._patrol_round

    def counting_round():
        rounds["n"] += 1
        return real_round()

    monkeypatch.setattr(patrol, "_patrol_round", counting_round)

    async def main():
        task = asyncio.create_task(patrol.patrol_loop(interval_seconds=0.01))
        await asyncio.sleep(0.1)
        assert rounds["n"] >= 2  # 至少跑了两个轮次
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert pending == []  # 无遗留任务（否则 pytest 会报 Task was destroyed...）

    asyncio.run(main())


def test_patrol_loop_survives_round_failure(monkeypatch, tmp_path):
    """单轮异常不终止巡检（下一轮照常执行）。"""
    import asyncio
    import contextlib

    from app.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "loop2.db"))
    init_db()

    calls = {"n": 0}

    def flaky_round():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"alert": None}

    monkeypatch.setattr(patrol, "_patrol_round", flaky_round)

    async def main():
        task = asyncio.create_task(patrol.patrol_loop(interval_seconds=0.01))
        await asyncio.sleep(0.08)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert calls["n"] >= 2  # 第一轮抛错后仍在继续跑


def test_start_stop_patrol_task_respects_config(monkeypatch):
    """启停：默认关闭不创建任务；开启时可创建并干净取消。"""
    import asyncio
    import contextlib

    from app.config import settings

    monkeypatch.setattr(patrol, "_patrol_round", lambda: {"alert": None})

    async def main():
        monkeypatch.setattr(settings, "patrol_enabled", False)
        assert await patrol.start_patrol_task() is None

        monkeypatch.setattr(settings, "patrol_enabled", True)
        monkeypatch.setattr(settings, "patrol_interval_seconds", 30.0)
        task = await patrol.start_patrol_task()
        assert task is not None and not task.done()
        await patrol.stop_patrol_task(task)
        assert task.cancelled()
        await patrol.stop_patrol_task(None)  # 空任务安全

    asyncio.run(main())
