"""阶段 3.3：Agent 编排器测试 —— 工具分发、循环终止、决策 JSON、规则降级。"""
from __future__ import annotations

import pytest

from app.agent.orchestrator import AgentOrchestrator, DecisionJSON, alert_level_from_coverage


def _fake_dispatch(name, arguments):
    if name == "grassland_coverage":
        return {"coverage": 0.7, "greenness": 0.5}
    if name == "carrying_capacity":
        return {"recommended_stocking": 250.0, "rotation_plan": "划分 3 个轮牧小区轮换放牧"}
    if name == "sheep_count":
        return {"count": 30, "per_class": {"sheep": 30}}
    if name == "pasture_status":
        return {
            "pasture": {
                "code": "P1",
                "name": "东区示范牧场",
                "area_mu": 120.0,
                "coverage": 0.80,
                "greenness": 0.2,
                "alert_level": "normal",
                "plot_count": 3,
                "analyzed_plot_count": 3,
                "image_count": 4,
                "theoretical": 800.0,
                "recommended_stocking": 800.0,
            }
        }
    return {"error": f"unknown tool {name}"}


class _FakeLLM:
    enabled = True

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "fallback": False,
                "content": None,
                "tool_calls": [
                    {"name": "grassland_coverage", "arguments": {}, "id": "c1"},
                    {
                        "name": "carrying_capacity",
                        "arguments": {
                            "yield_per_area": 1000,
                            "area": 100,
                            "utilization": 0.5,
                            "grazing_days": 100,
                        },
                        "id": "c2",
                    },
                ],
            }
        return {"fallback": False, "content": "建议正常放牧", "tool_calls": []}


class _NoKeyLLM:
    enabled = False


# ---------------------------------------------------------------------------
# 告警等级阈值
# ---------------------------------------------------------------------------

def test_alert_level_thresholds():
    assert alert_level_from_coverage(0.80) == "normal"
    assert alert_level_from_coverage(0.60) == "normal"  # 边界：>= warning 阈值
    assert alert_level_from_coverage(0.45) == "warning"
    assert alert_level_from_coverage(0.35) == "warning"  # 边界：>= danger 阈值
    assert alert_level_from_coverage(0.20) == "danger"
    assert alert_level_from_coverage(None) == "normal"


# ---------------------------------------------------------------------------
# LLM 驱动
# ---------------------------------------------------------------------------

def test_run_with_llm_dispatches_tools_and_valid_json():
    orchestrator = AgentOrchestrator(llm=_FakeLLM(), dispatch=_fake_dispatch, max_rounds=8)
    decision = orchestrator.run("东区草场够吗？能放多少羊？")

    assert isinstance(decision, DecisionJSON)
    assert decision.alert_level == "normal"  # 0.7 >= 0.60
    assert decision.recommended_stocking == 250.0
    assert decision.rotation_plan == "划分 3 个轮牧小区轮换放牧"
    assert decision.recommended_actions[0] == "建议正常放牧"
    assert orchestrator.last_tool_calls == ["grassland_coverage", "carrying_capacity"]
    # 决策 JSON 可序列化且字段齐全
    data = decision.model_dump()
    assert set(data) == {
        "alert_level",
        "recommended_actions",
        "recommended_stocking",
        "rotation_plan",
        "evidence",
    }


def test_run_loop_terminates_at_max_rounds():
    class _LoopLLM:
        enabled = True

        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools=None):
            self.calls += 1
            return {
                "fallback": False,
                "content": None,
                "tool_calls": [{"name": "grassland_coverage", "arguments": {}, "id": "c"}],
            }

    llm = _LoopLLM()
    orchestrator = AgentOrchestrator(llm=llm, dispatch=_fake_dispatch, max_rounds=3)
    decision = orchestrator.run("草场怎么样？")
    assert llm.calls == 3  # 循环在 max_rounds 处终止
    assert isinstance(decision, DecisionJSON)


def test_run_llm_fallback_midway_uses_rules():
    class _FailLLM:
        enabled = True

        def complete(self, messages, tools=None):
            return {"fallback": True, "content": None, "tool_calls": [], "reason": "timeout"}

    orchestrator = AgentOrchestrator(llm=_FailLLM(), dispatch=_fake_dispatch, max_rounds=8)
    decision = orchestrator.run("东区草场能放多少只羊？")
    # 规则命中：覆盖度 / 羊群计数 / 草场状况（无参数不触发载畜量试算）
    assert orchestrator.last_tool_calls == ["grassland_coverage", "sheep_count", "pasture_status"]
    assert isinstance(decision, DecisionJSON)


# ---------------------------------------------------------------------------
# 规则降级（无 key）
# ---------------------------------------------------------------------------

def test_run_rule_fallback_no_key():
    """无 key 规则降级：草场问题走草场口径（草场状况 + 覆盖度 + 羊数）。"""
    orchestrator = AgentOrchestrator(llm=_NoKeyLLM(), dispatch=_fake_dispatch, max_rounds=8)
    decision = orchestrator.run("东区草场够吗？能放多少只羊？")

    assert orchestrator.last_tool_calls == [
        "grassland_coverage",
        "sheep_count",
        "pasture_status",
    ]
    assert decision.alert_level == "normal"
    # 草场口径：覆盖度 0.80 × 草场面积 120 亩 → 理论 800，正常等级不折减
    assert decision.recommended_stocking == 800.0
    assert any("东区示范牧场" in e for e in decision.evidence)
    assert any("羊群数量 30" in e for e in decision.evidence)
    assert decision.model_dump()["alert_level"] in {"normal", "warning", "danger"}


def test_run_rule_fallback_question_with_no_keywords():
    orchestrator = AgentOrchestrator(llm=_NoKeyLLM(), dispatch=_fake_dispatch, max_rounds=8)
    decision = orchestrator.run("你好")
    assert orchestrator.last_tool_calls == []
    # 无工具结果时给出兜底依据
    assert any("未调用" in e for e in decision.evidence)


# ---------------------------------------------------------------------------
# 载畜量 × 告警等级联动（阶段 5）
# ---------------------------------------------------------------------------

def _orchestrator():
    return AgentOrchestrator(llm=_NoKeyLLM())


def test_decision_danger_zeroes_stocking():
    """危险等级：理论载畜量折减为 0（禁牧），轮牧文本整体替换为禁牧说明。"""
    results = {
        "grassland_coverage": {"coverage": 0.3375, "greenness": 0.14},
        "carrying_capacity": {"recommended_stocking": 666.67, "rotation_plan": "划分 6 个轮牧小区"},
    }
    d = _orchestrator()._build_decision("q", results, None)
    assert d.alert_level == "danger"
    assert d.recommended_stocking == 0.0  # 666.67 × 危险系数 0
    assert d.rotation_plan.startswith("【建议禁牧】")
    assert "轮牧小区" not in d.rotation_plan  # 原轮牧文本不再残留，避免与禁牧矛盾
    assert "666.67" in d.rotation_plan  # 恢复后参考的理论值
    assert any("禁牧" in e and "理论载畜量 666.67" in e for e in d.evidence)


def test_decision_warning_applies_factor():
    """预警等级：建议载畜量 = 理论值 × 0.6，轮牧文本注明折减。"""
    results = {
        "grassland_coverage": {"coverage": 0.45, "greenness": 0.2},
        "carrying_capacity": {"recommended_stocking": 250.0, "rotation_plan": "轮牧方案"},
    }
    d = _orchestrator()._build_decision("q", results, None)
    assert d.alert_level == "warning"
    assert d.recommended_stocking == 150.0  # 250 × 0.6
    assert "折减" in d.rotation_plan
    assert any("150.0" in e for e in d.evidence)


def test_decision_normal_factor_is_one():
    results = {
        "grassland_coverage": {"coverage": 0.70, "greenness": 0.4},
        "carrying_capacity": {"recommended_stocking": 250.0, "rotation_plan": "轮牧方案"},
    }
    d = _orchestrator()._build_decision("q", results, None)
    assert d.recommended_stocking == 250.0
    assert "折减" not in d.rotation_plan


def test_decision_coverage_only_auto_capacity():
    """未调载畜量工具时按覆盖度折算亩产自动计算（默认口径 100 亩/0.5/30 天）。"""
    # 覆盖度 0.30 → 亩产 400 → 理论 333.33；danger 系数 0 → 建议 0
    d = _orchestrator()._build_decision(
        "q", {"grassland_coverage": {"coverage": 0.30, "greenness": 0.1}}, None
    )
    assert d.alert_level == "danger"
    assert d.recommended_stocking == 0.0
    assert any("理论载畜量 333.33" in e for e in d.evidence)
    assert any("折算为 400.0" in e for e in d.evidence)

    # 覆盖度 0.80 → 满产 800 → 理论 666.67；normal 系数 1 → 建议 666.67
    d2 = _orchestrator()._build_decision(
        "q", {"grassland_coverage": {"coverage": 0.80, "greenness": 0.4}}, None
    )
    assert d2.alert_level == "normal"
    assert d2.recommended_stocking == 666.67


def test_decision_no_data_stocking_zero():
    d = _orchestrator()._build_decision("你好", {}, None)
    assert d.recommended_stocking == 0.0
    assert any("未调用" in e for e in d.evidence)


def test_decision_long_llm_report_not_stuffed_into_actions():
    """LLM 长报告（Markdown/多行）不并入措施列表（思考流完整展示），短句才并入。"""
    results = {"grassland_coverage": {"coverage": 0.70, "greenness": 0.4}}
    long_report = "## 分析\n\n| 指标 | 值 |\n|---|---|\n| 覆盖度 | 70% |\n" * 3
    d = _orchestrator()._build_decision("q", results, long_report)
    assert long_report not in d.recommended_actions
    assert all(len(a) <= 80 for a in d.recommended_actions)

    short = "建议本周降低放牧强度并观察草情变化"
    d2 = _orchestrator()._build_decision("q", results, short)
    assert d2.recommended_actions[0] == short


# ---------------------------------------------------------------------------
# 事件流（阶段 F1：SSE 思考流）
# ---------------------------------------------------------------------------

def test_run_emits_event_sequence_with_rules():
    """无 key 规则降级路径：thinking → tool_call/result/analysis × N → answer。

    带明确参数（亩产/面积/天数）的载畜量问题会额外调用 carrying_capacity 试算。
    """
    events: list = []
    orchestrator = AgentOrchestrator(llm=_NoKeyLLM(), dispatch=_fake_dispatch, max_rounds=8)
    orchestrator.run("东区草场够吗？亩产500公斤、100亩、放牧30天能放多少只羊？", on_event=events.append)

    types = [e["type"] for e in events]
    assert types[0] == "thinking"
    assert types[-1] == "answer"
    assert [e["name"] for e in events if e["type"] == "tool_call"] == [
        "grassland_coverage",
        "sheep_count",
        "carrying_capacity",
        "pasture_status",
    ]
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) == 4
    assert tool_results[0]["result"]["coverage"] == 0.7
    # 每个工具结果后跟一条确定性解读（analysis）
    analyses = [e for e in events if e["type"] == "analysis"]
    assert [a["name"] for a in analyses] == [
        "grassland_coverage",
        "sheep_count",
        "carrying_capacity",
        "pasture_status",
    ]
    assert "70.0%" in analyses[0]["text"]
    assert "东区示范牧场" in analyses[3]["text"]
    # 所有事件带毫秒时间戳且非递减
    tss = [e["ts"] for e in events]
    assert all(isinstance(t, int) for t in tss)
    assert tss == sorted(tss)
    assert events[-1]["decision"]["alert_level"] in {"normal", "warning", "danger"}


def test_rules_generic_capacity_question_skips_param_trial():
    """无参数的载畜量问题不调 carrying_capacity 试算（改由系统按草场口径折算）。"""
    events: list = []
    orchestrator = AgentOrchestrator(llm=_NoKeyLLM(), dispatch=_fake_dispatch, max_rounds=8)
    orchestrator.run("东区草场现在建议载畜量是多少？", on_event=events.append)
    names = [e["name"] for e in events if e["type"] == "tool_call"]
    assert "carrying_capacity" not in names
    assert "pasture_status" in names


def test_decision_prefers_pasture_aggregate():
    """草场口径优先：命中具体草场时用该草场的综合覆盖度与面积核定载畜量。"""
    results = {
        "grassland_coverage": {"coverage": 0.7, "greenness": 0.5, "source": "pastures"},
        "pasture_status": {
            "pasture": {
                "code": "P1",
                "name": "东区示范牧场",
                "area_mu": 120.0,
                "coverage": 0.80,
                "greenness": 0.2,
                "alert_level": "normal",
                "plot_count": 3,
                "analyzed_plot_count": 3,
                "image_count": 4,
            }
        },
    }
    d = _orchestrator()._build_decision("东区示范牧场草情如何？", results, None)
    assert d.alert_level == "normal"
    # 理论载畜量 = 折算亩产 800 × 120 亩 × 0.5 ÷ (2 × 30) = 800
    assert d.recommended_stocking == 800.0
    assert any("东区示范牧场" in e and "编号照片" in e for e in d.evidence)
    assert any("120" in e for e in d.evidence)  # 面积按草场数据核定


def test_decision_region_uses_pasture_sum_and_area():
    """全草场概览口径：**不取覆盖度均值**——等级取各场中最严重、建议载畜量取各场之和。"""
    results = {
        "pasture_status": {
            "count": 2,
            "matched": False,
            "pastures": [
                {
                    "code": "P1", "name": "东区示范牧场", "area_mu": 120.0, "coverage": 0.8,
                    "alert_level": "normal", "theoretical": 800.0, "recommended_stocking": 800.0,
                },
                {
                    "code": "P2", "name": "西区生态牧场", "area_mu": 90.0, "coverage": 0.5,
                    "alert_level": "warning", "theoretical": 600.0, "recommended_stocking": 360.0,
                },
            ],
        }
    }
    d = _orchestrator()._build_decision("现在建议载畜量是多少？", results, None)
    assert d.recommended_stocking == 1160.0  # 800 + 360（各草场折减后之和）
    assert d.alert_level == "warning"  # 最严重草场 P2（不做跨草场覆盖度平均）
    assert any("建议载畜量合计 1160.0" in e for e in d.evidence)
    assert any("P2 西区生态牧场" in e and "最严重" in e for e in d.evidence)
    assert any("210.0 亩" in e for e in d.evidence)  # 面积合计 120 + 90
    assert not any("按预警等级折减" in e for e in d.evidence)  # 不与汇总口径自相矛盾
    assert all("均值" not in e for e in d.evidence)  # 不存在跨草场平均
    # 轮牧方案逐场独立（各草场按自身等级执行）
    assert "P1" in d.rotation_plan and "P2" in d.rotation_plan


def test_decision_param_trial_wins_over_region_sum():
    """带参数试算时以参数结果为准（不受草场概览影响）。"""
    results = {
        "carrying_capacity": {"recommended_stocking": 500.0, "rotation_plan": "轮牧方案"},
        "pasture_status": {
            "pastures": [
                {"code": "P1", "name": "东区示范牧场", "area_mu": 120.0, "coverage": 0.8,
                 "recommended_stocking": 800.0},
            ]
        },
    }
    d = _orchestrator()._build_decision("亩产500公斤、100亩、放牧30天能放多少羊？", results, None)
    assert d.recommended_stocking == 500.0


def test_decision_coverage_tool_alone_keeps_pastures_independent():
    """只调覆盖度工具（未调草场工具）也按各草场独立口径：等级取最严重、载畜量取各场之和。"""
    results = {
        "grassland_coverage": {
            "source": "pastures",
            "pasture_count": 3,
            "analyzed_pasture_count": 3,
            "total_area_mu": 270.0,
            "recommended_stocking_sum": 1087.87,
            "pastures": [
                {"code": "P1", "name": "东区示范牧场", "coverage": 0.81, "alert_level": "normal",
                 "area_mu": 120.0, "theoretical": 1200.0, "recommended_stocking": 1087.87},
                {"code": "P2", "name": "西区生态牧场", "coverage": 0.52, "alert_level": "warning",
                 "area_mu": 90.0, "theoretical": 800.0, "recommended_stocking": 0.0},
                {"code": "P3", "name": "北坡轮休草场", "coverage": 0.22, "alert_level": "danger",
                 "area_mu": 60.0, "theoretical": 500.0, "recommended_stocking": 0.0},
            ],
        }
    }
    d = _orchestrator()._build_decision("当前草场覆盖度是多少？", results, None)
    assert d.recommended_stocking == 1087.87  # 各草场独立核定后求和
    assert d.alert_level == "danger"  # 最严重草场 P3（不是覆盖度均值判定）
    assert any("建议载畜量合计 1087.87" in e for e in d.evidence)
    assert any("P3 北坡轮休草场" in e and "最严重" in e for e in d.evidence)
    # 逐场列出覆盖度；任何一条依据都不含跨草场「均值」
    assert any("P1 东区示范牧场 覆盖度 81.00%" in e for e in d.evidence)
    assert all("均值" not in e for e in d.evidence)


def test_llm_without_tool_calls_supplements_with_rules():
    """模型只作答未调工具：用规则补齐工具调用（决策数字必须来自工具）。"""
    class _NoToolLLM:
        enabled = True

        def complete(self, messages, tools=None):
            return {"fallback": False, "content": "我觉得可以放牧", "tool_calls": []}

    events: list = []
    orchestrator = AgentOrchestrator(llm=_NoToolLLM(), dispatch=_fake_dispatch, max_rounds=8)
    decision = orchestrator.run("东区草场够吗？能放多少只羊？", on_event=events.append)

    assert orchestrator.last_tool_calls == ["grassland_coverage", "sheep_count", "pasture_status"]
    assert decision.recommended_stocking > 0  # 数字来自工具结果
    assert any(e["type"] == "thinking" and "补齐数据查询" in e["text"] for e in events)


def test_analyze_tool_result_pasture_status():
    """草场状况的确定性解读：单草场与全草场概览两种形态。"""
    from app.agent.orchestrator import analyze_tool_result

    single = analyze_tool_result(
        "pasture_status",
        {
            "pasture": {
                "name": "北坡轮休草场",
                "coverage": 0.219,
                "alert_level": "danger",
                "analyzed_plot_count": 3,
                "plot_count": 3,
            }
        },
    )
    assert "北坡轮休草场" in single and "21.9%" in single and "危险" in single

    overview = analyze_tool_result(
        "pasture_status",
        {
            "count": 2,
            "pastures": [
                {"code": "P1", "name": "东区示范牧场", "coverage": 0.8, "alert_level": "normal"},
                {"code": "P2", "name": "西区生态牧场", "coverage": 0.5, "alert_level": "warning"},
            ],
        },
    )
    assert "共 2 个草场" in overview and "东区示范牧场" in overview


def test_analyze_tool_result_thresholds():
    """确定性解读按覆盖度阈值分档；错误结果不产生解读。"""
    from app.agent.orchestrator import analyze_tool_result

    ok_text = analyze_tool_result("grassland_coverage", {"coverage": 0.8})
    warn_text = analyze_tool_result("grassland_coverage", {"coverage": 0.45})
    danger_text = analyze_tool_result("grassland_coverage", {"coverage": 0.3})
    assert "高于正常阈值" in ok_text
    assert "预警区间" in warn_text
    assert "危险区间" in danger_text

    assert "3 只羊" in analyze_tool_result("sheep_count", {"count": 3})
    assert "666.67" in analyze_tool_result("carrying_capacity", {"recommended_stocking": 666.67})
    assert analyze_tool_result("grassland_coverage", {"error": "无法读取图片"}) is None
    assert analyze_tool_result("unknown_tool", {"x": 1}) is None


def test_run_emits_events_with_llm():
    """LLM 路径：模型文字进入思考流，工具调用成对出现。"""
    events: list = []
    orchestrator = AgentOrchestrator(llm=_FakeLLM(), dispatch=_fake_dispatch, max_rounds=8)
    orchestrator.run("东区草场够吗？能放多少羊？", on_event=events.append)

    types = [e["type"] for e in events]
    assert types[-1] == "answer"
    # 第 2 轮模型输出文字进入思考流
    assert any(e["type"] == "thinking" and "建议正常放牧" in e["text"] for e in events)
    # 第 2 轮开始有轮次提示
    assert any(e["type"] == "thinking" and "第 2 轮" in e["text"] for e in events)
    assert [e["name"] for e in events if e["type"] == "tool_call"] == [
        "grassland_coverage",
        "carrying_capacity",
    ]


def test_run_emits_fallback_thinking_midway():
    """LLM 中途失败：思考流出现降级提示，随后规则工具事件。"""
    class _FailLLM:
        enabled = True

        def complete(self, messages, tools=None):
            return {"fallback": True, "content": None, "tool_calls": [], "reason": "timeout"}

    events: list = []
    orchestrator = AgentOrchestrator(llm=_FailLLM(), dispatch=_fake_dispatch, max_rounds=8)
    orchestrator.run("东区草场能放多少只羊？", on_event=events.append)

    assert any(e["type"] == "thinking" and "降级" in e["text"] for e in events)
    assert [e["name"] for e in events if e["type"] == "tool_call"] == [
        "grassland_coverage",
        "sheep_count",
        "pasture_status",
    ]
    assert events[-1]["type"] == "answer"


class _RecordingLLM:
    """记录每轮收到的 messages，用于断言回传格式符合 OpenAI 规范。"""

    enabled = True

    def __init__(self):
        self.calls = 0
        self.messages_seen: list[list[dict]] = []

    def complete(self, messages, tools=None):
        self.calls += 1
        self.messages_seen.append([dict(m) for m in messages])
        if self.calls == 1:
            return {
                "fallback": False,
                "content": None,
                "tool_calls": [{"name": "grassland_coverage", "arguments": {}, "id": "c1"}],
            }
        return {"fallback": False, "content": "完成", "tool_calls": []}


def test_llm_round2_uses_openai_tool_wire_format():
    """第 2 轮回传的 assistant.tool_calls 必须是 OpenAI 线格式（type=function + function 包装）。

    回归防护：GLM 曾因缺少 type 字段报「工具类型不能为空」(code 1214)，
    导致所有多轮对话静默降级为规则兜底。
    """
    import json as _json

    llm = _RecordingLLM()
    orchestrator = AgentOrchestrator(llm=llm, dispatch=_fake_dispatch, max_rounds=8)
    orchestrator.run("草场怎么样？")

    assert llm.calls == 2
    second_round = llm.messages_seen[1]
    assistant = next(m for m in second_round if m["role"] == "assistant")
    tc = assistant["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"] == "c1"
    assert tc["function"]["name"] == "grassland_coverage"
    assert _json.loads(tc["function"]["arguments"]) == {}  # arguments 为 JSON 字符串
    tool_msg = next(m for m in second_round if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1"
