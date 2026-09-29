"""动效规范（护栏测试）：入场/退场曲线与时长统一。

依据 improve-animations 审计（AUDIT.md §2 Easing & duration）：
- entering / exiting → **ease-out**（强曲线）；入场用裸 `ease` 即为缺陷；
- UI 动效时长 < 300ms；同一组件的退场不得慢于入场；
- 曲线与时长以 token 为准（``--ease-out`` / ``--dur-*``，避免字面量在多处漂移）。

纯静态解析 ``static/index.html``：不需要浏览器，作为动效规范的回归护栏。
"""
from __future__ import annotations

import re
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")

# 强 ease-out：仓库既有曲线（字面量或 token 皆可）
STRONG_EASE_OUT = re.compile(r"cubic-bezier\(\s*\.32,\s*\.72,\s*\.33,\s*1\s*\)|var\(--ease-out\)")
ENTRANCE_KEYFRAMES = ("viewIn", "lineIn", "toastIn", "spotIn")
BARE_EASE = re.compile(r"(?<![\w-])ease(?![\w-])")


def _rules_with(text: str) -> list[str]:
    return [line.strip() for line in HTML.splitlines() if text in line]


def _rule_line(selector: str) -> str:
    matches = [line.strip() for line in HTML.splitlines() if line.strip().startswith(selector)]
    assert len(matches) == 1, f"应恰好一条 {selector} 规则，实得 {len(matches)}"
    return matches[0]


def _resolve_seconds(value: str) -> float:
    """``.25s`` 或 ``var(--dur-enter)`` → 秒数（token 从 :root 定义解析）。"""
    if value.startswith("var("):
        name = value[len("var(") : -1]
        match = re.search(rf"{re.escape(name)}:\s*([.\d]+)s", HTML)
        assert match, f"缺少 token 定义：{name}"
        return float(match.group(1))
    return float(value.rstrip("s"))


def _transition_seconds(rule: str, prop: str = "opacity") -> float:
    match = re.search(rf"transition: {prop} ([.\d]+s|var\(--[\w-]+\))", rule)
    assert match, f"未解析到 {prop} 过渡时长：{rule}"
    return _resolve_seconds(match.group(1))


def test_entrance_animations_use_strong_ease_out():
    """所有入场动画（viewIn / lineIn / toastIn / spotIn）必须用强 ease-out，不得用裸 ease。"""
    rules: list[str] = []
    for keyframe in ENTRANCE_KEYFRAMES:
        rules.extend(_rules_with(f"animation: {keyframe}"))
    assert rules, "应能解析到入场动画规则"
    for rule in rules:
        assert STRONG_EASE_OUT.search(rule), f"入场缺少强 ease-out：{rule}"
        assert not BARE_EASE.search(rule), f"入场使用了裸 ease：{rule}"


def test_no_ease_in_and_no_transition_all():
    """AUDIT 猎捕项：UI 上不得出现 ease-in；不得使用 transition: all。"""
    without_ease_in_out = HTML.replace("ease-in-out", "")
    assert "ease-in" not in without_ease_in_out
    assert "transition: all" not in HTML


def test_exit_animations_use_ease_out_and_are_not_slower():
    """退场同样用强 ease-out，且**不得慢于入场**（用户已决定关闭，退场要利落）。"""
    toast_out = _rule_line(".alert-toast.out")
    spotlight_out = _rule_line(".spotlight.out")
    for rule in (toast_out, spotlight_out):
        assert STRONG_EASE_OUT.search(rule), f"退场缺少强 ease-out：{rule}"
        assert not BARE_EASE.search(rule), f"退场使用了裸 ease：{rule}"

    # toast：入场 .35s → 退场必须 ≤ 入场；聚光灯退场 ≤ .3s（UI 预算）
    entrance_match = re.search(r"toastIn ([.\d]+)s", HTML)
    assert entrance_match, "未解析到 toast 入场时长"
    assert _transition_seconds(toast_out) <= float(entrance_match.group(1))
    assert _transition_seconds(spotlight_out) <= 0.3


def test_motion_tokens_are_single_source_of_truth():
    """曲线与时长 token 化：字面量只允许出现在 :root 定义处（审计：cohesion/tokens）。"""
    for token in (
        "--ease-out: cubic-bezier(.32, .72, .33, 1)",
        "--dur-press: .12s",
        "--dur-hover: .18s",
        "--dur-enter: .25s",
    ):
        assert token in HTML, f"缺少 token 定义：{token}"
    assert HTML.count("cubic-bezier(.32, .72, .33, 1)") == 1, "曲线字面量应只在 token 定义处出现"
    # 入场规则必须引用 token，而不是各自写值
    for keyframe in ENTRANCE_KEYFRAMES:
        for rule in _rules_with(f"animation: {keyframe}"):
            assert "var(--ease-out)" in rule, rule


def test_press_feedback_uses_press_token_and_ease_out():
    """按压反馈统一为 120ms + ease-out（AUDIT §3），不再残留 .1s/.12s 裸 ease。"""
    assert HTML.count("transform var(--dur-press) var(--ease-out)") >= 6
    for leftover in ("transform .12s ease", "transform .1s ease", "transform .12s;"):
        assert leftover not in HTML, f"残留未 token 化的按压时长：{leftover}"


# ---------------------------------------------------------------------------
# 按键交互（apple-design：按下即响应 / 命中区 / 键盘可达 / 不可误选）
# ---------------------------------------------------------------------------

INTERACTIVE = (
    ".menu-item", ".btn", ".mini-btn", ".sample-btn", ".mode-btn", ".hero-chip",
    ".pasture-item", ".p-del", ".flow-item.clickable", ".hist-row.hlog", ".dropzone",
    ".qe-btn",
)


def test_every_interactive_element_has_press_feedback():
    """所有可点元素都要有 :active 按压反馈——Apple：反馈必须发生在 pointer-down 那一刻。"""
    missing = [sel for sel in INTERACTIVE if not re.search(rf"{re.escape(sel)}:active", HTML)]
    assert not missing, f"缺少按压反馈：{missing}"


def test_press_feedback_transitions_on_transform():
    """按压反馈都必须是 transform 过渡（合成层友好），且带 scale 缩放。"""
    active_rules = [line.strip() for line in HTML.splitlines() if ":active" in line]
    assert len(active_rules) >= len(INTERACTIVE)
    for rule in active_rules:
        assert "transform: scale(" in rule, f"按压缺少 scale：{rule}"


def test_buttons_keyboard_reachable_and_focus_visible():
    """键盘可达（Apple HIG）：可点行 role=button + tabindex，Enter/Space 触发，焦点环可见。"""
    assert HTML.count('role="button" tabindex="0"') >= 4
    assert "button:focus-visible" in HTML and "outline: 2px solid var(--blue)" in HTML
    assert "function bindRowKeyboard(" in HTML and 'e.key !== "Enter"' in HTML


def test_buttons_are_not_text_selectable_and_tap_delay_removed():
    """控件不是文本：不可选中；并去掉移动端点击高亮与双击缩放的 ~300ms 延迟。"""
    assert "user-select: none; -webkit-user-select: none;" in HTML
    assert "-webkit-tap-highlight-color: transparent" in HTML
    assert "touch-action: manipulation" in HTML


def test_system_preferences_supported():
    """Apple §14：低动效 / 低透明偏好必须有对应处理，且低动效不削弱按压反馈。"""
    assert "@media (prefers-reduced-motion: reduce)" in HTML
    assert "@media (prefers-reduced-transparency: reduce)" in HTML

    reduced_motion = HTML.split("@media (prefers-reduced-motion: reduce)")[1].split("@media")[0]
    assert "animation-name: fadeIn" in reduced_motion and "keyframes fadeIn" in reduced_motion
    assert ":active" not in reduced_motion  # 按压反馈属直接反馈，低动效下保留

    reduced_transparency = HTML.split("@media (prefers-reduced-transparency: reduce)")[1]
    assert "backdrop-filter: none" in reduced_transparency
