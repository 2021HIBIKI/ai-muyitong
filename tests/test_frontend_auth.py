"""前端登录鉴权（阶段 9.3）：登录屏 / fetch 令牌注入 / SSE 查询参数令牌。

纯静态解析 ``static/index.html``（与 test_motion 同风格）：
- 登录屏与演示账号提示存在；fetch 包装只接管 /api/* 且不消费响应体
  （Agent 思考流依赖 ``resp.body.getReader()``）；
- EventSource 必须带 ``?token=``（无法自定义请求头），CLOSED 后手动重建；
- 启动门：无令牌显示登录屏，有令牌先 /api/auth/me 校验再 startApp；
- 退出登录清除 localStorage 令牌并整页重载。
"""
from __future__ import annotations

import re
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")


def _function_text(name: str) -> str:
    """按函数名抽取 JS 函数体（单行函数声明 + 花括号配对）。"""
    marker = f"function {name}("
    idx = HTML.index(marker)
    start = HTML.index("{", idx)
    depth = 0
    for i in range(start, len(HTML)):
        if HTML[i] == "{":
            depth += 1
        elif HTML[i] == "}":
            depth -= 1
            if depth == 0:
                return HTML[idx : i + 1]
    raise AssertionError(f"未解析到函数 {name}")


def test_login_screen_markup():
    assert 'id="loginScreen"' in HTML and 'class="login-card"' in HTML
    for anchor in ('id="loginUser"', 'id="loginPass"', 'id="loginBtn"', 'id="loginMsg"',
                   'class="login-screen" hidden'):
        assert anchor in HTML, anchor
    assert "演示账号 admin / admin123" in HTML
    # 登录卡入场动画沿用 viewIn + 动效 token
    assert ".login-card {" in HTML
    match = re.search(r"\.login-card \{[^}]*animation: ([^;]+);", HTML)
    assert match and "viewIn" in match.group(1) and "var(--ease-out)" in match.group(1)
    # 登录屏初始覆盖舞台（舞台默认隐藏，登录后才可见）
    assert 'id="stage" style="visibility:hidden"' in HTML


def test_fetch_patch_scoped_to_api():
    """fetch 包装：只接管 /api/* 字符串 URL；非 /api（样例图/视频/assets）原样透传。"""
    patch = _function_text("patchedFetch")
    assert 'url.indexOf("/api/") !== 0' in patch
    assert 'headers.set("Authorization", "Bearer " + t)' in patch
    assert "return nativeFetch(input, init)" in patch  # 非 /api 直接透传
    assert patch.index('return nativeFetch(input, init)') < patch.index('var opts')


def test_fetch_patch_never_consumes_response_body():
    """401 检查只读 resp.status；绝不调 .json()/.text()/getReader()（Agent 思考流要原样拿 body）。"""
    patch = _function_text("patchedFetch")
    assert "resp.status === 401" in patch
    assert "resp.json()" not in patch and "resp.text()" not in patch
    assert "resp.body" not in patch
    assert "return resp" in patch  # 原样返回
    # 登录接口不触发「登录过期」回调（避免登录失败递归回登录屏）
    assert 'url.indexOf("/api/auth/login") < 0' in patch


def test_agent_stream_reader_preserved():
    """AI 助手 SSE 手动解析（fetch + getReader）不受包装影响——代码仍在。"""
    assert "resp.body.getReader()" in HTML
    assert '"/api/agent/stream"' in HTML


def test_token_persisted_in_localstorage():
    assert 'localStorage.getItem(AUTH_KEY)' in HTML
    assert 'localStorage.setItem(AUTH_KEY, t)' in HTML
    assert 'localStorage.removeItem(AUTH_KEY)' in HTML


def test_event_source_uses_query_token_and_reconnects_on_closed():
    """EventSource 无法带头 → ?token= 查询参数；401 使连接 CLOSED，需手动重建。"""
    stream = _function_text("connectAlertStream")
    assert '"/api/dashboard/alerts/stream?token=" + encodeURIComponent(authToken())' in stream
    assert "readyState === 2" in stream  # CLOSED：非 200 不自动重连
    assert "connectAlertStream()" in stream  # 15s 后重建


def test_handle_unauthorized_clears_token_and_stops_polling():
    fn = _function_text("handleUnauthorized")
    assert "clearToken()" in fn
    assert "clearInterval(overviewTimer)" in fn and "clearInterval(alertTimer)" in fn
    assert 'showLogin("登录已过期，请重新登录")' in fn
    assert "alertSource.close()" in fn


def test_boot_gated_by_token_and_me_check():
    """启动门：无令牌 → 登录屏；有令牌 → 先 /api/auth/me（走打过补丁的 fetch，
    自动带 Authorization；401 由 handleUnauthorized 清令牌回登录）→ startApp。"""
    boot = _function_text("boot")
    assert 'if (!authToken()) { showLogin(); return; }' in boot
    assert 'fetch("/api/auth/me")' in boot  # 补丁 fetch：必须带令牌（曾用 nativeFetch 漏带头）
    assert "startApp()" in boot
    # startApp 只跑一次并恢复舞台可见
    start = _function_text("startApp")
    assert "if (booted) return" in start
    assert 'stage.style.visibility = "visible"' in start
    assert '("loginScreen").hidden = true' in start or 'id="loginScreen"' in HTML


def test_logout_clears_token_and_reloads():
    logout = HTML.split('$("logoutBtn").addEventListener')[1].split("});")[0] + "});"
    assert "clearToken()" in logout
    assert "location.replace(location.pathname)" in logout
    assert "confirm(" in logout  # 退出前确认（防误触）


def test_login_failure_shows_message_and_shake():
    do = _function_text("doLogin")
    assert '"/api/auth/login"' in do
    assert "登录失败：" in do
    assert 'classList.add("shake")' in do
    # 登录走原生 fetch（不带令牌，避免 401 递归）
    assert "nativeFetch(" in do
