"""登录鉴权（阶段 9.1）：标准库 HMAC 无状态令牌与口令校验。

护栏要点：
- 令牌格式被逐字节钉死（测试独立重算 HMAC-SHA256 签名），防止后续改动悄悄换算法；
- 过期 / 篡改 / 垃圾输入一律拒绝；改密钥即全量失效；
- **不引入第三方依赖**：AST 检查 `app/auth.py` 的 import 集合。
"""
from __future__ import annotations

import ast
import base64
import hashlib
import hmac
import json
import re
import time
from pathlib import Path

import pytest

from app.auth import hash_password, make_token, verify_password, verify_token
from app.config import settings

AUTH_PY = Path(__file__).resolve().parent.parent / "app" / "auth.py"


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _recompute_signature(body: str, secret: str | None = None) -> str:
    """测试侧独立重算签名（只依赖标准库语义，不调用被测私有函数）。"""
    key = (secret if secret is not None else settings.auth_token_secret).encode("utf-8")
    return _b64e(hmac.new(key, body.encode("ascii"), hashlib.sha256).digest())


def _forge(payload: dict, secret: str | None = None) -> str:
    """用给定 payload 手工拼一个签名正确的令牌（用于测 payload 校验分支）。"""
    body = _b64e(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return body + "." + _recompute_signature(body, secret)


# ---------------------------------------------------------------------------
# 令牌：签发 / 校验 / 格式
# ---------------------------------------------------------------------------

def test_make_and_verify_token_roundtrip():
    issued = make_token("admin")
    assert issued["username"] == "admin"
    assert issued["token"] and issued["expires_at"] > time.time()
    assert verify_token(issued["token"]) == "admin"


def test_token_format_two_base64url_parts():
    token = make_token("admin")["token"]
    parts = token.split(".")
    assert len(parts) == 2
    for part in parts:
        assert part and re.fullmatch(r"[A-Za-z0-9_-]+", part), part  # base64url，无 padding


def test_signature_is_hmac_sha256_of_body():
    """格式钉死：第二段 == HMAC-SHA256(secret, 第一段) 的 base64url。"""
    token = make_token("admin")["token"]
    body, signature = token.split(".")
    assert signature == _recompute_signature(body)

    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    assert set(payload) == {"sub", "iat", "exp"}
    assert payload["sub"] == "admin"
    assert payload["exp"] > payload["iat"]


def test_custom_ttl_and_injected_now():
    issued = make_token("admin", ttl_seconds=60, now=1_000_000.0)
    assert issued["expires_at"] == 1_000_060
    assert verify_token(issued["token"], now=1_000_059.9) == "admin"
    assert verify_token(issued["token"], now=1_000_060.0) is None  # 到期即失效（闭区间外）


def test_expired_token_rejected():
    expired = make_token("admin", ttl_seconds=-1)["token"]
    assert verify_token(expired) is None


def test_changing_secret_invalidates_tokens(monkeypatch):
    token = make_token("admin")["token"]
    monkeypatch.setattr(settings, "auth_token_secret", "another-secret")
    assert verify_token(token) is None


def test_tampered_payload_rejected():
    """改 payload 但保留旧签名 → 拒绝（覆盖伪造用户名与延长过期两条路）。"""
    token = make_token("admin")["token"]
    body, signature = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["sub"] = "root"
    payload["exp"] += 10**6
    forged = _b64e(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert verify_token(forged + "." + signature) is None


def test_tampered_signature_rejected():
    token = make_token("admin")["token"]
    body, signature = token.split(".")
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    assert verify_token(body + "." + flipped) is None


@pytest.mark.parametrize("bad", [None, "", "abc", "a.b.c", "..", ".", "only-one-part",
                                 "签名不可解.x", "x.签名不可解"])
def test_garbage_tokens_rejected(bad):
    assert verify_token(bad) is None


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "a", "dict"],           # 顶层非对象
        {"iat": 1, "exp": 2},           # 缺 sub
        {"sub": "", "iat": 1, "exp": 2},      # 空用户名
        {"sub": 123, "iat": 1, "exp": 2},     # sub 非字符串
        {"sub": "admin", "iat": 1},           # 缺 exp
        {"sub": "admin", "iat": 1, "exp": "soon"},  # exp 非数值
    ],
)
def test_payload_shape_validated(payload):
    """签名正确但 payload 形状非法的令牌一律拒绝（防手改 payload 绕过）。"""
    assert verify_token(_forge(payload)) is None


# ---------------------------------------------------------------------------
# 口令哈希（PBKDF2 加盐，存储格式与常量时间比对）
# ---------------------------------------------------------------------------

def test_hash_password_roundtrip_and_salt_uniqueness():
    stored = hash_password("admin123")
    assert stored.startswith("pbkdf2_sha256$")
    assert "admin123" not in stored  # 明文绝不入库
    assert verify_password("admin123", stored)
    assert not verify_password("wrong", stored)
    # 加盐：同一口令两次哈希不同，但都能验证通过
    again = hash_password("admin123")
    assert again != stored
    assert verify_password("admin123", again)


@pytest.mark.parametrize("bad", [None, "", "plain-text", "pbkdf2_sha256$only-two",
                                 "md5$abc$def", "pbkdf2_sha256$!!!$!!!"])
def test_verify_password_rejects_malformed_hash(bad):
    assert not verify_password("anything", bad)


def test_verify_password_supports_unicode():
    """非 ASCII 口令同样可比对（按 UTF-8 字节参与派生）。"""
    stored = hash_password("牧易通·口令")
    assert verify_password("牧易通·口令", stored)
    assert not verify_password("牧易通", stored)


# ---------------------------------------------------------------------------
# 依赖护栏：只用标准库
# ---------------------------------------------------------------------------

def test_token_module_imports_stdlib_only():
    """AST 检查 app/auth.py 的 import：仅允许标准库与本项目模块（离线/最小依赖）。"""
    tree = ast.parse(AUTH_PY.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "base64", "hashlib", "hmac", "json", "os", "time",
                        "typing", "urllib", "app"}, imported


# ---------------------------------------------------------------------------
# 登录接口（POST /api/auth/login、GET /api/auth/me）
# ---------------------------------------------------------------------------

def test_login_returns_token_envelope(anon_client):
    resp = anon_client.post(
        "/api/auth/login",
        json={"username": settings.auth_username, "password": settings.auth_password},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0 and body["message"] == "ok"
    data = body["data"]
    assert set(data) == {"token", "username", "expires_at", "ttl_seconds", "role"}
    assert data["username"] == settings.auth_username
    assert data["role"] == "admin"  # 播种的管理员账号
    assert verify_token(data["token"]) == settings.auth_username


def test_login_wrong_password_envelope(anon_client):
    """失败返回统一信封（HTTP 200 承载业务码，与全站约定一致）。"""
    resp = anon_client.post(
        "/api/auth/login",
        json={"username": settings.auth_username, "password": "definitely-wrong"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"code", "data", "message"}
    assert body["code"] == 401 and body["data"] is None
    assert "密码" in body["message"]


def test_login_unknown_user_rejected(anon_client):
    body = anon_client.post(
        "/api/auth/login", json={"username": "root", "password": settings.auth_password}
    ).json()
    assert body["code"] == 401


def test_login_missing_or_empty_fields_422(anon_client):
    for payload in ({}, {"username": settings.auth_username}, {"password": "x"},
                    {"username": "", "password": ""}):
        resp = anon_client.post("/api/auth/login", json=payload)
        assert resp.status_code == 422, payload


def test_login_failure_never_echoes_password(anon_client):
    """错误响应不得回显提交的口令（防日志/截图泄密）。"""
    secret = "never-echo-me-123"
    text = anon_client.post(
        "/api/auth/login", json={"username": settings.auth_username, "password": secret}
    ).text
    assert secret not in text


def test_me_returns_username(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["username"] == settings.auth_username
    assert data["role"] == "admin"


def test_me_requires_token(anon_client):
    resp = anon_client.get("/api/auth/me")
    assert resp.status_code == 401
    assert resp.json()["code"] == 401


# ---------------------------------------------------------------------------
# 中间件：/api/* 保护与豁免
# ---------------------------------------------------------------------------

def _login_token(anon_client) -> str:
    return anon_client.post(
        "/api/auth/login",
        json={"username": settings.auth_username, "password": settings.auth_password},
    ).json()["data"]["token"]


def test_protected_endpoint_requires_token(anon_client):
    resp = anon_client.get("/api/pastures")
    assert resp.status_code == 401
    body = resp.json()
    assert set(body) == {"code", "data", "message"}
    assert body["code"] == 401 and "登录" in body["message"]


def test_protected_endpoint_accepts_bearer_header(anon_client):
    token = _login_token(anon_client)
    resp = anon_client.get("/api/pastures", headers={"Authorization": "Bearer " + token})
    assert resp.status_code == 200
    assert resp.json()["code"] == 0


def test_protected_endpoint_accepts_query_param_token(anon_client):
    """SSE（EventSource 无法带头）通道：?token= 同样放行。"""
    token = _login_token(anon_client)
    resp = anon_client.get(f"/api/pastures?token={token}")
    assert resp.status_code == 200
    assert resp.json()["code"] == 0


def test_protected_endpoint_rejects_garbage_token(anon_client):
    for bad in ("", "garbage", "a.b", "Bearer garbage"):
        headers = {} if bad.startswith("Bearer") else ({"Authorization": "Bearer " + bad} if bad else {})
        resp = anon_client.get("/api/pastures", headers=headers or None)
        assert resp.status_code == 401, bad


def test_protected_endpoint_rejects_expired_token(anon_client, monkeypatch):
    monkeypatch.setattr(settings, "auth_token_ttl_seconds", -10.0)
    token = make_token(settings.auth_username)["token"]
    resp = anon_client.get("/api/pastures", headers={"Authorization": "Bearer " + token})
    assert resp.status_code == 401
    assert resp.json()["code"] == 401


def test_header_token_takes_precedence_over_query(client):
    """头与查询参数都带令牌时：头里的坏令牌优先被使用（拒绝），不偷偷吃查询参数。"""
    valid = make_token(settings.auth_username)["token"]
    resp = client.get(
        f"/api/pastures?token={valid}", headers={"Authorization": "Bearer bad"}
    )
    assert resp.status_code == 401


def test_health_and_login_exempt(anon_client):
    assert anon_client.get("/api/health").status_code == 200
    assert anon_client.post(
        "/api/auth/login",
        json={"username": settings.auth_username, "password": settings.auth_password},
    ).status_code == 200


def test_static_and_samples_exempt(anon_client):
    """非 /api 路径不经鉴权：首页、静态资源、404 页面均照常。"""
    assert anon_client.get("/").status_code == 200
    assert anon_client.get("/assets/bg-pasture.jpg").status_code == 200
    assert anon_client.get("/admin.html").status_code == 404  # 早已移除，与鉴权无关


def test_alert_sse_accepts_query_token(anon_client, monkeypatch):
    """告警 SSE 端点用 ?token= 鉴权放行（有限生成器，不触真实无限流）。"""
    from fastapi.sse import ServerSentEvent

    from app.routers import dashboard

    async def finite_stream():
        yield ServerSentEvent(event="alert", data={"id": 1, "level": "warning"})

    monkeypatch.setattr(dashboard, "alert_event_stream", finite_stream)
    token = _login_token(anon_client)
    resp = anon_client.get(f"/api/dashboard/alerts/stream?token={token}")
    assert resp.status_code == 200
    assert "event: alert" in resp.text

    denied = anon_client.get("/api/dashboard/alerts/stream")
    assert denied.status_code == 401


def test_middleware_marks_scope_username(anon_client):
    """登录态注入 scope state（供 /api/auth/me 读取）。"""
    token = _login_token(anon_client)
    resp = anon_client.get("/api/auth/me", headers={"Authorization": "Bearer " + token})
    assert resp.json()["data"]["username"] == settings.auth_username
