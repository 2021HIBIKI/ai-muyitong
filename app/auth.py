"""登录鉴权：标准库 HMAC 无状态令牌 + 常量时间口令校验（无第三方依赖）。

设计要点（离线演示与最小依赖约束，刻意不引 jwt/pyjwt）：
- 令牌 = ``base64url(payload).base64url(HMAC-SHA256(secret, base64url(payload)))``，
  payload = ``{"sub": 用户名, "iat": 签发时间, "exp": 过期时间}``（epoch 秒，JSON 紧凑排序）；
- HMAC 密钥取 ``settings.auth_token_secret``（生产部署必须改）；
- 校验先比签名（``hmac.compare_digest`` 常量时间）再解 payload，过期/篡改/格式错误一律
  返回 ``None``；退出登录为客户端丢弃令牌（无状态令牌不做服务端吊销）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
from typing import Any

from app.config import settings
from app.schemas import err


# ---------------------------------------------------------------------------
# 编码与签名
# ---------------------------------------------------------------------------

def _b64e(raw: bytes) -> str:
    """base64url 编码（去 padding，便于放 URL 查询参数）。"""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    """base64url 解码（容忍缺失的 padding）。"""
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _secret() -> bytes:
    return settings.auth_token_secret.encode("utf-8")


def _sign(body: str) -> str:
    """对 payload 段（base64url 文本）做 HMAC-SHA256 签名，返回 base64url。"""
    digest = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest()
    return _b64e(digest)


# ---------------------------------------------------------------------------
# 令牌签发与校验
# ---------------------------------------------------------------------------

def make_token(
    username: str, ttl_seconds: float | None = None, now: float | None = None
) -> dict[str, Any]:
    """签发令牌。

    返回 ``{token, username, expires_at, ttl_seconds}``；``now`` 可注入便于测试过期分支。
    """
    issued = time.time() if now is None else now
    ttl = settings.auth_token_ttl_seconds if ttl_seconds is None else ttl_seconds
    payload = {"sub": username, "iat": int(issued), "exp": int(issued + ttl)}
    body = _b64e(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "token": body + "." + _sign(body),
        "username": username,
        "expires_at": payload["exp"],
        "ttl_seconds": ttl,
    }


def verify_token(token: str | None, now: float | None = None) -> str | None:
    """校验令牌：有效返回用户名；过期、签名不符、格式错误一律返回 ``None``。"""
    if not token or not isinstance(token, str):
        return None
    parts = token.split(".")
    if len(parts) != 2:
        return None
    body, signature = parts
    if not body or not signature:
        return None
    # 先验签（常量时间），再解析不可信内容；非 ASCII 输入按格式错误处理
    try:
        signature_bytes = signature.encode("ascii")
        expected_bytes = _sign(body).encode("ascii")
    except UnicodeEncodeError:
        return None
    if not hmac.compare_digest(signature_bytes, expected_bytes):
        return None
    try:
        payload = json.loads(_b64d(body))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    subject, expires = payload.get("sub"), payload.get("exp")
    if not isinstance(subject, str) or not subject:
        return None
    if not isinstance(expires, (int, float)):
        return None
    if (time.time() if now is None else now) >= expires:
        return None
    return subject


# ---------------------------------------------------------------------------
# 口令哈希（PBKDF2-HMAC-SHA256，标准库实现，加盐）
# ---------------------------------------------------------------------------

PBKDF2_ROUNDS = 120_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """生成口令哈希：``pbkdf2_sha256$<salt_b64>$<hash_b64>``（盐随机，同口令两次结果不同）。"""
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_b64e(salt)}${_b64e(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    """比对口令与已存哈希（常量时间）；格式非法或空哈希一律拒绝。"""
    if not stored or not isinstance(stored, str):
        return False
    parts = stored.split("$")
    if len(parts) != 3 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        salt, expected = _b64d(parts[1]), parts[2]
    except (ValueError, UnicodeDecodeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
    return hmac.compare_digest(_b64e(digest), expected)


# ---------------------------------------------------------------------------
# ASGI 鉴权中间件
# ---------------------------------------------------------------------------

class AuthMiddleware:
    """保护 ``/api/*``（除豁免路径）：无有效令牌一律 401。

    **刻意不用 ``BaseHTTPMiddleware``**（本项目有两条 SSE + 60MB 视频上传）：
    - BaseHTTPMiddleware 会把下游响应体包进内存流任务，逐块中转，SSE 延迟增加，
      且客户端断开（http.disconnect）传播不可靠——告警 SSE 的 finally 退订
      （AlertHub 订阅释放）会因此泄漏；纯 ASGI 直接透传 receive/send，无此问题；
    - 纯 ASGI 不接触请求体，视频上传不受影响。

    令牌通道：``Authorization: Bearer <token>`` 或 ``?token=<token>``
    （EventSource 无法自定义请求头，SSE 必须走查询参数）。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    @staticmethod
    def _exempt_paths() -> frozenset[str]:
        prefix = settings.api_prefix
        return frozenset({f"{prefix}/health", f"{prefix}/auth/login"})

    @staticmethod
    def _bearer_token(headers: list[tuple[bytes, bytes]]) -> str:
        for name, value in headers:
            if name == b"authorization":
                text = value.decode("latin-1", "replace").strip()
                if text[:7].lower() == "bearer ":
                    return text[7:].strip()
        return ""

    @staticmethod
    def _query_token(query_string: bytes) -> str:
        for name, value in urllib.parse.parse_qsl(query_string.decode("latin-1")):
            if name == "token":
                return value
        return ""

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if not path.startswith(settings.api_prefix) or path in self._exempt_paths():
            return await self.app(scope, receive, send)

        token = self._bearer_token(scope.get("headers") or []) or self._query_token(
            scope.get("query_string") or b""
        )
        username = verify_token(token) if token else None
        if username is None:
            body = json.dumps(err(401, "未登录或登录已过期"), ensure_ascii=False).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json; charset=utf-8"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        # 已登录信息放入 scope["state"]，路由内可用 request.state.username 读取
        scope.setdefault("state", {})["username"] = username
        await self.app(scope, receive, send)
