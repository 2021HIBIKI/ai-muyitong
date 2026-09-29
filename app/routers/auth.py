"""登录接口：签发无状态令牌（真实校验，账号存于 users 表）。令牌实现见 ``app/auth.py``。

认证口径：
- 账号由 ``users`` 表管理（首次启动用 .env 凭据播种管理员，之后在用户管理页维护）；
- 口令只存 PBKDF2 加盐哈希，常量时间比对，失败不回显口令；
- 登录失败返回统一信封 ``err(401, ...)``（与全站一致）；而中间件对未登录访问的
  拦截是 HTTP 401（基础设施层，供前端全局识别登录态失效）。
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.auth import make_token, verify_password
from app.data import repository
from app.data.db import connect
from app.schemas import err, ok

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """登录入参：必填且非空（缺字段/空串由 FastAPI 直接 422）。"""

    username: str = Field(..., min_length=1, max_length=100, description="用户名")
    password: str = Field(..., min_length=1, max_length=200, description="密码")


@router.post("/login")
def login(payload: LoginRequest) -> dict:
    """校验账号口令，成功签发令牌（默认有效期 12 小时，可配）。"""
    conn = connect()
    try:
        user = repository.get_user_by_username(conn, payload.username)
    finally:
        conn.close()
    if user is None or not verify_password(payload.password, user["password_hash"]):
        return err(401, "用户名或密码错误")
    issued = make_token(user["username"])
    issued["role"] = user["role"]
    return ok(issued)


@router.get("/me")
def me(request: Request) -> dict:
    """当前登录账号（中间件已校验令牌；账号被删除时返回 HTTP 401 触发重新登录）。"""
    username = getattr(request.state, "username", None)
    conn = connect()
    try:
        user = repository.get_user_by_username(conn, username) if username else None
    finally:
        conn.close()
    if user is None:
        return JSONResponse(status_code=401, content=err(401, "账号不存在或已被删除，请重新登录"))
    return ok({"username": user["username"], "role": user["role"]})
