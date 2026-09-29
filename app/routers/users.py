"""账号管理（用户管理页）：列表 / 新建 / 删除 / 重置密码 —— 仅管理员可操作。

简单权限模型（如实说明，规划中扩展见页面说明）：
- 角色：管理员（admin）/ 放牧员（herder）/ 巡护员（ranger）；
- 本阶段角色只控制「账号管理」入口（本路由要求管理员，其余页面登录即可访问）；
- 口令只存 PBKDF2 加盐哈希，任何响应与日志不回显口令。
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.auth import hash_password
from app.data import repository
from app.data.db import connect
from app.schemas import err, ok

router = APIRouter(prefix="/api/users", tags=["users"])

ROLES = {"admin": "管理员", "herder": "放牧员", "ranger": "巡护员"}

RoleLiteral = Literal["admin", "herder", "ranger"]


class CreateUserRequest(BaseModel):
    """新建账号入参（用户名唯一，口令最短 6 位）。"""

    username: str = Field(..., min_length=2, max_length=32, pattern=r"^[A-Za-z0-9_.\-]+$",
                          description="用户名（字母/数字/下划线/点/横线）")
    password: str = Field(..., min_length=6, max_length=64, description="初始口令（≥6 位）")
    role: RoleLiteral = Field("herder", description="角色：admin/herder/ranger")


class ResetPasswordRequest(BaseModel):
    password: str = Field(..., min_length=6, max_length=64, description="新口令（≥6 位）")


def _public(row) -> dict:
    """账号对外字段（绝不包含 password_hash）。"""
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "role_label": ROLES.get(row["role"], row["role"]),
        "created_at": row["created_at"],
    }


def _current_user(conn, request: Request):
    username = getattr(request.state, "username", None)
    return repository.get_user_by_username(conn, username) if username else None


def _admin_guard(conn, request: Request):
    """返回 (admin_row, error)；非管理员一律 403，账号已删除按未登录处理。"""
    user = _current_user(conn, request)
    if user is None:
        return None, err(401, "账号不存在或已被删除，请重新登录")
    if user["role"] != "admin":
        return None, err(403, "仅管理员可管理账号")
    return user, None


@router.get("")
def list_accounts(request: Request) -> dict:
    """账号列表（含角色字典，供前端下拉与标签渲染）。"""
    conn = connect()
    try:
        _, bad = _admin_guard(conn, request)
        if bad:
            return bad
        return ok({
            "users": [_public(u) for u in repository.list_users(conn)],
            "roles": ROLES,
        })
    finally:
        conn.close()


@router.post("")
def create_account(payload: CreateUserRequest, request: Request) -> dict:
    conn = connect()
    try:
        _, bad = _admin_guard(conn, request)
        if bad:
            return bad
        if repository.get_user_by_username(conn, payload.username) is not None:
            return err(400, f"账号「{payload.username}」已存在")
        user_id = repository.insert_user(
            conn, payload.username, hash_password(payload.password), role=payload.role
        )
        return ok({"user": _public(repository.get_user(conn, user_id))})
    finally:
        conn.close()


@router.delete("/{user_id}")
def delete_account(user_id: int, request: Request) -> dict:
    """删除账号：不能删自己；至少保留一个管理员。"""
    conn = connect()
    try:
        admin, bad = _admin_guard(conn, request)
        if bad:
            return bad
        target = repository.get_user(conn, user_id)
        if target is None:
            return err(404, "账号不存在")
        if target["id"] == admin["id"]:
            return err(400, "不能删除当前登录账号")
        if target["role"] == "admin" and repository.count_admins(conn) <= 1:
            return err(400, "至少保留一个管理员账号")
        repository.delete_user(conn, user_id)
        return ok({"deleted": user_id})
    finally:
        conn.close()


@router.post("/{user_id}/password")
def reset_password(user_id: int, payload: ResetPasswordRequest, request: Request) -> dict:
    """重置指定账号口令（管理员操作）。"""
    conn = connect()
    try:
        _, bad = _admin_guard(conn, request)
        if bad:
            return bad
        if repository.get_user(conn, user_id) is None:
            return err(404, "账号不存在")
        repository.update_user_password(conn, user_id, hash_password(payload.password))
        return ok({"updated": user_id})
    finally:
        conn.close()
