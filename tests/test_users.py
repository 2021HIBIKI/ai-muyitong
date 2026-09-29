"""用户管理（阶段 10）：账号 CRUD、角色权限、口令重置、哈希安全。

简单权限模型：仅管理员可维护账号（其余接口登录即可用）；口令只存 PBKDF2
加盐哈希，响应绝不回显；删除账号后其令牌在 /me 处即刻失效（前端回登录屏）。
"""
from __future__ import annotations

from app.config import settings


def _login(c, username, password):
    return c.post("/api/auth/login", json={"username": username, "password": password}).json()


def _token(c, username, password) -> str:
    body = _login(c, username, password)
    assert body["code"] == 0, body
    return body["data"]["token"]


def _as(token) -> dict:
    return {"Authorization": "Bearer " + token}


def _create(c, username, password="123456", role=None) -> dict:
    payload = {"username": username, "password": password}
    if role:
        payload["role"] = role
    return c.post("/api/users", json=payload).json()


# ---------------------------------------------------------------------------
# 播种与列表
# ---------------------------------------------------------------------------

def test_admin_seeded_on_startup(client):
    """首次启动用 .env 凭据播种管理员；列表不含任何哈希字段。"""
    body = client.get("/api/users").json()
    assert body["code"] == 0
    users = body["data"]["users"]
    assert body["data"]["roles"] == {"admin": "管理员", "herder": "放牧员", "ranger": "巡护员"}
    assert len(users) == 1
    admin = users[0]
    assert admin["username"] == settings.auth_username
    assert admin["role"] == "admin" and admin["role_label"] == "管理员"
    assert set(admin) == {"id", "username", "role", "role_label", "created_at"}
    assert "password_hash" not in body["message"] and "pbkdf2" not in str(users)


def test_users_endpoints_require_login(anon_client):
    assert anon_client.get("/api/users").status_code == 401
    assert anon_client.post("/api/users", json={"username": "x1", "password": "123456"}).status_code == 401


# ---------------------------------------------------------------------------
# 创建与角色权限
# ---------------------------------------------------------------------------

def test_create_account_and_non_admin_denied(client):
    """管理员建号成功且新账号可登录；非管理员访问账号管理 → 403。"""
    created = _create(client, "herder01", role="herder")
    assert created["code"] == 0
    user = created["data"]["user"]
    assert user["role"] == "herder" and user["role_label"] == "放牧员"

    token = _token(client, "herder01", "123456")
    denied = client.get("/api/users", headers=_as(token)).json()
    assert denied["code"] == 403 and "管理员" in denied["message"]
    denied2 = client.post(
        "/api/users", json={"username": "x01", "password": "123456"}, headers=_as(token)
    ).json()
    assert denied2["code"] == 403


def test_create_defaults_to_herder(client):
    user = _create(client, "def01")["data"]["user"]
    assert user["role"] == "herder"


def test_create_duplicate_and_validation(client):
    assert _create(client, "dup01")["code"] == 0
    dup = _create(client, "dup01")
    assert dup["code"] == 400 and "已存在" in dup["message"]
    # 口令过短 / 角色非法 / 用户名非法 → 422（结构校验）
    assert client.post("/api/users", json={"username": "ok01", "password": "123"}).status_code == 422
    assert client.post(
        "/api/users", json={"username": "ok02", "password": "123456", "role": "boss"}
    ).status_code == 422
    assert client.post("/api/users", json={"username": "空 格", "password": "123456"}).status_code == 422


# ---------------------------------------------------------------------------
# 口令重置
# ---------------------------------------------------------------------------

def test_password_reset(client):
    uid = _create(client, "reset01", password="old123456")["data"]["user"]["id"]
    assert _login(client, "reset01", "old123456")["code"] == 0

    assert client.post(f"/api/users/{uid}/password", json={"password": "new123456"}).json()["code"] == 0
    assert _login(client, "reset01", "old123456")["code"] == 401  # 旧口令失效
    assert _login(client, "reset01", "new123456")["code"] == 0
    # 过短口令 → 422；账号不存在 → 404
    assert client.post(f"/api/users/{uid}/password", json={"password": "123"}).status_code == 422
    assert client.post("/api/users/99999/password", json={"password": "123456"}).json()["code"] == 404


# ---------------------------------------------------------------------------
# 删除规则
# ---------------------------------------------------------------------------

def test_delete_rules(client):
    admin_id = client.get("/api/users").json()["data"]["users"][0]["id"]

    # 不能删除当前登录账号
    self_del = client.delete(f"/api/users/{admin_id}").json()
    assert self_del["code"] == 400 and "当前登录" in self_del["message"]

    # 删除普通账号后：其令牌在 /me 处即刻失效（HTTP 401 → 前端回登录屏）
    uid = _create(client, "todelete")["data"]["user"]["id"]
    token = _token(client, "todelete", "123456")
    assert client.get("/api/auth/me", headers=_as(token)).status_code == 200
    assert client.delete(f"/api/users/{uid}").json()["code"] == 0
    assert client.get("/api/auth/me", headers=_as(token)).status_code == 401

    # 至少保留一个管理员：第二个管理员可删；删到最后一个时拒绝
    second = _create(client, "admin02", role="admin")["data"]["user"]["id"]
    assert client.delete(f"/api/users/{second}").json()["code"] == 0
    last = client.delete(f"/api/users/{admin_id}").json()  # 此时只剩登录者自己 → 先撞「不能删自己」
    assert last["code"] == 400
    # 账号不存在 → 404
    assert client.delete("/api/users/99999").json()["code"] == 404
