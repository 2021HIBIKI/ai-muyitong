"""pytest 共享夹具：隔离数据库的 TestClient。

鉴权约定（阶段 9）：``client`` 夹具在启动时用 ``settings`` 中的账号登录一次并设置
全局 ``Authorization`` 头——**全部既有 API 测试零改动**；``anon_client`` 剥离令牌，
专供鉴权用例（httpx 合并头时按请求头优先，测试内仍可单发覆盖）。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """每个测试使用独立临时数据库的 TestClient（已登录）。"""
    from app.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    # 显式关闭后台巡检：不能只靠默认值（.env 会覆盖默认，测试必须确定）
    monkeypatch.setattr(settings, "patrol_enabled", False)

    from app.main import app

    with TestClient(app) as c:
        login = c.post(
            "/api/auth/login",
            json={"username": settings.auth_username, "password": settings.auth_password},
        )
        assert login.status_code == 200, login.text
        c.headers["Authorization"] = "Bearer " + login.json()["data"]["token"]
        yield c


@pytest.fixture
def anon_client(client):
    """未携带令牌的同一 TestClient（鉴权用例专用；不影响其余用例）。"""
    client.headers.pop("Authorization", None)
    return client
