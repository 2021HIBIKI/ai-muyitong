"""系统信息接口测试（阶段 9.7）：运行时配置安全子集——有值、受鉴权、不含密钥。"""
from __future__ import annotations


def test_system_info_requires_auth(anon_client):
    resp = anon_client.get("/api/system/info")
    assert resp.status_code == 401
    assert resp.json()["code"] == 401


def test_system_info_reports_real_config(client):
    """字段与 settings 实时一致（阈值/巡检/视觉/大模型/载畜量口径）。"""
    from app.config import settings

    resp = client.get("/api/system/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0 and body["message"] == "ok"
    data = body["data"]
    assert data["app_name"] == settings.app_name
    assert data["api_prefix"] == settings.api_prefix
    assert data["alert"] == {
        "warning_threshold": settings.alert_warning_threshold,
        "danger_threshold": settings.alert_danger_threshold,
    }
    assert data["patrol"]["enabled"] is False  # 测试夹具强制关闭巡检
    assert data["patrol"]["interval_seconds"] == settings.patrol_interval_seconds
    assert data["patrol"]["cooldown_seconds"] == settings.patrol_cooldown_seconds
    assert data["vision"]["model_file"] == "yolo11n_all.pt"  # 只给文件名，不给路径
    assert data["vision"]["fallback_model"] == "yolov8s.pt"
    assert data["vision"]["device"] == settings.vision_device
    assert data["llm"]["model"] == settings.glm_model
    assert isinstance(data["llm"]["configured"], bool)
    assert data["llm"]["max_retries"] == settings.llm_max_retries
    assert data["llm"]["retry_backoff"] == settings.llm_retry_backoff
    assert data["agent_max_rounds"] == settings.agent_max_rounds
    assert data["video"]["stride"] == settings.video_stride
    assert data["daily_intake_per_sheep"] == settings.daily_intake_per_sheep


def test_system_info_hides_secrets(client, monkeypatch):
    """绝不回显密钥：GLM key 值、鉴权口令、令牌密钥、DB 路径全部缺席。"""
    from app.config import settings

    monkeypatch.setattr(settings, "glm_api_key", "super-secret-glm-key-xyz")
    monkeypatch.setattr(settings, "db_path", "data/super-secret-db.db")

    text = client.get("/api/system/info").text
    for secret in ("super-secret-glm-key-xyz", "super-secret-db.db",
                   settings.auth_password, settings.auth_token_secret):
        assert secret not in text, secret
    # 字段名也不出现（防结构泄漏）
    assert "glm_api_key" not in text and "auth_password" not in text
    assert "auth_token_secret" not in text and "db_path" not in text


def test_system_info_reports_storage_counts(client):
    """数据存储：各业务表记录数（只给标签与数量，不给路径）。"""
    storage = client.get("/api/system/info").json()["data"]["storage"]
    assert storage["pastures"] == {"label": "草场", "count": 3}  # 演示库播种 3 个草场
    assert storage["users"]["label"] == "账号"
    assert storage["users"]["count"] >= 1  # 播种管理员
    for key in ("alerts", "grassland_metrics", "decision_logs", "pasture_images"):
        assert isinstance(storage[key]["count"], int), key
        assert storage[key]["count"] >= 0


def test_system_info_envelope_shape(client):
    body = client.get("/api/system/info").json()
    assert set(body) == {"code", "data", "message"}
    assert set(body["data"]) >= {"app_name", "alert", "patrol", "vision", "llm",
                                 "video", "daily_intake_per_sheep", "storage"}
