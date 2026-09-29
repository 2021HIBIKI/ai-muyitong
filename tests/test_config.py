"""阶段 0.2：配置模块测试 —— 默认值、.env 覆盖、缺 key 不抛错。"""
from __future__ import annotations

from app.config import Settings, settings


def test_defaults():
    """默认值：不读 .env 时返回内置默认，GLM key 为空。"""
    s = Settings(_env_file=None)
    assert s.glm_api_key == ""
    assert s.glm_base_url.startswith("https://open.bigmodel.cn")
    assert s.glm_model == "glm-4.7-flash"
    assert s.api_prefix == "/api"
    assert s.db_path == "data/app.db"


def test_env_override(monkeypatch):
    """环境变量可覆盖默认值。"""
    monkeypatch.setenv("GLM_API_KEY", "secret-key-123")
    monkeypatch.setenv("GLM_MODEL", "glm-test")
    monkeypatch.setenv("VISION_CONFIDENCE", "0.5")
    s = Settings(_env_file=None)
    assert s.glm_api_key == "secret-key-123"
    assert s.glm_model == "glm-test"
    assert s.vision_confidence == 0.5


def test_missing_key_no_error():
    """缺 API key 时不得抛错，应返回空字符串供降级判断。"""
    s = Settings(_env_file=None)
    assert s.glm_api_key == ""
    assert isinstance(s.glm_api_key, str)


def test_derived_paths():
    """派生路径：相对路径解析为项目根下的绝对路径。"""
    s = Settings(_env_file=None)
    assert s.db_path_resolved.is_absolute()
    assert s.db_path_resolved.name == "app.db"
    assert s.vision_model_path_resolved.is_absolute()


def test_singleton_is_settings():
    """模块级单例存在且为 Settings 实例。"""
    assert isinstance(settings, Settings)
    assert settings.app_name


def test_auth_defaults():
    """鉴权默认值：演示账号开箱即用（README/.env.example 注明正式部署必须改）。"""
    s = Settings(_env_file=None)
    assert s.auth_username == "admin"
    assert s.auth_password == "admin123"
    assert s.auth_token_secret
    assert s.auth_token_ttl_seconds == 43200.0


def test_auth_env_override(monkeypatch):
    """鉴权配置可被环境变量覆盖。"""
    monkeypatch.setenv("AUTH_USERNAME", "grader")
    monkeypatch.setenv("AUTH_PASSWORD", "s3cret!")
    monkeypatch.setenv("AUTH_TOKEN_TTL_SECONDS", "60")
    s = Settings(_env_file=None)
    assert s.auth_username == "grader"
    assert s.auth_password == "s3cret!"
    assert s.auth_token_ttl_seconds == 60.0
