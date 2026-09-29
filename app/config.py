"""应用配置：统一从环境变量 / .env 读取，禁止在业务代码里硬编码密钥与路径。"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/config.py 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """全局配置。字段名对应环境变量（大小写不敏感），也可从 .env 读取。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用
    app_name: str = "AI牧易通 · 数字牧业全链平台"
    api_prefix: str = "/api"
    db_path: str = "data/app.db"

    # GLM LLM（OpenAI 兼容）。留空 key 时 Agent 走本地规则降级。
    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    glm_model: str = "glm-4.7-flash"
    llm_max_retries: int = 2  # 瞬时错误（429/5xx/超时）重试次数
    llm_retry_backoff: float = 1.0  # 指数退避基准秒数（1s、2s、…）

    # 视觉模型：羊专用权重优先，不存在则用 COCO 兜底；设备 auto|cuda|cpu。
    vision_model_path: str = "models/yolo11n_all.pt"
    vision_fallback_model: str = "yolov8s.pt"
    vision_device: str = "auto"
    vision_confidence: float = 0.35
    # 单帧最大检测数：ultralytics 默认 300，航拍大群会被截断（实测三帧均恰好 300）
    vision_max_det: int = 600

    # 视频计数（阶段 6 A1）
    video_max_bytes: int = 60 * 1024 * 1024  # 上传体积上限
    video_max_seconds: float = 15.0  # 时长上限（按 fps × 帧数估算）
    video_stride: int = 2  # 检测步长（每 N 帧检测一次；跟踪仍稳定，耗时减半）
    video_max_previews: int = 32  # 返回的标注帧数量上限
    # 越线线位（归一化 0–1）。方向约定：线位方向决定 in/out —— 本例取「自右向左」的
    # 横线，使**走向镜头（画面下方）记为 inbound**；实测公路素材 y=40% 处进出最清晰。
    line_x1: float = 1.0
    line_y1: float = 0.40
    line_x2: float = 0.0
    line_y2: float = 0.40

    # 主动巡检（阶段 7 A3）：默认关闭，配置开启后后台定时跑；手动触发始终可用
    patrol_enabled: bool = False
    patrol_interval_seconds: float = 60.0
    patrol_cooldown_seconds: float = 300.0  # 同级告警冷却（期内抑制重复推送）
    patrol_image_path: str = ""  # 巡检点位图片（仅库中无草场时的单点位兜底；有草场则逐场巡检）

    # 载畜量：标准羊单位日食干物质（kg）
    daily_intake_per_sheep: float = 2.0

    # Agent 编排最大轮数
    agent_max_rounds: int = 8

    # 登录鉴权（真实校验：用户名/密码 + 标准库 HMAC 签名令牌，无第三方依赖）
    # 演示默认 admin / admin123；正式部署必须修改口令与令牌密钥（见 .env.example）
    auth_username: str = "admin"
    auth_password: str = "admin123"
    auth_token_secret: str = "muyitong-demo-secret-change-me"
    auth_token_ttl_seconds: float = 43200.0  # 令牌有效期（秒），默认 12 小时

    # 覆盖度告警阈值：>=warning 为 normal；[danger, warning) 为 warning；<danger 为 danger
    alert_warning_threshold: float = 0.60
    alert_danger_threshold: float = 0.35

    # ---- 派生路径 ----
    @property
    def db_path_resolved(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def vision_model_path_resolved(self) -> Path:
        p = Path(self.vision_model_path)
        return p if p.is_absolute() else BASE_DIR / p


settings = Settings()
