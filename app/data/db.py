"""SQLite 连接与建表（幂等）。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import settings

# 时间列采用 ISO8601（UTC，毫秒），符合设计规范。
SCHEMA = """
CREATE TABLE IF NOT EXISTS count_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    region TEXT,
    count INTEGER NOT NULL,
    per_class TEXT,
    annotated_image TEXT
);
CREATE TABLE IF NOT EXISTS grassland_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    region TEXT,
    coverage REAL,
    greenness REAL
);
CREATE TABLE IF NOT EXISTS carrying_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    params TEXT,
    recommended_stocking REAL,
    rotation_plan TEXT
);
CREATE TABLE IF NOT EXISTS decision_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    question TEXT,
    tool_calls TEXT,
    decision_json TEXT
);
CREATE TABLE IF NOT EXISTS environment_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    region TEXT,
    temperature REAL,
    humidity REAL
);
-- 草场：一个草场由多个分区的照片集合构成（照片编号：草场-分区-序号，如 P1-A-01）
CREATE TABLE IF NOT EXISTS pastures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    region TEXT,
    area_mu REAL NOT NULL DEFAULT 0,
    note TEXT
);
-- 草场照片（编号唯一标识归属：哪一草场的哪一分区）
CREATE TABLE IF NOT EXISTS pasture_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    pasture_id INTEGER NOT NULL REFERENCES pastures(id) ON DELETE CASCADE,
    plot_code TEXT NOT NULL,
    image_code TEXT NOT NULL,
    image_path TEXT,
    note TEXT,
    UNIQUE (pasture_id, image_code)
);
CREATE INDEX IF NOT EXISTS idx_pasture_images_pasture ON pasture_images(pasture_id);
-- 巡检告警（阶段 7 A3：主动巡检 → 告警落库 → 大屏实时弹出；巡检单位为草场，逐场告警）
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    level TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT,
    metrics_json TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    pasture_id INTEGER
);
-- 账号（阶段 10：用户管理。口令只存 PBKDF2 加盐哈希；角色控制账号管理入口）
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin'
);
"""

# 增量迁移：老库补齐新列（幂等，仅加列不删列）。列定义只允许代码内常量。
_MIGRATIONS: dict[str, dict[str, str]] = {
    # 覆盖度记录关联草场/分区/图片编号（为空表示未关联草场的独立分析）
    "grassland_metrics": {
        "pasture_id": "INTEGER",
        "plot_code": "TEXT",
        "image_code": "TEXT",
        "image_path": "TEXT",
    },
    # 决策日志保存 SSE 思考事件（供历史对话回放思考过程）
    "decision_logs": {
        "events": "TEXT",
    },
    # 告警归属草场（巡检逐草场判定：冷却/升级按本场历史独立）
    "alerts": {
        "pasture_id": "INTEGER",
    },
}


def _migrate(conn: sqlite3.Connection) -> None:
    """为既有库补充缺失列（幂等）。"""
    for table, columns in _MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, ddl in columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """打开连接；默认使用配置中的数据库路径，并自动创建父目录。"""
    path = Path(db_path) if db_path is not None else settings.db_path_resolved
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """建表（幂等）+ 增量迁移（老库补列）。"""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()
