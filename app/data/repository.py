"""数据仓储：四类实体的插入与查询 CRUD。函数均接收显式连接以便事务与测试隔离。"""
from __future__ import annotations

import json
import sqlite3
from typing import Any


def insert_count_record(
    conn: sqlite3.Connection,
    *,
    region: str | None,
    count: int,
    per_class: dict[str, int] | None,
    annotated_image: str | None,
    created_at: str | None = None,
) -> int:
    """插入计数记录；created_at 缺省取数据库当前时间（模拟数据可传入历史时间戳）。"""
    if created_at is not None:
        cur = conn.execute(
            "INSERT INTO count_records (created_at, region, count, per_class, annotated_image)"
            " VALUES (?, ?, ?, ?, ?)",
            (created_at, region, count, json.dumps(per_class, ensure_ascii=False), annotated_image),
        )
    else:
        cur = conn.execute(
            "INSERT INTO count_records (region, count, per_class, annotated_image) VALUES (?, ?, ?, ?)",
            (region, count, json.dumps(per_class, ensure_ascii=False), annotated_image),
        )
    return cur.lastrowid


def insert_grassland_metric(
    conn: sqlite3.Connection,
    *,
    region: str | None,
    coverage: float,
    greenness: float,
    created_at: str | None = None,
    pasture_id: int | None = None,
    plot_code: str | None = None,
    image_code: str | None = None,
    image_path: str | None = None,
) -> int:
    """插入覆盖度记录；草场相关字段可空（为空表示未关联草场的独立分析）。"""
    cur = conn.execute(
        "INSERT INTO grassland_metrics"
        " (created_at, region, coverage, greenness, pasture_id, plot_code, image_code, image_path)"
        " VALUES (COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), ?, ?, ?, ?, ?, ?, ?)",
        (created_at, region, coverage, greenness, pasture_id, plot_code, image_code, image_path),
    )
    return cur.lastrowid


def insert_carrying_result(
    conn: sqlite3.Connection,
    *,
    params: dict[str, Any],
    recommended_stocking: float,
    rotation_plan: str,
    created_at: str | None = None,
) -> int:
    if created_at is not None:
        cur = conn.execute(
            "INSERT INTO carrying_results (created_at, params, recommended_stocking, rotation_plan)"
            " VALUES (?, ?, ?, ?)",
            (created_at, json.dumps(params, ensure_ascii=False), recommended_stocking, rotation_plan),
        )
    else:
        cur = conn.execute(
            "INSERT INTO carrying_results (params, recommended_stocking, rotation_plan) VALUES (?, ?, ?)",
            (json.dumps(params, ensure_ascii=False), recommended_stocking, rotation_plan),
        )
    return cur.lastrowid


def insert_decision_log(
    conn: sqlite3.Connection,
    *,
    question: str,
    tool_calls: list[str],
    decision_json: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> int:
    """写入决策日志；events 为 SSE 思考事件列表（历史对话回放思考过程用，可空）。"""
    cur = conn.execute(
        "INSERT INTO decision_logs (created_at, question, tool_calls, decision_json, events)"
        " VALUES (COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), ?, ?, ?, ?)",
        (
            created_at,
            question,
            json.dumps(tool_calls, ensure_ascii=False),
            json.dumps(decision_json, ensure_ascii=False),
            json.dumps(events, ensure_ascii=False) if events is not None else None,
        ),
    )
    return cur.lastrowid


def get_decision_log(conn: sqlite3.Connection, log_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM decision_logs WHERE id = ?", (log_id,)).fetchone()


def insert_environment_record(
    conn: sqlite3.Connection,
    *,
    region: str | None,
    temperature: float,
    humidity: float,
    created_at: str | None = None,
) -> int:
    if created_at is not None:
        cur = conn.execute(
            "INSERT INTO environment_records (created_at, region, temperature, humidity) VALUES (?, ?, ?, ?)",
            (created_at, region, temperature, humidity),
        )
    else:
        cur = conn.execute(
            "INSERT INTO environment_records (region, temperature, humidity) VALUES (?, ?, ?)",
            (region, temperature, humidity),
        )
    return cur.lastrowid


def list_count_records(
    conn: sqlite3.Connection, limit: int = 100, ascending: bool = False
) -> list[sqlite3.Row]:
    """ascending=True 时返回「最近 limit 条」并按时间升序（供趋势图），否则按 id 降序。"""
    if ascending:
        return conn.execute(
            "SELECT * FROM (SELECT * FROM count_records ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (limit,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM count_records ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


def list_grassland_metrics(
    conn: sqlite3.Connection, limit: int = 100, ascending: bool = False
) -> list[sqlite3.Row]:
    """ascending=True 时返回「最近 limit 条」并按时间升序（供趋势图），否则按 id 降序。"""
    if ascending:
        return conn.execute(
            "SELECT * FROM (SELECT * FROM grassland_metrics ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (limit,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM grassland_metrics ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


def list_carrying_results(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM carrying_results ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


def list_decision_logs(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM decision_logs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


def list_environment_records(
    conn: sqlite3.Connection, limit: int = 100, ascending: bool = False
) -> list[sqlite3.Row]:
    """ascending=True 时返回「最近 limit 条」并按时间升序（供折线图），否则按 id 降序。"""
    if ascending:
        return conn.execute(
            "SELECT * FROM (SELECT * FROM environment_records ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (limit,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM environment_records ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


# ---------------------------------------------------------------------------
# 草场与草场照片（多草场管理：一个草场 = 多个分区编号照片的集合）
# ---------------------------------------------------------------------------

def insert_pasture(
    conn: sqlite3.Connection,
    *,
    code: str,
    name: str,
    region: str | None = None,
    area_mu: float = 0.0,
    note: str | None = None,
    created_at: str | None = None,
) -> int:
    """新建草场（code 唯一，重复时抛 sqlite3.IntegrityError 由调用方转 4xx）。"""
    cur = conn.execute(
        "INSERT INTO pastures (created_at, code, name, region, area_mu, note)"
        " VALUES (COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), ?, ?, ?, ?, ?)",
        (created_at, code, name, region, area_mu, note),
    )
    return cur.lastrowid


def list_pastures(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    """草场清单（按编号顺序，供管理界面）。"""
    return conn.execute(
        "SELECT * FROM pastures ORDER BY code ASC LIMIT ?", (limit,)
    ).fetchall()


def get_pasture(conn: sqlite3.Connection, pasture_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM pastures WHERE id = ?", (pasture_id,)).fetchone()


def get_pasture_by_code(conn: sqlite3.Connection, code: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM pastures WHERE code = ?", (code,)).fetchone()


def delete_pasture(conn: sqlite3.Connection, pasture_id: int) -> None:
    """删除草场及其照片与覆盖度记录（该草场的分析数据随草场一并移除）。"""
    conn.execute("DELETE FROM pasture_images WHERE pasture_id = ?", (pasture_id,))
    conn.execute("DELETE FROM grassland_metrics WHERE pasture_id = ?", (pasture_id,))
    conn.execute("DELETE FROM pastures WHERE id = ?", (pasture_id,))


def insert_pasture_image(
    conn: sqlite3.Connection,
    *,
    pasture_id: int,
    plot_code: str,
    image_code: str,
    image_path: str | None = None,
    note: str | None = None,
    created_at: str | None = None,
) -> int:
    """登记草场照片；(pasture_id, image_code) 已存在时更新分区与路径（重新登记）。"""
    cur = conn.execute(
        "INSERT INTO pasture_images (created_at, pasture_id, plot_code, image_code, image_path, note)"
        " VALUES (COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), ?, ?, ?, ?, ?)"
        " ON CONFLICT(pasture_id, image_code) DO UPDATE SET"
        " plot_code = excluded.plot_code, image_path = excluded.image_path, note = excluded.note",
        (created_at, pasture_id, plot_code, image_code, image_path, note),
    )
    return cur.lastrowid


def list_pasture_images(conn: sqlite3.Connection, pasture_id: int) -> list[sqlite3.Row]:
    """草场照片清单（按分区、编号排序）。"""
    return conn.execute(
        "SELECT * FROM pasture_images WHERE pasture_id = ? ORDER BY plot_code ASC, image_code ASC",
        (pasture_id,),
    ).fetchall()


def count_pasture_images(conn: sqlite3.Connection, pasture_id: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM pasture_images WHERE pasture_id = ?", (pasture_id,)
    ).fetchone()[0]


def latest_image_metrics(
    conn: sqlite3.Connection, pasture_id: int | None = None
) -> list[sqlite3.Row]:
    """各（分区, 图片）的最新覆盖度记录（供「图片集合 → 草场」归并）。

    pasture_id 给定时只看该草场；否则返回全部已关联草场的记录。
    无图片编号的分区级记录按空图片聚合（plot_coverage_map 会兜底使用）。
    """
    if pasture_id is not None:
        return conn.execute(
            "SELECT g.* FROM grassland_metrics g"
            " JOIN (SELECT plot_code, COALESCE(image_code, '') AS ic, MAX(id) AS mid"
            "       FROM grassland_metrics WHERE pasture_id = ? AND plot_code IS NOT NULL"
            "       GROUP BY plot_code, COALESCE(image_code, '')) t ON g.id = t.mid",
            (pasture_id,),
        ).fetchall()
    return conn.execute(
        "SELECT g.* FROM grassland_metrics g"
        " JOIN (SELECT pasture_id AS pid, plot_code, COALESCE(image_code, '') AS ic, MAX(id) AS mid"
        "       FROM grassland_metrics WHERE pasture_id IS NOT NULL AND plot_code IS NOT NULL"
        "       GROUP BY pasture_id, plot_code, COALESCE(image_code, '')) t ON g.id = t.mid",
    ).fetchall()


def list_metrics_for_pasture(
    conn: sqlite3.Connection, pasture_id: int, limit: int = 200, ascending: bool = False
) -> list[sqlite3.Row]:
    """指定草场的覆盖度记录（ascending=True 时返回「最近 limit 条」并按时间升序）。"""
    if ascending:
        return conn.execute(
            "SELECT * FROM (SELECT * FROM grassland_metrics WHERE pasture_id = ?"
            " ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (pasture_id, limit),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM grassland_metrics WHERE pasture_id = ? ORDER BY id DESC LIMIT ?",
        (pasture_id, limit),
    ).fetchall()


def daily_coverage_series(
    conn: sqlite3.Connection, days: int = 30, *, pasture_id: int
) -> list[sqlite3.Row]:
    """某草场按日聚合的覆盖度序列（当日该草场各条记录均值），升序返回。

    ``pasture_id`` 为必填关键字参数：**草场彼此独立**，不存在跨草场合并的覆盖度
    序列（未关联草场的独立分析记录也不计入，避免污染草场趋势口径）。
    """
    rows = conn.execute(
        "SELECT substr(created_at, 1, 10) AS day, AVG(coverage) AS coverage,"
        " AVG(greenness) AS greenness, COUNT(*) AS n"
        " FROM grassland_metrics WHERE pasture_id = ?"
        " GROUP BY day ORDER BY day DESC LIMIT ?",
        (pasture_id, days),
    ).fetchall()
    return list(reversed(rows))


# ---------------------------------------------------------------------------
# 巡检告警（阶段 7 A3）
# ---------------------------------------------------------------------------

def insert_alert(
    conn: sqlite3.Connection,
    *,
    level: str,
    title: str,
    message: str | None = None,
    metrics: dict[str, Any] | None = None,
    acknowledged: bool = False,
    created_at: str | None = None,
    pasture_id: int | None = None,
) -> int:
    """写入一条巡检告警（pasture_id 标记告警归属草场，供逐场冷却/升级判定）。"""
    cur = conn.execute(
        "INSERT INTO alerts (created_at, level, title, message, metrics_json, acknowledged, pasture_id)"
        " VALUES (COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), ?, ?, ?, ?, ?, ?)",
        (
            created_at,
            level,
            title,
            message,
            json.dumps(metrics, ensure_ascii=False) if metrics is not None else None,
            1 if acknowledged else 0,
            pasture_id,
        ),
    )
    return cur.lastrowid


def list_alerts(
    conn: sqlite3.Connection,
    limit: int = 50,
    ascending: bool = False,
    pasture_id: int | None = None,
) -> list[sqlite3.Row]:
    """告警列表（ascending=True 时返回「最近 limit 条」并按时间升序，供大屏时间轴）。

    pasture_id 给定时只看该草场的告警（巡检逐场判定：上一等级与冷却时间按本场历史）。
    """
    if pasture_id is not None:
        return conn.execute(
            "SELECT * FROM alerts WHERE pasture_id = ? ORDER BY id DESC LIMIT ?",
            (pasture_id, limit),
        ).fetchall()
    if ascending:
        return conn.execute(
            "SELECT * FROM (SELECT * FROM alerts ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (limit,),
        ).fetchall()
    return conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def get_alert(conn: sqlite3.Connection, alert_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()


def acknowledge_alert(conn: sqlite3.Connection, alert_id: int) -> bool:
    """确认告警；返回是否命中记录（未命中由调用方转 404）。"""
    cur = conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# 账号（阶段 10：用户管理）
# ---------------------------------------------------------------------------

def insert_user(
    conn: sqlite3.Connection, username: str, password_hash: str, role: str = "admin"
) -> int:
    """新增账号（username 唯一，重复由调用方先查后判）。"""
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        (username, password_hash, role),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_user_by_username(conn: sqlite3.Connection, username: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def get_user(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def list_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM users ORDER BY id ASC").fetchall()


def count_admins(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'").fetchone()
    return int(row["n"])


def delete_user(conn: sqlite3.Connection, user_id: int) -> bool:
    """删除账号；返回是否命中记录（未命中由调用方转 404）。"""
    cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    return cur.rowcount > 0


def update_user_password(conn: sqlite3.Connection, user_id: int, password_hash: str) -> bool:
    cur = conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
    )
    conn.commit()
    return cur.rowcount > 0


def ensure_admin_user(conn: sqlite3.Connection, username: str, password: str) -> bool:
    """首次启动（账号表为空/该账号不存在）时用 .env 凭据播种管理员账号。

    已存在则不动（改密走用户管理页的重置密码；.env 口令只在账号不存在时生效）。
    返回是否新建。
    """
    from app.auth import hash_password

    if get_user_by_username(conn, username) is not None:
        return False
    insert_user(conn, username, hash_password(password), role="admin")
    return True
