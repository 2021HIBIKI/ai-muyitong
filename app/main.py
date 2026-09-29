"""FastAPI 入口：应用装配、生命周期、健康检查、静态资源挂载。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders

from app.auth import AuthMiddleware
from app.config import settings
from app.data.db import connect, init_db
from app.routers import agent, auth, dashboard, pasture, patrol, system, users, vision
from app.schemas import ok

BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时建表（幂等）；演示数据表全空时播种模拟序列，保证大屏开箱即有数据。
    init_db()
    from app.data.repository import ensure_admin_user
    from app.data.simulator import populate_if_empty

    conn = connect()
    try:
        # 账号表无该账号时用 .env 凭据播种管理员（之后在用户管理页维护账号）
        ensure_admin_user(conn, settings.auth_username, settings.auth_password)
        populate_if_empty(conn)
    finally:
        conn.close()

    # 主动巡检（阶段 7）：仅当配置开启时创建后台任务；关闭时干净退出
    from app.agent.patrol import start_patrol_task, stop_patrol_task

    patrol_task = await start_patrol_task()
    try:
        yield
    finally:
        await stop_patrol_task(patrol_task)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# 登录鉴权：保护 /api/*（豁免 /api/health 与 /api/auth/login）。
# 纯 ASGI 中间件——不缓冲流式响应，SSE（告警/思考流）与视频上传不受影响。
app.add_middleware(AuthMiddleware)


class NoCacheStaticMiddleware:
    """静态资源（页面/素材/样例图）响应加 ``Cache-Control: no-cache``。

    no-cache 语义是「用前必须重新验证」：浏览器仍带 ETag/Last-Modified 询问，
    未变时 304 走本地缓存；**变化后普通刷新即可拿到新文件**。避免浏览器启发式
    缓存长期吃旧图（演示前替换样例/页面后必须立刻可见）。
    仅作用于非 ``/api/*`` 路径（接口响应不受影响）。
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http" or scope.get("path", "").startswith(settings.api_prefix):
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["cache-control"] = "no-cache"
            await send(message)

        await self.app(scope, receive, send_wrapper)


app.add_middleware(NoCacheStaticMiddleware)


@app.get(f"{settings.api_prefix}/health")
def health() -> dict:
    return ok({"status": "ok"})


app.include_router(vision.router)
app.include_router(agent.router)
app.include_router(dashboard.router)
app.include_router(pasture.router)
app.include_router(patrol.router)
app.include_router(auth.router)
app.include_router(system.router)
app.include_router(users.router)


# 样例图（供前端「快速样例」按钮直接引用；目录存在时挂载）。
# 注意：必须先于 "/" 挂载注册，否则会被根挂载拦截。
samples_dir = BASE_DIR / "data" / "samples"
if samples_dir.exists():
    app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="samples")

# 大屏 / 后台静态资源（static 目录存在时挂载）。
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
