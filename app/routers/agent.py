"""Agent 路由：POST /api/agent/chat（同步）与 /api/agent/stream（SSE 思考流）。

历史对话：每次问答（同步/流式）都落库 decision_logs（含 SSE 思考事件），
GET /api/agent/history 供 AI 助手界面回看，大屏决策流点击可跳转到对应回答。
"""
from __future__ import annotations

import json
import queue
import threading

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.orchestrator import AgentOrchestrator
from app.data import repository
from app.data.db import connect
from app.schemas import err, ok

router = APIRouter(prefix="/api/agent", tags=["agent"])

# 模块级编排器；测试中可替换。
orchestrator = AgentOrchestrator()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户自然语言问题")


def _log_decision(
    question: str,
    tool_calls: list[str],
    decision_dict: dict,
    events: list[dict] | None = None,
) -> None:
    conn = connect()
    try:
        repository.insert_decision_log(
            conn,
            question=question,
            tool_calls=tool_calls,
            decision_json=decision_dict,
            events=events,
        )
        conn.commit()
    finally:
        conn.close()


def _json_or(text, default):
    try:
        return json.loads(text) if text else default
    except (json.JSONDecodeError, TypeError):
        return default


def _log_payload(row) -> dict:
    """决策日志 → 接口形状（历史对话：问题、工具调用、决策、思考事件）。"""
    keys = row.keys()
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "question": row["question"],
        "tool_calls": _json_or(row["tool_calls"], []),
        "decision": _json_or(row["decision_json"], {}),
        "events": _json_or(row["events"], []) if "events" in keys else [],
    }


@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """自然语言问题 → 结构化决策 JSON（并写入 decision_logs）。"""
    try:
        decision = orchestrator.run(req.question)
        tool_calls = list(orchestrator.last_tool_calls)
    except Exception as e:
        return err(500, str(e))

    decision_dict = decision.model_dump()
    _log_decision(req.question, tool_calls, decision_dict)
    return ok(decision_dict)


@router.post("/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """SSE 事件流：thinking / tool_call / tool_result / answer 逐步推送（供思考流展示）。

    编排器在后台线程运行（同步阻塞），事件经队列实时转发；完成后落库 decision_logs。
    事件格式：`data: {"type": ..., ...}\\n\\n`。
    """

    def event_stream():
        events: queue.Queue = queue.Queue()

        def worker() -> None:
            collected: list[dict] = []   # 同步留存思考事件，供历史对话回放
            pending: list[dict] = []     # 暂存 answer 事件：落库后再推给前端

            def sink(event: dict) -> None:
                collected.append(event)
                if event.get("type") == "answer":
                    # 前端收到 answer 后会立即刷新历史对话列表；先落库再推送，
                    # 避免「列表里还没有这条记录」的竞态
                    pending.append(event)
                    return
                events.put(event)

            try:
                decision = orchestrator.run(req.question, on_event=sink)
                _log_decision(
                    req.question,
                    list(orchestrator.last_tool_calls),
                    decision.model_dump(),
                    collected,
                )
                for event in pending:
                    events.put(event)
            except Exception as e:  # 兜底：错误也以事件形式推送，前端可展示
                events.put({"type": "error", "message": str(e)})
            finally:
                events.put(None)  # 结束哨兵

        threading.Thread(target=worker, daemon=True).start()
        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history")
def history_endpoint(limit: int = Query(20, ge=1, le=100)):
    """历史对话列表（最新在前；含思考事件，供 AI 助手界面回放）。"""
    try:
        conn = connect()
        try:
            rows = repository.list_decision_logs(conn, limit=limit)
            return ok({"logs": [_log_payload(r) for r in rows]})
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))


@router.get("/history/{log_id}")
def history_detail_endpoint(log_id: int):
    """单条历史对话（供大屏决策流点击跳转到对应回答）。"""
    try:
        conn = connect()
        try:
            row = repository.get_decision_log(conn, log_id)
            if row is None:
                return err(404, f"历史记录不存在：{log_id}")
            return ok(_log_payload(row))
        finally:
            conn.close()
    except Exception as e:
        return err(500, str(e))
