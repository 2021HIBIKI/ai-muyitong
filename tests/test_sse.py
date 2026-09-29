"""阶段 7.5：告警推送中枢与 SSE 端点测试。

注意：**绝不用 TestClient 请求真实无限流**（TestClient 会整体缓冲响应，永久挂死）；
端点测试把事件生成器工厂 monkeypatch 成有限生成器。
"""
from __future__ import annotations

import asyncio
import json
import threading

from fastapi.sse import ServerSentEvent

from app.agent import patrol
from app.routers import dashboard


# ---------------------------------------------------------------------------
# AlertHub
# ---------------------------------------------------------------------------

def test_alert_hub_publish_to_subscribers():
    """订阅/广播/退订；队列满的慢消费者被丢弃但不阻塞广播。"""
    hub = patrol.AlertHub(maxsize=2)
    assert hub.subscriber_count == 0

    async def main():
        q1, q2 = hub.subscribe(), hub.subscribe()
        assert hub.subscriber_count == 2
        assert hub.publish({"id": 1, "level": "warning"}) == 2
        assert q1.get_nowait()["id"] == 1 and q2.get_nowait()["id"] == 1

        hub.unsubscribe(q1)
        assert hub.subscriber_count == 1 and hub.publish({"id": 2}) == 1

        hub.publish({"id": 3})
        hub.publish({"id": 4})
        assert hub.publish({"id": 5}) == 0  # q2 已满 → 丢弃本条，广播不抛错

    asyncio.run(main())


def test_alert_hub_publish_threadsafe():
    """从工作线程推送：走 loop.call_soon_threadsafe，订阅者能收到。"""
    hub = patrol.AlertHub()
    received: list[dict] = []

    async def main():
        queue = hub.subscribe()
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=hub.publish_threadsafe, args=({"id": 9}, loop))
        thread.start()
        received.append(await asyncio.wait_for(queue.get(), timeout=2.0))
        thread.join()

    asyncio.run(main())
    assert received == [{"id": 9}]


# ---------------------------------------------------------------------------
# 事件生成器（真 hub，直接取事件）
# ---------------------------------------------------------------------------

def test_alert_event_generator_emits_and_unsubscribes():
    """生成器订阅 hub：广播的告警成为 event: alert 事件；关闭时退订。"""

    async def main():
        generator = dashboard.alert_event_stream()
        pending = asyncio.ensure_future(generator.__anext__())
        for _ in range(100):  # 等生成器完成订阅
            if patrol.alert_hub.subscriber_count:
                break
            await asyncio.sleep(0.01)
        assert patrol.alert_hub.subscriber_count == 1

        patrol.alert_hub.publish({"id": 7, "level": "warning", "title": "草场预警告警"})
        event = await asyncio.wait_for(pending, timeout=2.0)
        assert isinstance(event, ServerSentEvent)
        assert event.event == "alert"
        assert event.data["id"] == 7  # 生成器产出的 data 仍是对象，编码在路由层完成

        await generator.aclose()
        assert patrol.alert_hub.subscriber_count == 0  # finally 中已退订

    asyncio.run(main())


# ---------------------------------------------------------------------------
# 端点（有限生成器）
# ---------------------------------------------------------------------------

def test_alerts_stream_endpoint_wire_format(client, monkeypatch):
    """端点返回 text/event-stream，事件名 alert，data 为 JSON 载荷。"""

    async def finite_stream():
        yield ServerSentEvent(
            event="alert",
            data={"id": 1, "level": "warning", "title": "草场预警告警", "message": "覆盖度 45%"},
        )
        yield ServerSentEvent(
            event="alert",
            data={"id": 2, "level": "danger", "title": "草场危险告警", "message": "覆盖度 20%"},
        )

    monkeypatch.setattr(dashboard, "alert_event_stream", finite_stream)

    resp = client.get("/api/dashboard/alerts/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.text.count("event: alert") == 2
    assert '"id": 1' in resp.text and '"level": "danger"' in resp.text
    # 每行 data 都是合法 JSON
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            assert json.loads(line[len("data: "):])["level"] in {"warning", "danger"}


def test_alerts_stream_registered_and_referenced_by_frontend(client):
    """端点已注册；前端引用 EventSource 与 /api/dashboard/alerts（7.6 断言同步）。"""
    html = client.get("/").text
    assert "EventSource(" in html
    assert "/api/dashboard/alerts" in html
