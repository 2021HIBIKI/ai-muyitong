"""阶段 3.1：LLM 客户端测试 —— 无 key 降级、mock 响应解析、异常降级。"""
from __future__ import annotations

from app.agent.llm_client import LLMClient


def test_enabled_false_without_key():
    client = LLMClient(api_key="")
    assert client.enabled is False


def test_enabled_true_with_key():
    client = LLMClient(api_key="sk-test")
    assert client.enabled is True


def test_complete_fallback_without_key():
    client = LLMClient(api_key="")
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp["fallback"] is True
    assert resp["reason"] == "no_api_key"
    assert resp["tool_calls"] == []


def test_complete_parses_tool_calls(monkeypatch):
    import openai

    class _Function:
        name = "sheep_count"
        arguments = '{"image_path": null}'

    class _ToolCall:
        id = "call_1"

        def __init__(self):
            self.function = _Function()

    class _Message:
        content = "让我数一下羊"
        tool_calls = [_ToolCall()]

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            return _Response()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    client = LLMClient(api_key="sk-test")
    resp = client.complete([{"role": "user", "content": "有几只羊？"}], tools=[])
    assert resp["fallback"] is False
    assert resp["content"] == "让我数一下羊"
    assert resp["tool_calls"][0]["name"] == "sheep_count"
    assert resp["tool_calls"][0]["arguments"] == {"image_path": None}


def test_complete_fallback_on_exception(monkeypatch):
    import openai

    class _FailingClient:
        def __init__(self, **kwargs):
            raise RuntimeError("connection timeout")

    monkeypatch.setattr(openai, "OpenAI", _FailingClient)

    client = LLMClient(api_key="sk-test")
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp["fallback"] is True
    assert "connection timeout" in resp["reason"]


# ---------------------------------------------------------------------------
# 瞬时错误重试（阶段 5）
# ---------------------------------------------------------------------------

class _TransientErr(Exception):
    status_code = 429


class _AuthErr(Exception):
    status_code = 401


class _APITimeoutError(Exception):
    """模拟 openai.APITimeoutError（按类名含 timeout 判定）。"""


def _fake_openai(monkeypatch, outcomes):
    """注入按序返回结果/抛错的假 OpenAI 客户端；outcome 为 Exception 或 "ok"。"""
    import openai

    calls = {"n": 0}

    class _Message:
        content = "ok"
        tool_calls = []

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            outcome = outcomes[min(calls["n"], len(outcomes) - 1)]
            calls["n"] += 1
            if isinstance(outcome, Exception):
                raise outcome
            return _Response()

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)
    return calls


def test_complete_retries_transient_then_succeeds(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)  # 测试不真正等待
    calls = _fake_openai(monkeypatch, [_TransientErr("429"), _TransientErr("429"), "ok"])

    client = LLMClient(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp["fallback"] is False
    assert resp["content"] == "ok"
    assert calls["n"] == 3  # 两次重试后成功


def test_complete_fallback_after_retries_exhausted(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = _fake_openai(monkeypatch, [_TransientErr("always 429")])

    client = LLMClient(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp["fallback"] is True
    assert "always 429" in resp["reason"]
    assert calls["n"] == 3  # 1 次原始 + 2 次重试


def test_complete_no_retry_on_permanent_error(monkeypatch):
    calls = _fake_openai(monkeypatch, [_AuthErr("invalid api key")])

    client = LLMClient(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp["fallback"] is True
    assert calls["n"] == 1  # 永久错误（鉴权失败）不重试


def test_complete_retries_timeout(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = _fake_openai(monkeypatch, [_APITimeoutError("timed out"), "ok"])

    client = LLMClient(api_key="sk-test", max_retries=2, retry_backoff=0.01)
    resp = client.complete([{"role": "user", "content": "hi"}])
    assert resp["fallback"] is False
    assert calls["n"] == 2  # 超时重试一次后成功
