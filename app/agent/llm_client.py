"""GLM LLM 客户端（OpenAI 兼容）+ 无 key / 瞬时错误重试 / 降级。

complete() 永不抛错：瞬时错误（429/5xx/超时/连接失败）按指数退避重试，
重试耗尽或永久错误（如鉴权失败）则返回 fallback 哨兵，由编排器走规则降级。
"""
from __future__ import annotations

import json
import time
from typing import Any

from app.config import settings

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    """瞬时错误判定：HTTP 429/5xx、超时、连接类异常（按状态码与异常类名判断）。"""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status in _TRANSIENT_STATUS or 500 <= status < 600):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


class LLMClient:
    """OpenAI 兼容 LLM 客户端。complete() 永不抛错，失败时返回 fallback 哨兵。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.glm_api_key
        self.base_url = base_url if base_url is not None else settings.glm_base_url
        self.model = model if model is not None else settings.glm_model
        self.timeout = timeout
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        self.retry_backoff = (
            retry_backoff if retry_backoff is not None else settings.llm_retry_backoff
        )

    @property
    def enabled(self) -> bool:
        """是否配置了 API key（决定走真实 LLM 还是规则降级）。"""
        return bool(self.api_key)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """调用 LLM 并返回标准化结果，永不抛错。

        返回：
            {"fallback": bool, "content": str|None, "tool_calls": list[dict], "reason": str|None}
            tool_calls 元素形如 {"name": str, "arguments": dict, "id": str|None}。
        """
        if not self.enabled:
            return {"fallback": True, "content": None, "tool_calls": [], "reason": "no_api_key"}

        attempt = 0
        while True:
            try:
                return self._call_once(messages, tools)
            except Exception as e:
                if attempt < self.max_retries and _is_transient(e):
                    attempt += 1
                    time.sleep(self.retry_backoff * (2 ** (attempt - 1)))  # 指数退避
                    continue
                # 重试耗尽 / 永久错误（超时 / 网络 / 鉴权 / 解析失败等）一律降级
                return {"fallback": True, "content": None, "tool_calls": [], "reason": str(e)}

    def _call_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """单次调用（异常向上抛，由 complete 决定重试或降级）。"""
        import openai

        client = openai.OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                {"name": tc.function.name, "arguments": arguments, "id": getattr(tc, "id", None)}
            )

        return {
            "fallback": False,
            "content": msg.content,
            "tool_calls": tool_calls,
            "reason": None,
        }
