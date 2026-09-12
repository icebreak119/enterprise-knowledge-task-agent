"""OpenAI 兼容协议的 Provider 实现。

同一个类可对接 OpenAI 官方、GLM（智谱）、DeepSeek、通义以及本地 vLLM，
只需在配置里换 base_url 与 model。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings, get_settings
from app.llm.base import ChatMessage, LLMResponse, ToolCall
from app.llm.errors import LLMConnectionError, LLMError, LLMParseError, LLMRateLimitError

logger = logging.getLogger(__name__)

_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _should_retry(exc: BaseException) -> bool:
    """只对可恢复的错误重试：连接/超时、限流、5xx。"""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return False


def _to_llm_error(exc: Exception) -> LLMError:
    if isinstance(exc, LLMError):
        return exc
    status = getattr(exc, "status_code", None)
    if status == 429:
        return LLMRateLimitError(str(exc))
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return LLMConnectionError(str(exc))
    if status is not None and status >= 500:
        return LLMConnectionError(str(exc))
    return LLMError(str(exc))


def _usage(raw: Any) -> dict[str, int] | None:
    usage = getattr(raw, "usage", None)
    if not usage:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def _tool_calls_from(message: Any) -> list[ToolCall]:
    calls = getattr(message, "tool_calls", None) or []
    return [
        ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
        for call in calls
    ]


def _strip_code_fence(text: str) -> str:
    matched = _CODE_FENCE.match(text)
    return matched.group(1) if matched else text


def _parse_extra_body(raw: str) -> dict[str, Any]:
    """解析厂商私有参数。配置写错要立刻炸，而不是静默发出错误请求。"""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM_EXTRA_BODY 不是合法 JSON：{raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM_EXTRA_BODY 必须是 JSON 对象，当前为 {type(parsed).__name__}：{raw!r}")
    return parsed


class OpenAICompatProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = AsyncOpenAI(
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url or None,
            timeout=self._settings.llm_timeout,
            max_retries=0,  # 重试统一交给 tenacity
        )
        # 厂商私有参数（关闭思维链等），各家字段名不统一，交给配置决定
        self._extra_body = _parse_extra_body(self._settings.llm_extra_body)
        # None = 还没试过；True/False = 该厂商是否支持 response_format=json_schema
        self._json_schema_supported: bool | None = None

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [message.to_payload() for message in messages],
            "temperature": (
                self._settings.llm_temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self._settings.llm_max_tokens if max_tokens is None else max_tokens
            ),
        }
        if tools:
            kwargs["tools"] = tools
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        try:
            if response_model is not None:
                return await self._achat_structured(kwargs, response_model)
            raw = await self._request("create", **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise _to_llm_error(exc) from exc

        message = raw.choices[0].message
        return LLMResponse(
            content=message.content or "",
            tool_calls=_tool_calls_from(message),
            usage=_usage(raw),
            model=raw.model,
            reasoning=getattr(message, "reasoning_content", None),
        )

    async def _achat_structured(
        self, kwargs: dict[str, Any], response_model: type[BaseModel]
    ) -> LLMResponse:
        # 优先用 response_format=json_schema（OpenAI 官方及部分厂商支持）
        if self._json_schema_supported is not False:
            try:
                raw = await self._request("parse", response_format=response_model, **kwargs)
                self._json_schema_supported = True
                message = raw.choices[0].message
                return LLMResponse(
                    content=message.content or "",
                    parsed=message.parsed,
                    usage=_usage(raw),
                    model=raw.model,
                    reasoning=getattr(message, "reasoning_content", None),
                )
            except Exception as exc:  # noqa: BLE001  厂商不支持时静默降级
                # 限流 / 连接 / 5xx 属于可恢复错误，不是"厂商不支持"，
                # 绝不能据此永久关闭 json_schema，否则一次 429 会拖垮整个进程。
                if _should_retry(exc):
                    raise
                self._json_schema_supported = False
                logger.warning("json_schema 结构化输出不可用，降级为 json_object：%s", exc)

        # 降级：json_object 模式 + 用 JSON Schema 提示 + 手动校验
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        hint = ChatMessage(
            role="user",
            content=(
                "只输出 JSON，不要解释、不要代码块，"
                f"严格符合以下 JSON Schema：\n{schema}"
            ),
        )
        degraded = {**kwargs, "messages": [*kwargs["messages"], hint.to_payload()]}
        raw = await self._request("create", response_format={"type": "json_object"}, **degraded)
        content = raw.choices[0].message.content or ""
        try:
            parsed = response_model.model_validate_json(_strip_code_fence(content))
        except ValidationError as exc:
            raise LLMParseError(
                f"模型输出无法解析为 {response_model.__name__}：{content[:200]}"
            ) from exc
        return LLMResponse(
            content=content,
            parsed=parsed,
            usage=_usage(raw),
            model=raw.model,
            reasoning=getattr(raw.choices[0].message, "reasoning_content", None),
        )

    async def _request(self, method: str, **kwargs: Any) -> Any:
        """带重试地发起一次请求。method 为 create 或 parse。"""
        call = (
            self._client.chat.completions.create
            if method == "create"
            else self._client.chat.completions.parse
        )
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self._settings.llm_max_retries),
            # 免费额度的 429 常常持续数秒，退避窗口太窄等于白重试
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception(_should_retry),
        ):
            with attempt:
                return await call(**kwargs)
