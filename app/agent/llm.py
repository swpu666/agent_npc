"""模型客户端：OpenAI 兼容 HTTP 接口，带可配置超时、有限重试与 Mock 分支。

- 只对「超时 / 429 / 5xx」重试；4xx 说明请求本身有问题，重试只会更慢。
- 单次调用的耗时单独统计（latency_ms），供接口 timings.llm_ms 使用，与本地检索耗时分离。
- MOCK_LLM=1 时不发起任何网络请求，返回明显标记的固定回复；接口/页面/评测报告均标记 mock=true。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import httpx

from app.agent.intent import INTENTS
from app.agent.prompt import build_intent_classifier_prompt
from app.config import Settings

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """模型调用失败。retryable 决定前端是否展示「重新发送」。"""

    def __init__(self, message: str, retryable: bool = False, kind: str = "unknown", status: int | None = None):
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.kind = kind
        self.status = status


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int
    mock: bool = False
    attempts: int = 1


def extract_json(text: str) -> dict | None:
    """从模型输出里稳健地取出第一个 JSON 对象。"""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mock = settings.mock_llm
        self.model = settings.llm_model

    # ---------------------------------------------------------------- 基础调用
    def _post(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        if self.mock:
            return self._mock_reply(messages)

        if not self.settings.has_api_key:
            raise LLMError(
                "未配置模型密钥：请在 .env 中填写 LLM_API_KEY（可参考 .env.example）",
                retryable=False,
                kind="config",
            )

        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        last_error: LLMError | None = None
        attempts = self.settings.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=self.settings.llm_timeout_s) as client:
                    resp = client.post(url, json=payload, headers=headers)
                if resp.status_code in RETRYABLE_STATUS:
                    raise LLMError(
                        f"模型服务返回 {resp.status_code}",
                        retryable=True,
                        kind="http_5xx" if resp.status_code >= 500 else "http_retryable",
                        status=resp.status_code,
                    )
                if resp.status_code >= 400:
                    raise LLMError(
                        f"模型服务返回 {resp.status_code}：{resp.text[:200]}",
                        retryable=False,
                        kind="http_4xx",
                        status=resp.status_code,
                    )
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError("模型返回为空", retryable=True, kind="empty_choices")
                content = (choices[0].get("message") or {}).get("content") or ""
                if not content.strip():
                    raise LLMError("模型返回空内容", retryable=True, kind="empty_content")
                return content
            except httpx.TimeoutException as exc:
                last_error = LLMError(
                    f"模型调用超时（>{self.settings.llm_timeout_s:g}s）",
                    retryable=True,
                    kind="timeout",
                )
                last_error.__cause__ = exc
            except httpx.HTTPError as exc:
                last_error = LLMError(f"模型服务连接失败：{type(exc).__name__}", retryable=True, kind="connection")
                last_error.__cause__ = exc
            except LLMError as exc:
                last_error = exc
                if not exc.retryable:
                    raise

            if attempt < attempts - 1:
                time.sleep(0.5 * (attempt + 1))

        raise last_error or LLMError("模型调用失败", retryable=True)

    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> LLMResult:
        started = time.perf_counter()
        text = self._post(
            messages,
            temperature=self.settings.llm_temperature if temperature is None else temperature,
            max_tokens=self.settings.llm_max_tokens if max_tokens is None else max_tokens,
        )
        return LLMResult(
            text=text.strip(),
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            mock=self.mock,
        )

    # ---------------------------------------------------------------- 意图兜底
    def classify_intent(self, text: str) -> str | None:
        """规则不确定时的受限分类；失败返回 None，由上层走 knowledge_qa 兜底。"""
        if self.mock:
            return None
        try:
            result = self.chat(build_intent_classifier_prompt(text), temperature=0.0, max_tokens=40)
        except LLMError:
            return None
        payload = extract_json(result.text) or {}
        intent = str(payload.get("intent", "")).strip()
        return intent if intent in INTENTS else None

    # ---------------------------------------------------------------- 评审
    def judge_json(self, system_prompt: str, user_prompt: str) -> tuple[dict | None, LLMResult]:
        result = self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        return extract_json(result.text), result

    # ---------------------------------------------------------------- Mock
    def _mock_reply(self, messages: list[dict]) -> str:
        material = ""
        for m in messages:
            if m.get("role") == "system" and "【知识材料】" in str(m.get("content", "")):
                material = str(m["content"])
                break
        ids = re.findall(r'id="([^"]+)"', material)
        tag = "[MOCK 模式·未调用真实模型]"
        if ids:
            return f"{tag} 已检索到知识条目：{', '.join(ids)}。此处为固定回复，仅用于验证链路连通，不代表真实回答质量。"
        return f"{tag} 这是固定回复，未调用真实模型，不代表真实回答质量。"


_client: LLMClient | None = None


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    global _client
    if settings is not None:
        return LLMClient(settings)
    if _client is None:
        from app.config import get_settings

        _client = LLMClient(get_settings())
    return _client
