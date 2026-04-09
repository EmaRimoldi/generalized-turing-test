"""Provider-agnostic OpenAI-compatible chat client (stdlib urllib only)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib import error, request

from .base import ChatBackend, GenerationConfig


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    extra_headers: Dict[str, str] = field(default_factory=dict)


PROVIDERS: Dict[str, ProviderConfig] = {
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_headers={
            "HTTP-Referer": "https://github.com/suahong/llm-eval",
            "X-Title": "turing-eval",
        },
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "moonshot": ProviderConfig(
        name="moonshot",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
    ),
    "together": ProviderConfig(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
    ),
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "mistral": ProviderConfig(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
    ),
}


class APIChatBackend(ChatBackend):
    """OpenAI-compatible chat client usable with any provider in PROVIDERS."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        display_name: str,
        generation: GenerationConfig,
        timeout_seconds: float = 180.0,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}. Known: {sorted(PROVIDERS)}")
        self.provider_config = PROVIDERS[provider]
        self.model_id = model_id
        self.display_name = display_name
        self.generation = generation
        self.timeout_seconds = timeout_seconds

        api_key = os.environ.get(self.provider_config.api_key_env)
        if not api_key or api_key.endswith("PLACEHOLDER"):
            raise RuntimeError(
                f"Environment variable {self.provider_config.api_key_env} is unset or a placeholder. "
                f"Set a real key before calling APIChatBackend for provider {provider!r}."
            )
        self._api_key = api_key

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.provider_config.extra_headers)
        return headers

    @staticmethod
    def _sanitize_text(text: str) -> str:
        return "".join("\ufffd" if 0xD800 <= ord(char) <= 0xDFFF else char for char in text)

    @classmethod
    def _sanitize_messages(cls, messages: List[Dict[str, object]]) -> List[Dict[str, object]]:
        sanitized: List[Dict[str, object]] = []
        for message in messages:
            sanitized_message: Dict[str, object] = {}
            for key in ("role", "content", "name"):
                if key not in message:
                    continue
                value = message[key]
                sanitized_message[key] = cls._sanitize_text(value) if isinstance(value, str) else value
            sanitized.append(sanitized_message)
        return sanitized

    def _post_chat_completion(self, messages: List[Dict[str, object]]) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": self.model_id,
            "messages": self._sanitize_messages(messages),
        }
        payload.update(self.generation.to_openai_payload())
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.provider_config.base_url.rstrip('/')}/chat/completions"
        req = request.Request(url=url, data=body, headers=self._headers(), method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.provider_config.name} HTTP {exc.code} {exc.reason} for {self.model_id}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"{self.provider_config.name} network error for {self.model_id}: {exc.reason}"
            ) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{self.provider_config.name} returned non-JSON for {self.model_id}: {raw[:500]}"
            ) from exc

    @staticmethod
    def _extract_text_from_message(message: Dict[str, object]) -> Optional[str]:
        """Port of Susan's extraction logic from run_openrouter_single_stage_experiment.py."""
        content = message.get("content")
        if isinstance(content, str):
            stripped = content.strip()
            if stripped:
                return stripped
        if isinstance(content, list):
            text_chunks = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in {"text", "output_text"}:
                    text = item.get("text", "")
                    if isinstance(text, str):
                        text_chunks.append(text)
            merged = "".join(text_chunks).strip()
            if merged:
                return merged

        reasoning = message.get("reasoning")
        if isinstance(reasoning, str):
            stripped = reasoning.strip()
            if stripped:
                return stripped

        reasoning_details = message.get("reasoning_details")
        if isinstance(reasoning_details, list):
            text_chunks = []
            for item in reasoning_details:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", "")
                if isinstance(text, str):
                    text_chunks.append(text)
            merged = "\n".join(chunk.strip() for chunk in text_chunks if chunk.strip()).strip()
            if merged:
                return merged

        return None

    def generate_with_metadata(
        self, messages: List[Dict[str, object]]
    ) -> tuple[str, Dict[str, object]]:
        response = self._post_chat_completion(messages)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                f"{self.provider_config.name} returned no choices for {self.model_id}: {response}"
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError(f"{self.provider_config.name} returned invalid choice: {choice}")
        message = choice.get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError(f"{self.provider_config.name} returned invalid message: {message}")
        extracted = self._extract_text_from_message(message)
        if extracted is None:
            raise RuntimeError(
                f"{self.provider_config.name} returned unparseable message for {self.model_id}: {message}"
            )
        metadata: Dict[str, object] = {}
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason
        if finish_reason == "length":
            metadata["truncated"] = True
        return extracted, metadata

    def generate(self, messages: List[Dict[str, object]]) -> str:
        reply, _metadata = self.generate_with_metadata(messages)
        return reply
