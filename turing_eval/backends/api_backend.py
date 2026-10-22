"""Provider-agnostic OpenAI-compatible chat client (stdlib urllib only)."""
from __future__ import annotations

import json
import math
import os
import ssl
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional
from urllib import error, request

from .base import ChatBackend, GenerationConfig


@lru_cache(maxsize=1)
def _ssl_context_for_https() -> ssl.SSLContext:
    """Prefer certifi's CA bundle so HTTPS works on macOS/Framework Pythons without system certs."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    extra_headers: Dict[str, str] = field(default_factory=dict)


# HTTP status codes where a short backoff retry is reasonable (provider overload / rate limits).
_TRANSIENT_HTTP_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _http_retryable(exc: error.HTTPError) -> bool:
    return exc.code in _TRANSIENT_HTTP_STATUS


def _network_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionResetError, BrokenPipeError, error.URLError)):
        return True
    if isinstance(exc, OSError):
        # e.g. 8 DNS, 54 reset, 104 connection reset, 110 timeout, 111 refused, 113 unreachable
        return exc.errno in {8, 54, 104, 110, 111, 113, 32, 35}
    return False


PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra_headers={
            "HTTP-Referer": "https://github.com/EmaRimoldi/generalized-turing-test",
            "X-Title": "Generalized Turing Test",
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


def is_placeholder_api_key(api_key: str | None) -> bool:
    key = (api_key or "").strip().upper()
    return not key or any(marker in key for marker in ("PLACEHOLDER", "REPLACE_ME", "YOUR-API-KEY"))


class APIChatBackend(ChatBackend):
    """OpenAI-compatible chat client usable with any provider in PROVIDERS."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        display_name: str,
        generation: GenerationConfig,
        timeout_seconds: float | None = None,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}. Known: {sorted(PROVIDERS)}")
        self.provider_config = PROVIDERS[provider]
        self.backend_kind = "api"
        self.provider_name = provider
        self.model_id = model_id
        self.display_name = display_name
        self.generation = generation
        if timeout_seconds is not None:
            self.timeout_seconds = float(timeout_seconds)
        else:
            env_t = os.environ.get("TURING_EVAL_API_TIMEOUT_SECONDS", "").strip()
            self.timeout_seconds = float(env_t) if env_t else 480.0
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("API timeout must be finite and positive")

        api_key = os.environ.get(self.provider_config.api_key_env, "").strip()
        if is_placeholder_api_key(api_key):
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
        max_retries = int(os.environ.get("TURING_EVAL_API_HTTP_RETRIES", "6"))
        max_retries = max(1, min(max_retries, 20))
        last_error: BaseException | None = None
        for attempt in range(max_retries):
            try:
                with request.urlopen(
                    req, timeout=self.timeout_seconds, context=_ssl_context_for_https()
                ) as response:
                    raw = response.read().decode("utf-8")
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{self.provider_config.name} returned non-JSON for {self.model_id}: {raw[:500]}"
                    ) from exc
                if not isinstance(decoded, dict):
                    raise RuntimeError(f"{self.provider_config.name} returned a non-object JSON response for {self.model_id}")
                return decoded
            except error.HTTPError as exc:
                details = ""
                try:
                    details = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    details = ""
                if _http_retryable(exc) and attempt + 1 < max_retries:
                    last_error = exc
                else:
                    raise RuntimeError(
                        f"{self.provider_config.name} HTTP {exc.code} {exc.reason} for {self.model_id}: {details}"
                    ) from exc
            except error.URLError as exc:
                last_error = exc
            except (TimeoutError, ConnectionResetError, BrokenPipeError) as exc:
                last_error = exc
            except OSError as exc:
                if _network_retryable(exc):
                    last_error = exc
                else:
                    raise RuntimeError(
                        f"{self.provider_config.name} network error for {self.model_id}: {exc}"
                    ) from exc
            if attempt + 1 < max_retries:
                delay = min(45.0, 1.6**attempt)
                time.sleep(delay)
        assert last_error is not None
        raise RuntimeError(
            f"{self.provider_config.name} request failed after {max_retries} attempts for {self.model_id}: {last_error!r}"
        ) from last_error

    @staticmethod
    def _extract_text_from_message(message: Dict[str, object]) -> Optional[str]:
        """Extract assistant text from an OpenRouter-style message (string or content blocks)."""
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

        # Reasoning tokens are a separate channel, not the agent's reply. In
        # particular, an answer tag in reasoning must never become a verdict.
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
