"""Backend abstraction: one interface for API-backed and local HF models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Dict, List, Optional


@dataclass
class GenerationConfig:
    """Generation parameters expressible in both OpenAI-compatible and HF formats."""

    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: bool = True

    def __post_init__(self) -> None:
        for name in ("max_new_tokens", "top_k"):
            value = getattr(self, name)
            minimum = 0 if name == "top_k" else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.temperature is not None and (not isfinite(self.temperature) or self.temperature < 0):
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_p is not None and (not isfinite(self.top_p) or not 0 < self.top_p <= 1):
            raise ValueError("top_p must be finite and in (0, 1]")

    def to_openai_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        if self.max_new_tokens is not None:
            payload["max_tokens"] = self.max_new_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        elif not self.do_sample:
            payload["temperature"] = 0.0
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        return payload

    def to_hf_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "do_sample": self.do_sample and self.temperature != 0,
            "return_full_text": True,
        }
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        if self.temperature is not None and kwargs["do_sample"]:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        return kwargs


class ChatBackend(ABC):
    """Contract every backend must satisfy."""

    display_name: str
    model_id: str

    @abstractmethod
    def generate(self, messages: List[Dict[str, object]]) -> str:
        """Given an OpenAI-style messages list, return the assistant's reply text."""

    def generate_with_metadata(
        self, messages: List[Dict[str, object]]
    ) -> tuple[str, Dict[str, object]]:
        """Return the reply plus optional provider metadata for transcript persistence."""
        return self.generate(messages), {}


class ChatSession:
    """Mutable chat history bound to a single backend instance."""

    def __init__(
        self,
        backend: ChatBackend,
        name: str,
        initial_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.name = name
        self.messages: List[Dict[str, object]] = []
        if initial_prompt is not None and system_prompt is not None:
            raise ValueError("Pass only one of initial_prompt or system_prompt.")
        prompt = initial_prompt if initial_prompt is not None else system_prompt
        if prompt:
            self.messages.append({"role": "user", "content": prompt})

    def _append_assistant_message(self, reply: str, metadata: Dict[str, object]) -> None:
        message: Dict[str, object] = {**metadata, "role": "assistant", "content": reply}
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def send(self, content: str) -> str:
        self.add_user_message(content)
        reply, metadata = self.backend.generate_with_metadata(self.messages)
        self._append_assistant_message(reply, metadata)
        return reply

    def generate_opening(self) -> str:
        """Generate the first assistant turn from the current session state."""
        reply, metadata = self.backend.generate_with_metadata(self.messages)
        self._append_assistant_message(reply, metadata)
        return reply
