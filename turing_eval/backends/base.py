"""Backend abstraction: one interface for API-backed and local HF models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GenerationConfig:
    """Generation parameters expressible in both OpenAI-compatible and HF formats."""

    max_new_tokens: int = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: bool = True

    def to_openai_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"max_tokens": self.max_new_tokens}
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
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "return_full_text": True,
        }
        if self.temperature is not None:
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
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Given an OpenAI-style messages list, return the assistant's reply text."""


class ChatSession:
    """Mutable chat history bound to a single backend instance."""

    def __init__(self, backend: ChatBackend, name: str, system_prompt: Optional[str] = None) -> None:
        self.backend = backend
        self.name = name
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def send(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        reply = self.backend.generate(self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def generate_opening(self) -> str:
        """Generate the first assistant turn from the current session state."""
        reply = self.backend.generate(self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        return reply
