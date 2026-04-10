"""Backend abstraction: one interface for API-backed and local HF models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GenerationConfig:
    """Generation parameters expressible in both OpenAI-compatible and HF formats."""

    max_new_tokens: int = 2048
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
        message: Dict[str, object] = {"role": "assistant", "content": reply}
        message.update(metadata)
        self.messages.append(message)

    def send(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        reply, metadata = self.backend.generate_with_metadata(self.messages)
        self._append_assistant_message(reply, metadata)
        return reply

    def generate_opening(self) -> str:
        """Generate the first assistant turn from the current session state."""
        reply, metadata = self.backend.generate_with_metadata(self.messages)
        self._append_assistant_message(reply, metadata)
        return reply
