"""HuggingFace local backend. Stub until Step 12.

The real implementation lives in a separate container image with torch/transformers/bitsandbytes.
Any attempt to instantiate this class in the default image will fail with
NotImplementedError so Steps 4-11 can run without a GPU stack.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import ChatBackend, GenerationConfig


class HFChatBackend(ChatBackend):
    def __init__(
        self,
        repo_id: str,
        display_name: str,
        generation: GenerationConfig,
        quantization: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
    ) -> None:
        raise NotImplementedError(
            "HFChatBackend is stubbed in the default image. "
            "Implement it in Step 12 and run it via the HF image variant."
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:  # pragma: no cover
        raise NotImplementedError
