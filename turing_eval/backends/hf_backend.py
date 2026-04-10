"""Hugging Face local backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ChatBackend, GenerationConfig


def _import_hf_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - exercised via error path in tests
        raise RuntimeError(
            "HF dependencies are not installed in the default image. "
            "Build and run with ./scripts/build-container-hf.sh and ./scripts/run-in-container-hf.sh."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _resolve_dtype(torch_module: Any, dtype: Optional[str]) -> Any:
    if dtype is None:
        return None
    if not hasattr(torch_module, dtype):
        raise ValueError(f"Unsupported HF dtype {dtype!r}")
    return getattr(torch_module, dtype)


def _default_cache_dir() -> str:
    return (
        os.environ.get("TRANSFORMERS_CACHE")
        or os.environ.get("HF_HOME")
        or f"/tmp/{os.environ.get('USER', 'user')}-llm-eval-hf-cache"
    )


def _summarize_devices(model: Any) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        return ", ".join(f"{name}={device}" for name, device in sorted(device_map.items()))
    device = getattr(model, "device", None)
    if device is None:
        return "unknown"
    return str(device)


class HFChatBackend(ChatBackend):
    def __init__(
        self,
        repo_id: str,
        display_name: str,
        generation: GenerationConfig,
        quantization: Optional[str] = None,
        dtype: Optional[str] = "bfloat16",
    ) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = _import_hf_dependencies()

        self.backend_kind = "hf"
        self.provider_name = None
        self.model_id = repo_id
        self.display_name = display_name
        self.generation = generation
        self.quantization = quantization
        self.dtype = dtype
        self.cache_dir = str(Path(_default_cache_dir()))

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
        }
        resolved_dtype = _resolve_dtype(torch, dtype)
        cuda_available = bool(getattr(torch.cuda, "is_available", lambda: False)())

        if quantization == "4bit":
            if not cuda_available:
                raise RuntimeError("4bit HF quantization requires CUDA and the HF container wrapper.")
            compute_dtype = resolved_dtype or getattr(torch, "float16")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["device_map"] = "auto"
        else:
            if resolved_dtype is not None:
                model_kwargs["torch_dtype"] = resolved_dtype
            if cuda_available:
                model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
        self.device_summary = _summarize_devices(self._model)

    def _prepare_inputs(self, messages: List[Dict[str, object]]) -> Dict[str, Any]:
        chat_messages = [{"role": message["role"], "content": message["content"]} for message in messages]
        prompt = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None and hasattr(encoded, "to"):
            encoded = encoded.to(model_device)
        return dict(encoded)

    def generate(self, messages: List[Dict[str, object]]) -> str:
        inputs = self._prepare_inputs(messages)
        input_ids = inputs["input_ids"]
        prompt_length = int(input_ids.shape[-1])
        generation_kwargs = self.generation.to_hf_kwargs()
        generation_kwargs.pop("return_full_text", None)
        output_ids = self._model.generate(**inputs, pad_token_id=self._tokenizer.pad_token_id, **generation_kwargs)
        generated_ids = output_ids[:, prompt_length:]
        text = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if text:
            return text
        full_text = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if full_text:
            return full_text
        raise RuntimeError(f"{self.model_id} returned an empty decoded response.")
