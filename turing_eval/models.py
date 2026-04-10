"""Single source of truth for model nicknames, backends, and provider routing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from .backends.api_backend import APIChatBackend
from .backends.base import ChatBackend, GenerationConfig

BackendKind = Literal["api", "hf"]


@dataclass
class ModelSpec:
    nickname: str
    display_name: str
    tier: str

    primary_backend: BackendKind
    primary_provider: Optional[str] = None
    primary_model_id: Optional[str] = None

    fallback_backend: Optional[BackendKind] = None
    fallback_provider: Optional[str] = None
    fallback_model_id: Optional[str] = None
    fallback_hf_repo: Optional[str] = None
    fallback_hf_quantization: Optional[str] = None
    fallback_hf_dtype: Optional[str] = "bfloat16"
    alternate_api_routes: Dict[str, str] = field(default_factory=dict)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ---- Frontier closed ----
    "claude-opus-4.6": ModelSpec(
        nickname="claude-opus-4.6",
        display_name="anthropic/claude-opus-4.6",  # VERIFIED
        tier="frontier",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="anthropic/claude-opus-4.6",
    ),
    "claude-sonnet-4.6": ModelSpec(
        nickname="claude-sonnet-4.6",
        display_name="anthropic/claude-sonnet-4.6",  # TODO-VERIFY
        tier="frontier",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="anthropic/claude-sonnet-4.6",
    ),
    "gpt-5.4": ModelSpec(
        nickname="gpt-5.4",
        display_name="openai/gpt-5.4",  # TODO-VERIFY
        tier="frontier",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="openai/gpt-5.4",
        alternate_api_routes={"openai": "gpt-5.4"},
    ),
    "gpt-5.4-pro": ModelSpec(
        nickname="gpt-5.4-pro",
        display_name="openai/gpt-5.4-pro",  # TODO-VERIFY
        tier="frontier",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="openai/gpt-5.4-pro",
        alternate_api_routes={"openai": "gpt-5.4-pro"},
    ),
    "gemini-3-pro": ModelSpec(
        nickname="gemini-3-pro",
        display_name="google/gemini-3.1-pro-preview",  # VERIFIED
        tier="frontier",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="google/gemini-3.1-pro-preview",
    ),
    # ---- Tier 1: Frontier open ----
    "deepseek-v3.2": ModelSpec(
        nickname="deepseek-v3.2",
        display_name="deepseek/deepseek-v3.2",  # VERIFIED
        tier="tier1",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="deepseek/deepseek-v3.2",
        fallback_backend="api",
        fallback_provider="deepseek",
        fallback_model_id="deepseek-chat",
    ),
    "glm-5-reasoning": ModelSpec(
        nickname="glm-5-reasoning",
        display_name="z-ai/glm-5",  # VERIFIED
        tier="tier1",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="z-ai/glm-5",
    ),
    "kimi-k2.5": ModelSpec(
        nickname="kimi-k2.5",
        display_name="moonshotai/kimi-k2.5",  # TODO-VERIFY
        tier="tier1",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="moonshotai/kimi-k2.5",
    ),
    "llama-3.3-70b": ModelSpec(
        nickname="llama-3.3-70b",
        display_name="meta-llama/llama-3.3-70b-instruct",  # VERIFIED
        tier="tier1",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="meta-llama/llama-3.3-70b-instruct",
        fallback_backend="hf",
        fallback_hf_repo="meta-llama/Llama-3.3-70B-Instruct",
        fallback_hf_quantization="4bit",
    ),
    # ---- Tier 2: Mid ----
    "qwen3-30b-a3b": ModelSpec(
        nickname="qwen3-30b-a3b",
        display_name="qwen/qwen3-30b-a3b",  # VERIFIED
        tier="tier2",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="qwen/qwen3-30b-a3b",
        fallback_backend="hf",
        fallback_hf_repo="Qwen/Qwen3-30B-A3B-Instruct",
    ),
    "qwen2.5-32b": ModelSpec(
        nickname="qwen2.5-32b",
        display_name="qwen/qwen3-32b",  # VERIFIED
        tier="tier2",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="qwen/qwen3-32b",
        fallback_backend="hf",
        fallback_hf_repo="Qwen/Qwen2.5-32B-Instruct",
    ),
    "exaone-4-32b": ModelSpec(
        nickname="exaone-4-32b",
        display_name="LGAI-EXAONE/EXAONE-4.0-32B-Instruct",
        tier="tier2",
        primary_backend="hf",
        fallback_hf_repo="LGAI-EXAONE/EXAONE-4.0-32B-Instruct",
        fallback_hf_dtype="bfloat16",
    ),
    "gemma-3-27b": ModelSpec(
        nickname="gemma-3-27b",
        display_name="google/gemma-3-27b-it",  # VERIFIED
        tier="tier2",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="google/gemma-3-27b-it",
        fallback_backend="hf",
        fallback_hf_repo="google/gemma-3-27b-it",
    ),
    # ---- Tier 3: Weak baseline ----
    "gemma-3-12b": ModelSpec(
        nickname="gemma-3-12b",
        display_name="google/gemma-3-12b-it",  # VERIFIED
        tier="tier3",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="google/gemma-3-12b-it",
        fallback_backend="hf",
        fallback_hf_repo="google/gemma-3-12b-it",
    ),
    "qwen2.5-7b": ModelSpec(
        nickname="qwen2.5-7b",
        display_name="qwen/qwen3-8b",  # VERIFIED
        tier="tier3",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="qwen/qwen3-8b",
        fallback_backend="hf",
        fallback_hf_repo="Qwen/Qwen2.5-7B-Instruct",
    ),
    "llama-3.1-8b": ModelSpec(
        nickname="llama-3.1-8b",
        display_name="meta-llama/llama-3.1-8b-instruct",  # TODO-VERIFY
        tier="tier3",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="meta-llama/llama-3.1-8b-instruct",
        fallback_backend="hf",
        fallback_hf_repo="meta-llama/Llama-3.1-8B-Instruct",
    ),
    "phi-4": ModelSpec(
        nickname="phi-4",
        display_name="microsoft/phi-4",  # TODO-VERIFY
        tier="tier3",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="microsoft/phi-4",
        fallback_backend="hf",
        fallback_hf_repo="microsoft/phi-4",
    ),
    "mistral-7b-v0.3": ModelSpec(
        nickname="mistral-7b-v0.3",
        display_name="mistralai/mistral-7b-instruct-v0.1",  # VERIFIED
        tier="tier3",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="mistralai/mistral-7b-instruct-v0.1",
        fallback_backend="hf",
        fallback_hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
    ),
}


def load_backend(
    nickname: str,
    generation: GenerationConfig,
    force_backend: Optional[BackendKind] = None,
    force_api_provider: Optional[str] = None,
) -> ChatBackend:
    """Factory: given a nickname, return a ready-to-use ChatBackend instance."""
    if nickname not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model nickname {nickname!r}. Known: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[nickname]
    chosen: BackendKind = force_backend or ("api" if force_api_provider else spec.primary_backend)

    if chosen == "hf" and force_api_provider is not None:
        raise ValueError(f"Cannot request API provider {force_api_provider!r} with HF backend for {nickname}")

    if chosen == "api":
        if force_api_provider is None and spec.primary_backend == "api":
            assert spec.primary_provider and spec.primary_model_id
            return APIChatBackend(
                provider=spec.primary_provider,
                model_id=spec.primary_model_id,
                display_name=spec.display_name,
                generation=generation,
            )
        if force_api_provider is None and spec.fallback_backend == "api":
            assert spec.fallback_provider and spec.fallback_model_id
            return APIChatBackend(
                provider=spec.fallback_provider,
                model_id=spec.fallback_model_id,
                display_name=spec.display_name,
                generation=generation,
            )
        if force_api_provider is not None:
            if spec.primary_backend == "api" and spec.primary_provider == force_api_provider:
                assert spec.primary_model_id
                return APIChatBackend(
                    provider=force_api_provider,
                    model_id=spec.primary_model_id,
                    display_name=spec.display_name,
                    generation=generation,
                )
            if spec.fallback_backend == "api" and spec.fallback_provider == force_api_provider:
                assert spec.fallback_model_id
                return APIChatBackend(
                    provider=force_api_provider,
                    model_id=spec.fallback_model_id,
                    display_name=spec.display_name,
                    generation=generation,
                )
            if force_api_provider in spec.alternate_api_routes:
                return APIChatBackend(
                    provider=force_api_provider,
                    model_id=spec.alternate_api_routes[force_api_provider],
                    display_name=spec.display_name,
                    generation=generation,
                )
            raise ValueError(f"No API provider route {force_api_provider!r} configured for {nickname}")
        raise ValueError(f"No API backend configured for {nickname}")

    if chosen == "hf":
        from .backends.hf_backend import HFChatBackend

        if not spec.fallback_hf_repo:
            raise ValueError(f"No HF fallback configured for {nickname}")
        return HFChatBackend(
            repo_id=spec.fallback_hf_repo,
            display_name=spec.display_name,
            generation=generation,
            quantization=spec.fallback_hf_quantization,
            dtype=spec.fallback_hf_dtype,
        )

    raise ValueError(f"Unknown backend kind {chosen!r}")
