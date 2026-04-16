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
    # ---- Tier 0: Frontier closed ----
    "claude-sonnet-4.6": ModelSpec(
        nickname="claude-sonnet-4.6",
        display_name="anthropic/claude-sonnet-4.6",
        tier="frontier",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="anthropic/claude-sonnet-4.6",
    ),
    "claude-opus-4.6": ModelSpec(
        nickname="claude-opus-4.6",
        display_name="anthropic/claude-opus-4.6",  # VERIFIED
        tier="frontier",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="anthropic/claude-opus-4.6",
    ),
    "gpt-5.4": ModelSpec(
        nickname="gpt-5.4",
        display_name="openai/gpt-5.4",
        tier="frontier",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="openai/gpt-5.4",
        alternate_api_routes={"openai": "gpt-5.4"},
    ),
    "gpt-5.4-pro": ModelSpec(
        nickname="gpt-5.4-pro",
        display_name="openai/gpt-5.4-pro",  # VERIFIED
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
    # ---- Tier 1: Large MoE ----
    "deepseek-v3.2": ModelSpec(
        nickname="deepseek-v3.2",
        display_name="deepseek/deepseek-v3.2",  # VERIFIED
        tier="tier1",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="deepseek/deepseek-v3.2",
    ),
    "mistral-large-2512": ModelSpec(
        nickname="mistral-large-2512",
        display_name="mistralai/mistral-large-2512",
        tier="tier1",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="mistralai/mistral-large-2512",
    ),
    # ---- Tier 2: Medium dense ----
    "qwen3-32b": ModelSpec(
        nickname="qwen3-32b",
        display_name="qwen/qwen3-32b",  # VERIFIED
        tier="tier2",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="qwen/qwen3-32b",
    ),
    "gemma-4-31b": ModelSpec(
        nickname="gemma-4-31b",
        display_name="google/gemma-4-31b-it",
        tier="tier2",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="google/gemma-4-31b-it",
    ),
    # ---- Tier 3: Small dense ----
    "ministral-8b-2512": ModelSpec(
        nickname="ministral-8b-2512",
        display_name="mistralai/ministral-8b-2512",
        tier="tier3",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="mistralai/ministral-8b-2512",
    ),
    "llama-3.3-8b": ModelSpec(
        nickname="llama-3.3-8b",
        display_name="meta-llama/llama-3.3-8b-instruct",
        tier="tier3",
        primary_backend="api",
        primary_provider="openrouter",
        primary_model_id="meta-llama/llama-3.3-8b-instruct",
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
