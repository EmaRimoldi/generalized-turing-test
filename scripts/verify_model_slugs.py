#!/usr/bin/env python3
"""Validate every OpenRouter slug in MODEL_REGISTRY against the live /models endpoint.

Exit codes:
  0 - all good
  1 - missing OPENROUTER_API_KEY
  2 - one or more slugs are invalid
  3 - network error
"""
from __future__ import annotations

import json
import os
import sys
from urllib import error, request

from turing_eval.models import MODEL_REGISTRY


def fetch_openrouter_catalog() -> set[str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or api_key.endswith("PLACEHOLDER"):
        print("ERROR: OPENROUTER_API_KEY is unset or placeholder.", file=sys.stderr)
        sys.exit(1)
    req = request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (error.HTTPError, error.URLError) as exc:
        print(f"ERROR: OpenRouter /models fetch failed: {exc}", file=sys.stderr)
        sys.exit(3)
    return {str(model.get("id")) for model in data.get("data", [])}


def main() -> int:
    catalog = fetch_openrouter_catalog()
    print(f"Fetched {len(catalog)} models from OpenRouter /models")
    bad: list[tuple[str, str, str]] = []
    for nickname, spec in MODEL_REGISTRY.items():
        if spec.primary_backend == "api" and spec.primary_provider == "openrouter":
            if spec.primary_model_id not in catalog:
                bad.append((nickname, "primary", spec.primary_model_id or ""))
        if spec.fallback_backend == "api" and spec.fallback_provider == "openrouter":
            if spec.fallback_model_id not in catalog:
                bad.append((nickname, "fallback", spec.fallback_model_id or ""))

    if not bad:
        print(f"OK: all {len(MODEL_REGISTRY)} registry entries are valid on OpenRouter.")
        return 0

    print(f"INVALID: {len(bad)} slug(s) do not exist on OpenRouter:")
    for nickname, which, model_id in bad:
        print(f"  - {nickname} ({which}): {model_id}")
        prefix = model_id.split("/")[-1].split("-")[0].lower() if model_id else ""
        if prefix:
            suggestions = sorted(candidate for candidate in catalog if prefix in candidate.lower())[:5]
            if suggestions:
                print(f"      similar: {suggestions}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
