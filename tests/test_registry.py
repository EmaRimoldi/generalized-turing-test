from turing_eval.models import MODEL_REGISTRY, ModelSpec


def test_registry_not_empty():
    assert len(MODEL_REGISTRY) >= 10


def test_every_spec_has_required_fields():
    for nickname, spec in MODEL_REGISTRY.items():
        assert isinstance(spec, ModelSpec)
        assert spec.nickname == nickname
        assert spec.display_name
        assert spec.tier in {"frontier", "tier1", "tier2", "tier3"}
        assert spec.primary_backend in {"api", "hf"}


def test_api_primary_has_provider_and_model_id():
    for nickname, spec in MODEL_REGISTRY.items():
        if spec.primary_backend == "api":
            assert spec.primary_provider, f"{nickname} missing primary_provider"
            assert spec.primary_model_id, f"{nickname} missing primary_model_id"


def test_hf_primary_has_repo():
    for nickname, spec in MODEL_REGISTRY.items():
        if spec.primary_backend == "hf":
            assert spec.fallback_hf_repo, f"{nickname} missing fallback_hf_repo"


def test_display_name_matches_primary_model_id_for_api():
    for nickname, spec in MODEL_REGISTRY.items():
        if spec.primary_backend == "api" and spec.primary_provider == "openrouter":
            assert spec.display_name == spec.primary_model_id, (
                f"{nickname}: display_name {spec.display_name!r} != model_id {spec.primary_model_id!r}"
            )
