"""Unit tests for configuration and Gemini provider."""

import pytest
from agentops.config import Settings, DEFAULT_MODEL_PRICING
from agentops.providers.gemini_provider import GeminiProvider, GenerationResult


def test_settings_defaults():
    """Verify default configuration settings."""
    settings = Settings()
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.max_agent_iterations == 3
    assert settings.allow_mock_fallback is True
    assert "gemini-2.5-flash" in DEFAULT_MODEL_PRICING


def test_pricing_and_cost_calculation():
    """Verify cost formula: (input*price + output*price)/1,000,000."""
    settings = Settings()
    # gemini-2.5-flash: input=0.075/1M, output=0.30/1M
    cost = settings.calculate_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        model_name="gemini-2.5-flash"
    )
    assert cost is not None
    assert round(cost, 4) == round(0.075 + 0.30, 4)

    # Unknown model should return None
    assert settings.calculate_cost(100, 100, model_name="unknown-model-xyz") is None


@pytest.mark.asyncio
async def test_mock_provider_generation():
    """Verify deterministic mock provider generates valid results with tokens and cost."""
    settings = Settings(allow_mock_fallback=True)
    provider = GeminiProvider(settings=settings)

    # Test researcher role
    res = await provider.generate(prompt="Analyze repository structure", agent_role="researcher")
    assert isinstance(res, GenerationResult)
    assert res.is_mock is True
    assert res.input_tokens > 0
    assert res.output_tokens > 0
    assert res.total_tokens == res.input_tokens + res.output_tokens
    assert res.parsed_json is not None
    assert "identified_facts" in res.parsed_json

    # Test judge pass
    judge_pass = await provider.generate(prompt="Check report", agent_role="judge")
    assert judge_pass.parsed_json["status"] == "pass"

    # Test judge fail trigger
    judge_fail = await provider.generate(prompt="Reject this fabricated draft", agent_role="judge")
    assert judge_fail.parsed_json["status"] == "fail"
