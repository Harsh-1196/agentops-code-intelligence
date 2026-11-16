"""Live integration test for Google AI Studio Gemini API.

Only runs when GEMINI_API_KEY environment variable is configured with a valid key.
Skips explicitly if no key is present. Never fakes pass using mocks.
"""

import os
import pytest
from agentops.config import Settings
from agentops.providers.gemini_provider import GeminiProvider, GenerationResult


@pytest.mark.live_gemini
@pytest.mark.asyncio
async def test_live_gemini_api_connection():
    """Verify live Gemini Developer API generation and real token usage metadata."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        pytest.skip("Skipping live Gemini test: GEMINI_API_KEY is not set in environment.")

    settings = Settings(gemini_api_key=api_key, allow_mock_fallback=False)
    provider = GeminiProvider(settings=settings)

    assert provider.is_live is True, "Provider failed to initialize live Google AI Studio client"

    prompt = "Reply with exactly one word: 'CONFIRMED'."
    result = await provider.generate(
        prompt=prompt,
        model=settings.gemini_model,
        agent_role="general",
    )

    assert isinstance(result, GenerationResult)
    assert result.is_mock is False, "Live test must not return mock result"
    assert "CONFIRMED" in result.text.upper()
    assert result.input_tokens > 0, "Live Gemini API must return actual prompt token count"
    assert result.output_tokens > 0, "Live Gemini API must return actual candidate token count"
    assert result.total_tokens >= result.input_tokens + result.output_tokens
    assert result.cost_usd is not None, "Cost must be calculated for live model call"
