"""Configuration and settings for AgentOps platform."""

from functools import lru_cache
from typing import Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPricing(BaseSettings):
    """Cost configuration per 1 million tokens."""
    input_price_per_1m: float
    output_price_per_1m: float
    currency: str = "USD"
    pricing_source: str = "Google AI Studio Official Gemini Developer API Pricing"
    verification_date: str = "2025-06-01"


# Default verified model pricing table (per 1M tokens)
DEFAULT_MODEL_PRICING: Dict[str, ModelPricing] = {
    "gemini-2.5-flash": ModelPricing(
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
        currency="USD"
    ),
    "gemini-3.5-flash": ModelPricing(
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
        currency="USD"
    ),
    "gemini-3-flash-preview": ModelPricing(
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
        currency="USD"
    ),
    "gemini-2.0-flash": ModelPricing(
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
        currency="USD"
    ),
    "gemini-1.5-flash": ModelPricing(
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
        currency="USD"
    ),
    "gemini-1.5-pro": ModelPricing(
        input_price_per_1m=1.25,
        output_price_per_1m=5.00,
        currency="USD"
    ),
    "mock-evaluator": ModelPricing(
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        currency="USD",
        pricing_source="Deterministic Test Fixture"
    ),
}


class Settings(BaseSettings):
    """Global application settings loaded from environment or .env file."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys & LLM Config (Google AI Studio - Zero Vertex AI required)
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.2, alias="GEMINI_TEMPERATURE")

    # Orchestrator parameters
    max_agent_iterations: int = Field(default=3, alias="MAX_AGENT_ITERATIONS")
    agent_timeout_seconds: int = Field(default=120, alias="AGENT_TIMEOUT_SECONDS")
    eval_strictness: str = Field(default="strict", alias="EVAL_STRICTNESS")

    # Server configuration
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT")
    frontend_port: int = Field(default=3000, alias="FRONTEND_PORT")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="CORS_ORIGINS"
    )

    # OpenTelemetry & Jaeger
    otel_service_name: str = Field(default="agentops-platform", alias="OTEL_SERVICE_NAME")
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    jaeger_ui_url: str = Field(default="http://localhost:16686", alias="JAEGER_UI_URL")

    # Database
    database_url: str = Field(default="sqlite:///./agentops.db", alias="DATABASE_URL")

    # Mock fallback for test suites and offline operation
    allow_mock_fallback: bool = Field(default=True, alias="ALLOW_MOCK_FALLBACK")

    def get_model_pricing(self, model_name: Optional[str] = None) -> Optional[ModelPricing]:
        """Retrieve model pricing metadata."""
        target = model_name or self.gemini_model
        return DEFAULT_MODEL_PRICING.get(target)

    def calculate_cost(self, input_tokens: int, output_tokens: int, model_name: Optional[str] = None) -> Optional[float]:
        """
        Calculate LLM inference cost using formula:
        (input_tokens * input_price + output_tokens * output_price) / 1,000,000
        """
        pricing = self.get_model_pricing(model_name)
        if not pricing:
            return None
        return (
            (input_tokens * pricing.input_price_per_1m) +
            (output_tokens * pricing.output_price_per_1m)
        ) / 1_000_000.0


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
