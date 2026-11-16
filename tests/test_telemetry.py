"""Unit tests for OpenTelemetry tracing and span annotations."""

import pytest
from opentelemetry import trace
from agentops.config import Settings
from agentops.telemetry import init_telemetry, TelemetryContext, get_current_trace_id


def test_telemetry_initialization():
    """Verify OpenTelemetry tracer initializes and creates active spans."""
    settings = Settings(otel_service_name="test-agentops")
    tracer = init_telemetry(settings)
    assert tracer is not None

    with tracer.start_as_current_span("test.span") as span:
        trace_id = get_current_trace_id()
        assert trace_id is not None
        assert len(trace_id) == 32

        # Annotate LLM metrics
        TelemetryContext.annotate_llm_metrics(
            span=span,
            input_tokens=150,
            output_tokens=75,
            latency_ms=320.0,
            model_name="gemini-2.5-flash",
            cost_usd=0.00003,
            is_mock=True,
        )

        assert span.is_recording()
