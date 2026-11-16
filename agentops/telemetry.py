"""OpenTelemetry instrumentation, hierarchical spans, and Jaeger OTLP export."""

import logging
import time
from typing import Any, Dict, Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode, Span

from agentops.config import Settings, get_settings

logger = logging.getLogger("agentops.telemetry")

_tracer_initialized = False


def init_telemetry(settings: Optional[Settings] = None) -> trace.Tracer:
    """Initialize OpenTelemetry TracerProvider with OTLP Jaeger exporter."""
    global _tracer_initialized
    cfg = settings or get_settings()

    if _tracer_initialized:
        return trace.get_tracer("agentops", "0.1.0")

    resource = Resource.create({
        "service.name": cfg.otel_service_name,
        "service.version": "0.1.0",
        "deployment.environment": cfg.environment,
    })

    provider = TracerProvider(resource=resource)

    # Configure OTLP HTTP Span Exporter (Jaeger local collector)
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=cfg.otel_exporter_otlp_endpoint,
            timeout=5,
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"OpenTelemetry initialized with OTLP exporter at {cfg.otel_exporter_otlp_endpoint}")
    except Exception as e:
        logger.warning(f"Could not configure OTLP exporter ({e}). Telemetry will record locally.")

    trace.set_tracer_provider(provider)
    _tracer_initialized = True
    return trace.get_tracer("agentops", "0.1.0")


def get_current_trace_id() -> Optional[str]:
    """Retrieve active 32-character hexadecimal OpenTelemetry trace ID."""
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return None


class TelemetryContext:
    """Helper for safely attaching LLM-native token, latency, and cost metadata to spans."""

    @staticmethod
    def annotate_llm_metrics(
        span: Span,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model_name: str,
        cost_usd: Optional[float] = None,
        is_mock: bool = False,
    ) -> None:
        """Attach standardized, safe telemetry attributes without exposing prompts/keys."""
        span.set_attribute("llm.model", model_name)
        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)
        span.set_attribute("llm.total_tokens", input_tokens + output_tokens)
        span.set_attribute("llm.latency_ms", latency_ms)
        span.set_attribute("llm.is_mock", is_mock)
        if cost_usd is not None:
            span.set_attribute("llm.cost_usd", cost_usd)

    @staticmethod
    def annotate_error(span: Span, error: Exception) -> None:
        """Record error safely on span."""
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
