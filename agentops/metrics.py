"""LLM-native metrics accounting: tokens, latency percentiles, and cost calculation."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from agentops.config import DEFAULT_MODEL_PRICING, ModelPricing, get_settings


class RequestCostBreakdown(BaseModel):
    """Detailed dollar cost breakdown across agent roles and model calls."""
    model_name: str
    input_price_per_1m: float
    output_price_per_1m: float
    currency: str = "USD"
    pricing_source: str
    verification_date: str

    input_tokens: int
    output_tokens: int
    total_tokens: int

    researcher_cost_usd: float = 0.0
    writer_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


class LatencyPercentiles(BaseModel):
    """Latency distribution percentiles (in milliseconds)."""
    count: int
    min_ms: float
    max_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class MetricsCalculator:
    """Calculates LLM token costs and percentile latencies."""

    @staticmethod
    def calculate_percentiles(durations_ms: List[float]) -> LatencyPercentiles:
        """Computes min, max, avg, P50, P95, and P99 latency percentiles."""
        if not durations_ms:
            return LatencyPercentiles(
                count=0, min_ms=0.0, max_ms=0.0, avg_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0
            )

        sorted_d = sorted(durations_ms)
        n = len(sorted_d)
        avg = sum(sorted_d) / n

        p50 = sorted_d[int(n * 0.50)]
        p95 = sorted_d[min(int(n * 0.95), n - 1)]
        p99 = sorted_d[min(int(n * 0.99), n - 1)]

        return LatencyPercentiles(
            count=n,
            min_ms=round(sorted_d[0], 2),
            max_ms=round(sorted_d[-1], 2),
            avg_ms=round(avg, 2),
            p50_ms=round(p50, 2),
            p95_ms=round(p95, 2),
            p99_ms=round(p99, 2),
        )

    @staticmethod
    def calculate_cost(
        input_tokens: int,
        output_tokens: int,
        model_name: Optional[str] = None,
    ) -> Optional[float]:
        """
        Calculates exact inference cost using verified Gemini API pricing:
        Cost = (input_tokens * input_price + output_tokens * output_price) / 1,000,000
        """
        model = model_name or get_settings().gemini_model
        pricing = DEFAULT_MODEL_PRICING.get(model)
        if not pricing:
            return None

        cost = (
            (input_tokens * pricing.input_price_per_1m) +
            (output_tokens * pricing.output_price_per_1m)
        ) / 1_000_000.0
        return round(cost, 8)

    @classmethod
    def breakdown_cost(
        cls,
        agent_tokens: Dict[str, Dict[str, int]],
        model_name: Optional[str] = None,
    ) -> RequestCostBreakdown:
        """
        Computes detailed cost attribution per agent role.
        agent_tokens format: {"researcher": {"in": 100, "out": 50}, ...}
        """
        model = model_name or get_settings().gemini_model
        pricing = DEFAULT_MODEL_PRICING.get(model, ModelPricing(
            input_price_per_1m=0.075,
            output_price_per_1m=0.30,
            currency="USD"
        ))

        tot_in = sum(v.get("in", 0) for v in agent_tokens.values())
        tot_out = sum(v.get("out", 0) for v in agent_tokens.values())

        def agent_c(role: str) -> float:
            d = agent_tokens.get(role, {})
            c = cls.calculate_cost(d.get("in", 0), d.get("out", 0), model)
            return c or 0.0

        res_c = agent_c("researcher")
        wrt_c = agent_c("writer")
        jdg_c = agent_c("judge")
        tot_c = res_c + wrt_c + jdg_c

        return RequestCostBreakdown(
            model_name=model,
            input_price_per_1m=pricing.input_price_per_1m,
            output_price_per_1m=pricing.output_price_per_1m,
            currency=pricing.currency,
            pricing_source=pricing.pricing_source,
            verification_date=pricing.verification_date,
            input_tokens=tot_in,
            output_tokens=tot_out,
            total_tokens=tot_in + tot_out,
            researcher_cost_usd=round(res_c, 8),
            writer_cost_usd=round(wrt_c, 8),
            judge_cost_usd=round(jdg_c, 8),
            total_cost_usd=round(tot_c, 8),
        )
