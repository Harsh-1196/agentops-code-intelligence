"""Unit tests for latency percentiles and cost accounting."""

import pytest
from agentops.metrics import MetricsCalculator, RequestCostBreakdown, LatencyPercentiles


def test_latency_percentiles():
    """Verify calculation of P50, P95, and average latency."""
    durations = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p = MetricsCalculator.calculate_percentiles(durations)

    assert p.count == 10
    assert p.min_ms == 10.0
    assert p.max_ms == 100.0
    assert p.avg_ms == 55.0
    assert p.p50_ms == 60.0
    assert p.p95_ms == 100.0


def test_cost_breakdown_by_agent():
    """Verify cost attribution across researcher, writer, and judge agents."""
    agent_tokens = {
        "researcher": {"in": 1_000_000, "out": 500_000},
        "writer": {"in": 500_000, "out": 500_000},
        "judge": {"in": 500_000, "out": 100_000},
    }

    breakdown = MetricsCalculator.breakdown_cost(
        agent_tokens=agent_tokens,
        model_name="gemini-2.5-flash",
    )

    assert isinstance(breakdown, RequestCostBreakdown)
    assert breakdown.model_name == "gemini-2.5-flash"
    assert breakdown.input_tokens == 2_000_000
    assert breakdown.output_tokens == 1_100_000
    assert breakdown.total_tokens == 3_100_000

    # researcher: 1M*0.075 + 0.5M*0.30 = 0.075 + 0.15 = 0.225
    assert round(breakdown.researcher_cost_usd, 4) == 0.2250
    assert breakdown.total_cost_usd > 0
