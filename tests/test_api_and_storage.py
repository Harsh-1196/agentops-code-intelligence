"""Integration and unit tests for FastAPI backend and SQLite execution store."""

import os
import pytest
from httpx import AsyncClient, ASGITransport
from agentops.api import app, store
from agentops.schemas import WorkflowExecutionState


@pytest.fixture(autouse=True)
async def setup_test_db(tmp_path):
    """Use a temporary SQLite database for tests."""
    test_db = str(tmp_path / "test_agentops.db")
    store.db_path = test_db
    await store.init_db()
    yield
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_storage_lifecycle():
    """Verify storing, retrieving, and listing workflow executions."""
    state = WorkflowExecutionState(
        request_id="test-store-1",
        trace_id="test-trace-1",
        task="Test storage layer",
        status="approved",
        total_tokens=250,
        total_latency_ms=450.0,
        total_cost_usd=0.00005,
    )

    await store.save_execution(state)

    fetched = await store.get_execution("test-store-1")
    assert fetched is not None
    assert fetched.request_id == "test-store-1"
    assert fetched.total_tokens == 250
    assert fetched.status == "approved"

    runs = await store.list_executions()
    assert len(runs) >= 1
    assert runs[0]["request_id"] == "test-store-1"

    metrics = await store.get_summary_metrics()
    assert metrics["total_requests"] >= 1
    assert metrics["successful_requests"] >= 1
    assert metrics["p50_latency_ms"] > 0


@pytest.mark.asyncio
async def test_fastapi_endpoints():
    """Verify health and orchestration REST endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Health check
        res = await client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "gemini_live" in data

        # Orchestrate request
        orch_res = await client.post(
            "/api/orchestrate",
            json={"task": "Inspect repository architecture", "max_iterations": 2}
        )
        assert orch_res.status_code == 200
        orch_data = orch_res.json()
        assert "request_id" in orch_data
        assert orch_data["status"] in ["approved", "max_iterations_reached"]
        req_id = orch_data["request_id"]

        # History list
        hist_res = await client.get("/api/history")
        assert hist_res.status_code == 200
        assert any(item["request_id"] == req_id for item in hist_res.json())

        # History detail
        detail_res = await client.get(f"/api/history/{req_id}")
        assert detail_res.status_code == 200
        assert detail_res.json()["request_id"] == req_id

        # Metrics
        metric_res = await client.get("/api/metrics")
        assert metric_res.status_code == 200
        assert metric_res.json()["total_requests"] >= 1
