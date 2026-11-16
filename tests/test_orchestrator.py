"""Tests for Google ADK Orchestrator and bounded loop execution."""

import pytest
from unittest.mock import AsyncMock
from agentops.orchestrator import ADKOrchestrator
from agentops.schemas import (
    JudgeFeedback,
    WorkflowExecutionState,
)


@pytest.mark.asyncio
async def test_orchestrator_successful_approval():
    """Verify standard single-pass workflow approval and trajectory generation."""
    orchestrator = ADKOrchestrator()
    state = await orchestrator.run(
        task="Investigate repository architecture and tracing setup",
        max_iterations=3,
    )

    assert state.status == "approved"
    assert state.current_iteration == 1
    assert state.draft_report is not None
    assert state.final_response is not None
    assert "Approved" in state.final_response

    events = [s.canonical_event for s in state.trajectory]
    assert "orchestrator.classify" in events
    assert "researcher.execute" in events
    assert "writer.generate" in events
    assert "judge.evaluate" in events
    assert "orchestrator.approve" in events

    # Ensure tokens and latency are tracked
    assert state.total_tokens > 0
    assert state.total_latency_ms > 0


@pytest.mark.asyncio
async def test_orchestrator_rejection_and_revision_cycle():
    """Verify rejection-and-revision cycle: Judge rejects in iter 1, then approves in iter 2."""
    orchestrator = ADKOrchestrator()

    # Mock Judge to reject on iteration 1 and pass on iteration 2
    iter_counter = 0

    async def mock_judge_execute(task, research_findings, draft_report, state):
        nonlocal iter_counter
        iter_counter += 1
        state.record_step(
            agent_name="judge",
            action="enter",
            canonical_event="judge.execute"
        )
        state.record_step(
            agent_name="judge",
            action="evaluate",
            canonical_event="judge.evaluate"
        )
        if iter_counter == 1:
            fb = JudgeFeedback(
                status="fail",
                overall_score=2.0,
                hallucination_score=2.0,
                response_match_score=3.0,
                unsupported_claims=["Deployment claims lack evidence."],
                missing_information=["Add file citations."],
                revision_instructions="Remove deployment claim and cite README."
            )
        else:
            fb = JudgeFeedback(
                status="pass",
                overall_score=4.8,
                hallucination_score=5.0,
                response_match_score=4.8,
                unsupported_claims=[],
                missing_information=[],
                revision_instructions="Approved after revision."
            )
        state.record_step(
            agent_name="judge",
            action="exit",
            canonical_event="judge.complete"
        )
        state.judge_feedback = fb
        return fb

    orchestrator.judge.execute = mock_judge_execute

    state = await orchestrator.run(
        task="Test rejection and revision workflow",
        max_iterations=3,
    )

    assert state.status == "approved"
    assert state.current_iteration == 2
    assert len(state.revision_history) == 1
    assert state.revision_history[0]["iteration"] == 1

    events = [s.canonical_event for s in state.trajectory]
    assert "orchestrator.reject" in events
    assert "orchestrator.revision_cycle" in events
    assert "orchestrator.approve" in events


@pytest.mark.asyncio
async def test_orchestrator_max_iterations_bounded_termination():
    """Verify loop strictly terminates when max_iterations is reached without infinite looping."""
    orchestrator = ADKOrchestrator()

    # Force judge to always reject
    async def always_reject(task, research_findings, draft_report, state):
        state.record_step("judge", "enter", "judge.execute")
        state.record_step("judge", "evaluate", "judge.evaluate")
        fb = JudgeFeedback(
            status="fail",
            overall_score=1.5,
            hallucination_score=1.0,
            response_match_score=2.0,
            unsupported_claims=["Always ungrounded"],
            missing_information=["Everything missing"],
            revision_instructions="Try again"
        )
        state.record_step("judge", "exit", "judge.complete")
        state.judge_feedback = fb
        return fb

    orchestrator.judge.execute = always_reject

    state = await orchestrator.run(
        task="Test bounded loop limit",
        max_iterations=2,
    )

    assert state.status == "max_iterations_reached"
    assert state.current_iteration == 2
    assert len(state.revision_history) == 2

    events = [s.canonical_event for s in state.trajectory]
    assert "orchestrator.max_iterations" in events
    assert "orchestrator.approve" not in events
