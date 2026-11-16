"""Unit tests for core schemas and workflow state management."""

from agentops.schemas import (
    ResearchFindings,
    DraftReport,
    JudgeFeedback,
    WorkflowExecutionState,
)


def test_research_findings_validation():
    """Verify ResearchFindings fields and defaults."""
    rf = ResearchFindings(
        summary="Analysis of system architecture",
        identified_facts=["Point A", "Point B"],
        supporting_evidence=["Evidence 1"],
        source_references=["file.py#L1-L10"]
    )
    assert rf.summary == "Analysis of system architecture"
    assert len(rf.identified_facts) == 2
    assert len(rf.uncertainties) == 0


def test_judge_feedback_scoring_bounds():
    """Verify JudgeFeedback enforces score bounds (0.0 to 5.0)."""
    jf = JudgeFeedback(
        status="pass",
        overall_score=4.5,
        hallucination_score=5.0,
        response_match_score=4.0,
        revision_instructions="Good draft."
    )
    assert jf.status == "pass"
    assert jf.hallucination_score == 5.0


def test_workflow_state_recording():
    """Verify trajectory recording and agent metrics accumulation."""
    state = WorkflowExecutionState(
        request_id="req-123",
        trace_id="trace-abc",
        task="Evaluate multi-agent design"
    )

    # Record trajectory steps
    s1 = state.record_step(
        agent_name="orchestrator",
        action="classify",
        canonical_event="orchestrator.classify"
    )
    assert s1.step_number == 1
    assert s1.canonical_event == "orchestrator.classify"

    # Record metrics
    state.record_agent_metrics(
        agent_name="researcher",
        input_tokens=100,
        output_tokens=50,
        latency_ms=250.0,
        cost_usd=0.0001
    )
    assert state.total_tokens == 150
    assert state.agent_metrics["researcher"].latency_ms == 250.0
    assert state.agent_metrics["researcher"].call_count == 1
