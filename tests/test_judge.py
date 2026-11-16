"""Unit tests for Judge Agent."""

import pytest
from agentops.agents.judge import JudgeAgent
from agentops.schemas import (
    DraftReport,
    JudgeFeedback,
    ResearchFindings,
    WorkflowExecutionState,
)


@pytest.mark.asyncio
async def test_judge_agent_approval():
    """Verify Judge Agent approves a properly grounded draft."""
    state = WorkflowExecutionState(
        request_id="test-req-judge",
        trace_id="test-trace-judge",
        task="Evaluate code intelligence framework"
    )

    findings = ResearchFindings(
        summary="AgentOps coordinates multi-agent research.",
        identified_facts=["Built on Google ADK.", "Integrates Gemini API keys."],
        supporting_evidence=["README.md documents the architecture."],
        source_references=["README.md"],
    )

    draft = DraftReport(
        title="AgentOps Architecture",
        executive_summary="AgentOps coordinates multi-agent research.",
        detailed_analysis="Built on Google ADK [Source: README.md]. Integrates Gemini API keys.",
        cited_sources=["README.md"],
    )

    judge = JudgeAgent()
    feedback = await judge.execute(
        task=state.task,
        research_findings=findings,
        draft_report=draft,
        state=state,
    )

    assert isinstance(feedback, JudgeFeedback)
    assert feedback.status == "pass"
    assert feedback.hallucination_score >= 4.0
    assert state.judge_feedback is not None

    events = [step.canonical_event for step in state.trajectory]
    assert "judge.execute" in events
    assert "judge.evaluate" in events
    assert "judge.complete" in events


@pytest.mark.asyncio
async def test_judge_agent_rejection():
    """Verify Judge Agent rejects when prompt indicates fabricated or unsupported claims."""
    state = WorkflowExecutionState(
        request_id="test-req-judge-fail",
        trace_id="test-trace-judge-fail",
        task="Evaluate trigger_reject condition with fabricated claims"
    )

    findings = ResearchFindings(
        summary="Only local SQLite is supported.",
        identified_facts=["SQLite is the local storage engine."],
        supporting_evidence=["agentops/config.py"],
        source_references=["agentops/config.py"],
    )

    draft = DraftReport(
        title="Unverified Cloud DB Report",
        executive_summary="System uses paid enterprise BigQuery warehouse.",
        detailed_analysis="Fabricated deployment on high-cost cloud databases without evidence.",
        cited_sources=[],
    )

    judge = JudgeAgent()
    feedback = await judge.execute(
        task=state.task,
        research_findings=findings,
        draft_report=draft,
        state=state,
    )

    assert isinstance(feedback, JudgeFeedback)
    assert feedback.status == "fail"
    assert len(feedback.revision_instructions) > 0
