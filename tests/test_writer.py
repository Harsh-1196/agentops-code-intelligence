"""Unit tests for Writer Agent."""

import pytest
from agentops.agents.writer import WriterAgent
from agentops.schemas import DraftReport, ResearchFindings, WorkflowExecutionState


@pytest.mark.asyncio
async def test_writer_agent_execution():
    """Verify Writer Agent synthesizes ResearchFindings into a structured DraftReport."""
    state = WorkflowExecutionState(
        request_id="test-req-writer",
        trace_id="test-trace-writer",
        task="Synthesize evaluation methodology for multi-agent systems"
    )

    findings = ResearchFindings(
        summary="Evaluation requires trajectory exact match, classification metrics, and LLM-as-a-Judge.",
        identified_facts=[
            "Trajectory exact match compares normalized canonical event sequences.",
            "LLM-as-a-Judge evaluates hallucination and response match on 0-5 rubrics."
        ],
        supporting_evidence=["docs/architecture.md outlines the evaluation methodology."],
        source_references=["docs/architecture.md#L40-L60"],
    )

    writer = WriterAgent()
    draft = await writer.execute(task=state.task, research_findings=findings, state=state)

    assert isinstance(draft, DraftReport)
    assert len(draft.title) > 0
    assert len(draft.executive_summary) > 0
    assert len(draft.detailed_analysis) > 0
    assert state.draft_report is not None

    # Check trajectory events
    events = [step.canonical_event for step in state.trajectory]
    assert "writer.execute" in events
    assert "writer.generate" in events
    assert "writer.complete" in events

    # Check metrics
    assert "writer" in state.agent_metrics
    assert state.agent_metrics["writer"].total_tokens > 0
    assert state.agent_metrics["writer"].latency_ms > 0
