"""Unit tests for structured agent handoffs and validation."""

import pytest
from agentops.handoffs import HandoffTracker, HandoffValidator
from agentops.schemas import DraftReport, ResearchFindings, JudgeFeedback


def test_handoff_validator_checks():
    """Verify HandoffValidator catches missing or invalid data across boundaries."""
    # Orchestrator to researcher: empty task fails
    with pytest.raises(ValueError, match="Task string cannot be empty"):
        HandoffValidator.validate_orchestrator_to_researcher("")

    # Researcher to writer: None findings fails
    with pytest.raises(ValueError, match="ResearchFindings cannot be None"):
        HandoffValidator.validate_researcher_to_writer(None)

    # Empty facts fails
    invalid_findings = ResearchFindings(summary="Summary", identified_facts=[])
    with pytest.raises(ValueError, match="at least one identified fact"):
        HandoffValidator.validate_researcher_to_writer(invalid_findings)

    # Writer to judge: empty analysis fails
    valid_findings = ResearchFindings(
        summary="Summary",
        identified_facts=["Fact 1"],
        supporting_evidence=["Ev 1"]
    )
    invalid_draft = DraftReport(
        title="Title",
        executive_summary="Exec",
        detailed_analysis="",
        cited_sources=[]
    )
    with pytest.raises(ValueError, match="detailed_analysis is empty"):
        HandoffValidator.validate_writer_to_judge(valid_findings, invalid_draft)


def test_handoff_tracker_records():
    """Verify HandoffTracker generates valid structured envelopes."""
    tracker = HandoffTracker()
    env = tracker.record(
        source="researcher",
        target="writer",
        payload_type="research_findings",
        payload={"summary": "Overview", "facts": ["A"]},
        iteration=1
    )
    assert env.source_agent == "researcher"
    assert env.target_agent == "writer"
    assert env.iteration == 1
    assert len(tracker.envelopes) == 1
