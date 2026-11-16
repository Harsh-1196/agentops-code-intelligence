"""Unit tests for Researcher Agent and tools."""

import pytest
from agentops.agents.researcher import ResearcherAgent
from agentops.schemas import WorkflowExecutionState, ResearchFindings
from agentops.tools.research_tools import (
    inspect_repository_structure,
    read_source_file,
    search_codebase,
    retrieve_web_documentation,
)


def test_research_tools_execution():
    """Verify inspection tools execute cleanly against current repository."""
    # Repository inspection
    struct = inspect_repository_structure(".")
    assert "Repository Structure Overview" in struct
    assert "pyproject.toml" in struct

    # Read source file
    content = read_source_file("pyproject.toml", start_line=1, end_line=10)
    assert "pyproject.toml" in content
    assert "agentops" in content

    # Search codebase
    search_res = search_codebase("ResearcherAgent")
    assert "search_codebase" in search_res or "ResearcherAgent" in search_res

    # Retrieve web doc
    doc = retrieve_web_documentation("google_adk")
    assert "Google Agent Development Kit" in doc


@pytest.mark.asyncio
async def test_researcher_agent_execution():
    """Verify Researcher Agent produces structured ResearchFindings and records trajectory."""
    state = WorkflowExecutionState(
        request_id="test-req-res",
        trace_id="test-trace-res",
        task="Investigate ADK orchestration mechanisms in the codebase"
    )

    agent = ResearcherAgent()
    findings = await agent.execute(task=state.task, state=state)

    assert isinstance(findings, ResearchFindings)
    assert len(findings.identified_facts) > 0
    assert len(findings.supporting_evidence) > 0
    assert state.research_findings is not None

    # Check trajectory steps
    events = [step.canonical_event for step in state.trajectory]
    assert "researcher.execute" in events
    assert "researcher.retrieve_evidence" in events
    assert "researcher.complete" in events

    # Check metrics
    assert "researcher" in state.agent_metrics
    assert state.agent_metrics["researcher"].total_tokens > 0
    assert state.agent_metrics["researcher"].latency_ms > 0
