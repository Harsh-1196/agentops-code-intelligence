"""Structured agent handoffs and communication contract enforcement."""

import time
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

from agentops.schemas import (
    DraftReport,
    JudgeFeedback,
    ResearchFindings,
    WorkflowExecutionState,
)


class HandoffEnvelope(BaseModel):
    """Structured envelope passed across agent execution boundaries."""
    handoff_id: str
    source_agent: str
    target_agent: str
    payload_type: Literal["task", "research_findings", "draft_report", "judge_feedback", "revision_directive"]
    payload: Dict[str, Any]
    iteration: int
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HandoffValidator:
    """Validates structural prerequisites before executing agent transitions."""

    @staticmethod
    def validate_orchestrator_to_researcher(task: str) -> None:
        if not task or not task.strip():
            raise ValueError("Handoff to Researcher failed: Task string cannot be empty.")

    @staticmethod
    def validate_researcher_to_writer(findings: Optional[ResearchFindings]) -> None:
        if not findings:
            raise ValueError("Handoff to Writer failed: ResearchFindings cannot be None.")
        if not findings.summary or not findings.summary.strip():
            raise ValueError("Handoff to Writer failed: ResearchFindings summary is empty.")
        if not findings.identified_facts:
            raise ValueError("Handoff to Writer failed: ResearchFindings must contain at least one identified fact.")

    @staticmethod
    def validate_writer_to_judge(findings: Optional[ResearchFindings], draft: Optional[DraftReport]) -> None:
        if not findings:
            raise ValueError("Handoff to Judge failed: Baseline ResearchFindings missing.")
        if not draft:
            raise ValueError("Handoff to Judge failed: DraftReport is missing.")
        if not draft.detailed_analysis or not draft.detailed_analysis.strip():
            raise ValueError("Handoff to Judge failed: DraftReport detailed_analysis is empty.")

    @staticmethod
    def validate_judge_to_revision(feedback: Optional[JudgeFeedback]) -> None:
        if not feedback:
            raise ValueError("Handoff to Revision failed: JudgeFeedback is missing.")
        if feedback.status == "fail" and not feedback.revision_instructions.strip():
            raise ValueError("Handoff to Revision failed: Rejection must specify revision instructions.")


class HandoffTracker:
    """Records and verifies handoff history within the workflow execution."""

    def __init__(self):
        self.envelopes: list[HandoffEnvelope] = []

    def record(
        self,
        source: str,
        target: str,
        payload_type: Literal["task", "research_findings", "draft_report", "judge_feedback", "revision_directive"],
        payload: Dict[str, Any],
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HandoffEnvelope:
        import uuid
        env = HandoffEnvelope(
            handoff_id=f"hnd-{uuid.uuid4().hex[:8]}",
            source_agent=source,
            target_agent=target,
            payload_type=payload_type,
            payload=payload,
            iteration=iteration,
            metadata=metadata or {},
        )
        self.envelopes.append(env)
        return env
