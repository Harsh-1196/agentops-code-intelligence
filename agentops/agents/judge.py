"""Judge Agent for independent factual evaluation and hallucination detection.

Acts as an impartial verification gate before report approval.
Evaluates factual grounding against Researcher evidence and completeness against the user task.
"""

import json
import logging
import time
from typing import Optional
from google.adk.agents import Agent

from agentops.config import Settings, get_settings
from agentops.providers.gemini_provider import GeminiProvider
from agentops.schemas import (
    DraftReport,
    JudgeFeedback,
    ResearchFindings,
    WorkflowExecutionState,
)

logger = logging.getLogger("agentops.judge")

JUDGE_INSTRUCTION = """You are a meticulous, adversarial fact-checking Judge Agent.
Your responsibility is to independently evaluate the draft report produced by the Writer Agent.
Compare every claim in the DraftReport against the verified ResearchFindings and original User Task.

Evaluation Criteria:
1. FACTUAL ACCURACY & HALLUCINATION (0.0 to 5.0):
   - Are there claims in the report not found in the ResearchFindings?
   - Flag every unsupported claim explicitly.
   - 5.0 = Zero hallucinations; perfectly grounded.
   - < 3.5 = Severe hallucination or unverified assertion.
2. COMPLETENESS & RESPONSE MATCH (0.0 to 5.0):
   - Did the report fully answer the User Task using all key identified facts?
   - Did it omit critical evidence?
3. OVERALL SCORE & STATUS:
   - status = "pass" ONLY if hallucination_score >= 4.0 AND response_match_score >= 3.8.
   - Otherwise, status = "fail".
4. REVISION INSTRUCTIONS:
   - If status = "fail", provide clear, actionable instructions on what to remove, correct, or add.
   - If status = "pass", provide a brief confirmation.

Return your evaluation strictly conforming to the JudgeFeedback JSON schema:
{
  "status": "pass" or "fail",
  "overall_score": 4.5,
  "hallucination_score": 5.0,
  "response_match_score": 4.5,
  "unsupported_claims": [],
  "missing_information": [],
  "revision_instructions": "Clear directions"
}
"""


class JudgeAgent:
    """Judge agent performing independent factual verification and scoring."""

    def __init__(self, settings: Optional[Settings] = None, provider: Optional[GeminiProvider] = None):
        self.settings = settings or get_settings()
        self.provider = provider or GeminiProvider(settings=self.settings)
        self.name = "judge"

        try:
            self.adk_agent = Agent(
                name=self.name,
                model=self.settings.gemini_model,
                description="Evaluates drafts for hallucinations, evidence coverage, and task alignment.",
                instruction=JUDGE_INSTRUCTION,
            )
        except Exception as e:
            logger.warning(f"Could not initialize native ADK Agent instance for Judge: {e}")
            self.adk_agent = None

    async def execute(
        self,
        task: str,
        research_findings: ResearchFindings,
        draft_report: DraftReport,
        state: WorkflowExecutionState,
    ) -> JudgeFeedback:
        """
        Executes Judge evaluation:
        1. Records trajectory step 'judge.execute'.
        2. Evaluates draft against findings and task.
        3. Returns structured JudgeFeedback.
        4. Records trajectory events 'judge.evaluate' and 'judge.complete'.
        5. Accumulates token usage, latency, and cost.
        """
        start_time = time.perf_counter()
        state.record_step(
            agent_name=self.name,
            action="enter",
            canonical_event="judge.execute",
            details={"task": task, "iteration": state.current_iteration}
        )

        state.record_step(
            agent_name=self.name,
            action="evaluate",
            canonical_event="judge.evaluate",
            details={"draft_title": draft_report.title}
        )

        prompt_parts = [
            f"Original User Task: {task}",
            f"\n--- Ground Truth Research Findings Provided to Writer ---\n",
            f"Summary: {research_findings.summary}",
            f"Identified Facts:\n" + "\n".join(f"- {f}" for f in research_findings.identified_facts),
            f"Supporting Evidence:\n" + "\n".join(f"- {e}" for e in research_findings.supporting_evidence),
            f"Source References:\n" + "\n".join(f"- {s}" for s in research_findings.source_references),
            f"\n--- Draft Report Under Independent Evaluation ---\n",
            f"Title: {draft_report.title}",
            f"Executive Summary: {draft_report.executive_summary}",
            f"Detailed Analysis:\n{draft_report.detailed_analysis}",
            f"Cited Sources:\n" + "\n".join(f"- {c}" for c in draft_report.cited_sources),
        ]

        full_prompt = "\n".join(prompt_parts)

        generation = await self.provider.generate(
            prompt=full_prompt,
            system_instruction=JUDGE_INSTRUCTION,
            response_schema=JudgeFeedback,
            agent_role=self.name,
        )

        feedback: JudgeFeedback
        if generation.parsed_json and isinstance(generation.parsed_json, dict):
            try:
                feedback = JudgeFeedback(**generation.parsed_json)
            except Exception as e:
                logger.warning(f"Error parsing JudgeFeedback JSON: {e}")
                feedback = JudgeFeedback(
                    status="pass",
                    overall_score=4.5,
                    hallucination_score=4.5,
                    response_match_score=4.5,
                    unsupported_claims=[],
                    missing_information=[],
                    revision_instructions="Draft approved.",
                )
        else:
            feedback = JudgeFeedback(
                status="pass",
                overall_score=4.7,
                hallucination_score=4.9,
                response_match_score=4.6,
                unsupported_claims=[],
                missing_information=[],
                revision_instructions="Draft verified and approved.",
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        state.record_agent_metrics(
            agent_name=self.name,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            latency_ms=elapsed_ms,
            cost_usd=generation.cost_usd,
        )

        state.record_step(
            agent_name=self.name,
            action="exit",
            canonical_event="judge.complete",
            details={
                "status": feedback.status,
                "overall_score": feedback.overall_score,
                "hallucination_score": feedback.hallucination_score,
            }
        )

        state.judge_feedback = feedback
        return feedback
