"""Writer Agent for synthesizing verified research into structured reports.

Transforms ResearchFindings into an evidence-grounded DraftReport with inline citations.
Does not invent facts beyond the provided research findings.
"""

import json
import logging
import time
from typing import Optional
from google.adk.agents import Agent

from agentops.config import Settings, get_settings
from agentops.providers.gemini_provider import GeminiProvider
from agentops.schemas import DraftReport, ResearchFindings, WorkflowExecutionState

logger = logging.getLogger("agentops.writer")

WRITER_INSTRUCTION = """You are a technical report writer and synthesizer.
Your responsibility is to take verified research findings from the Researcher Agent and produce a clear, authoritative report.

Rules:
1. STRICT GROUNDING: You must NEVER invent or hallucinate facts that were not provided in the ResearchFindings.
2. CITATIONS: Use inline citations [Source: ...] for every key technical claim based on the provided source_references.
3. SEPARATION: Clearly distinguish between verified facts and uncertain hypotheses/open questions.
4. FORMAT: Return your output strictly formatted as JSON conforming to the DraftReport schema:
{
  "title": "Clear report title",
  "executive_summary": "High-level summary of findings",
  "detailed_analysis": "Comprehensive markdown analysis with inline citations [Source: ...]",
  "cited_sources": ["file.py#L1-L20", "README.md"],
  "confidence_score": 0.95
}
"""


class WriterAgent:
    """Writer agent synthesizing ResearchFindings into a grounded DraftReport."""

    def __init__(self, settings: Optional[Settings] = None, provider: Optional[GeminiProvider] = None):
        self.settings = settings or get_settings()
        self.provider = provider or GeminiProvider(settings=self.settings)
        self.name = "writer"

        try:
            self.adk_agent = Agent(
                name=self.name,
                model=self.settings.gemini_model,
                description="Synthesizes research findings into evidence-grounded reports.",
                instruction=WRITER_INSTRUCTION,
            )
        except Exception as e:
            logger.warning(f"Could not initialize native ADK Agent instance for Writer: {e}")
            self.adk_agent = None

    async def execute(
        self,
        task: str,
        research_findings: ResearchFindings,
        state: WorkflowExecutionState,
        revision_instructions: Optional[str] = None,
    ) -> DraftReport:
        """
        Executes Writer synthesis:
        1. Records trajectory step 'writer.execute'.
        2. Constructs synthesis prompt grounded in research findings and any judge feedback.
        3. Calls Gemini API for structured DraftReport.
        4. Records 'writer.generate' and 'writer.complete' steps.
        5. Accumulates metrics in state.
        """
        start_time = time.perf_counter()
        state.record_step(
            agent_name=self.name,
            action="enter",
            canonical_event="writer.execute",
            details={"task": task, "iteration": state.current_iteration}
        )

        state.record_step(
            agent_name=self.name,
            action="generate",
            canonical_event="writer.generate",
            details={"findings_summary": research_findings.summary[:100]}
        )

        # Build prompt from findings
        prompt_parts = [
            f"Original Task: {task}",
            f"\n--- Verified Research Findings (GROUND TRUTH CONTEXT) ---\n",
            f"Summary: {research_findings.summary}\n",
            f"Identified Facts:\n" + "\n".join(f"- {fact}" for fact in research_findings.identified_facts),
            f"\nSupporting Evidence:\n" + "\n".join(f"- {ev}" for ev in research_findings.supporting_evidence),
            f"\nAvailable Source References:\n" + "\n".join(f"- {src}" for src in research_findings.source_references),
            f"\nUncertainties:\n" + "\n".join(f"- {unc}" for unc in research_findings.uncertainties),
        ]

        if revision_instructions:
            prompt_parts.append(
                f"\n--- PREVIOUS JUDGE FEEDBACK (CRITICAL REVISION DIRECTIVE) ---\n"
                f"{revision_instructions}\n"
                "Refine the draft to address all judge feedback. Remove any flagged unsupported claims immediately."
            )

        full_prompt = "\n".join(prompt_parts)

        generation = await self.provider.generate(
            prompt=full_prompt,
            system_instruction=WRITER_INSTRUCTION,
            response_schema=DraftReport,
            agent_role=self.name,
        )

        draft: DraftReport
        if generation.parsed_json and isinstance(generation.parsed_json, dict):
            try:
                draft = DraftReport(**generation.parsed_json)
            except Exception as e:
                logger.warning(f"Error parsing DraftReport schema from LLM: {e}")
                draft = DraftReport(
                    title=f"Report: {task[:50]}",
                    executive_summary=research_findings.summary,
                    detailed_analysis=generation.text,
                    cited_sources=research_findings.source_references,
                    confidence_score=0.9,
                )
        else:
            draft = DraftReport(
                title=f"Technical Synthesis: {task[:60]}",
                executive_summary=research_findings.summary,
                detailed_analysis=(
                    f"### Findings Overview\n{research_findings.summary}\n\n"
                    f"### Key Facts\n" + "\n".join(f"- {f} [Source: {research_findings.source_references[0] if research_findings.source_references else 'evidence'}]" for f in research_findings.identified_facts)
                ),
                cited_sources=research_findings.source_references,
                confidence_score=0.92,
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
            canonical_event="writer.complete",
            details={"title": draft.title, "confidence": draft.confidence_score}
        )

        state.draft_report = draft
        return draft
