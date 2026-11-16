"""Google ADK Multi-Agent Orchestrator with bounded iterative loop.

Coordinates Researcher, Writer, and Judge agents.
Guarantees loop termination via Judge approval or max iteration limit.
"""

import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional
from google.adk.agents import BaseAgent, LoopAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from opentelemetry import trace

from agentops.config import Settings, get_settings
from agentops.providers.gemini_provider import GeminiProvider
from agentops.agents.researcher import ResearcherAgent
from agentops.agents.writer import WriterAgent
from agentops.agents.judge import JudgeAgent
from agentops.handoffs import HandoffTracker, HandoffValidator
from agentops.telemetry import init_telemetry, TelemetryContext
from agentops.schemas import (
    WorkflowExecutionState,
    ResearchFindings,
    DraftReport,
    JudgeFeedback,
)

logger = logging.getLogger("agentops.orchestrator")


class EscalationChecker(BaseAgent):
    """
    ADK agent that checks the Judge's feedback in session state.
    Escalates (breaks the ADK LoopAgent) if Judge feedback is 'pass'.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        feedback = ctx.session.state.get("judge_feedback")
        if feedback and isinstance(feedback, dict) and feedback.get("status") == "pass":
            logger.info("[EscalationChecker] Judge feedback PASSED. Escalating to terminate loop.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        elif isinstance(feedback, str) and '"status": "pass"' in feedback:
            logger.info("[EscalationChecker] Judge feedback PASSED (str). Escalating.")
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            logger.info("[EscalationChecker] Judge feedback did not pass. Continuing loop.")
            yield Event(author=self.name)


class ADKOrchestrator:
    """
    Main Multi-Agent Orchestrator managing workflow execution,
    bounded revision cycles, telemetry recording, and state handoffs.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        provider: Optional[GeminiProvider] = None,
        researcher: Optional[ResearcherAgent] = None,
        writer: Optional[WriterAgent] = None,
        judge: Optional[JudgeAgent] = None,
    ):
        self.settings = settings or get_settings()
        self.provider = provider or GeminiProvider(settings=self.settings)
        self.researcher = researcher or ResearcherAgent(settings=self.settings, provider=self.provider)
        self.writer = writer or WriterAgent(settings=self.settings, provider=self.provider)
        self.judge = judge or JudgeAgent(settings=self.settings, provider=self.provider)
        self.escalation_checker = EscalationChecker(name="escalation_checker")
        self.tracer = init_telemetry(self.settings)

        # ADK Loop Agent structure
        try:
            sub_agents = []
            if self.researcher.adk_agent:
                sub_agents.append(self.researcher.adk_agent)
            if self.writer.adk_agent:
                sub_agents.append(self.writer.adk_agent)
            if self.judge.adk_agent:
                sub_agents.append(self.judge.adk_agent)
            sub_agents.append(self.escalation_checker)

            self.adk_loop = LoopAgent(
                name="research_and_verification_loop",
                description="Iteratively researches, writes, and fact-checks until standards are met.",
                sub_agents=sub_agents,
                max_iterations=self.settings.max_agent_iterations,
            )
        except Exception as e:
            logger.warning(f"Could not build full ADK LoopAgent graph: {e}")
            self.adk_loop = None

    def classify_task(self, task: str) -> Dict[str, str]:
        """Classifies incoming task category and complexity."""
        t_lower = task.lower()
        if any(w in t_lower for w in ["repo", "code", "file", "github", "bug", "refactor"]):
            category = "repository_investigation"
        elif any(w in t_lower for w in ["summarize", "overview", "tldr"]):
            category = "source_grounded_summarization"
        elif any(w in t_lower for w in ["why", "how", "explain", "architecture"]):
            category = "technical_explanation"
        elif any(w in t_lower for w in ["compare", "benchmark", "difference"]):
            category = "multi_step_reasoning"
        else:
            category = "factual_research"

        return {
            "category": category,
            "priority": "high",
            "requires_tools": "true",
        }

    async def run(
        self,
        task: str,
        repo_url: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> WorkflowExecutionState:
        """
        Runs the complete bounded multi-agent workflow:
        1. Classifies task.
        2. Executes bounded loop (Researcher -> Writer -> Judge).
        3. Handles rejection & revision if Judge fails.
        4. Terminates on approval or maximum iterations.
        5. Finalizes response and aggregates metrics.
        """
        req_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        tr_id = trace_id or f"trace-{uuid.uuid4().hex[:16]}"
        max_iters = max_iterations or self.settings.max_agent_iterations
        start_time = time.perf_counter()

        state = WorkflowExecutionState(
            request_id=req_id,
            trace_id=tr_id,
            task=task,
            repo_url=repo_url,
            max_iterations=max_iters,
            status="running",
            is_mock=not self.provider.is_live,
        )

        with self.tracer.start_as_current_span("request.incoming") as root_span:
            root_span.set_attribute("request.id", req_id)
            root_span.set_attribute("task", task[:120])
            root_span.set_attribute("llm.model", self.settings.gemini_model)
            root_span.set_attribute("is_mock", not self.provider.is_live)

            # Extract active trace ID
            span_ctx = root_span.get_span_context()
            if span_ctx and span_ctx.is_valid:
                tr_id = format(span_ctx.trace_id, "032x")
                state.trace_id = tr_id

            with self.tracer.start_as_current_span("agent.orchestrator") as orch_span:
                # 1. Classification Step
                state.record_step(
                    agent_name="orchestrator",
                    action="classify",
                    canonical_event="orchestrator.classify",
                    details={"task": task, "repo_url": repo_url}
                )
                classification = self.classify_task(task)
                orch_span.set_attribute("task.category", classification["category"])

                handoff_tracker = HandoffTracker()
                revision_instructions: Optional[str] = None
                current_iteration = 1
                approved = False

                while current_iteration <= max_iters:
                    state.current_iteration = current_iteration
                    logger.info(f"--- Starting Orchestrator Iteration {current_iteration}/{max_iters} ---")

                    try:
                        async with asyncio.timeout(self.settings.agent_timeout_seconds):
                            # Step A: Researcher Agent Handoff & Execution
                            HandoffValidator.validate_orchestrator_to_researcher(task)
                            handoff_tracker.record(
                                source="orchestrator",
                                target="researcher",
                                payload_type="task",
                                payload={"task": task, "revision_instructions": revision_instructions},
                                iteration=current_iteration,
                            )

                            with self.tracer.start_as_current_span(f"iteration_{current_iteration}.researcher") as res_span:
                                findings = await self.researcher.execute(
                                    task=task,
                                    state=state,
                                    revision_instructions=revision_instructions,
                                )
                                res_span.set_attribute("findings.facts_count", len(findings.identified_facts))

                            # Step B: Writer Agent Handoff & Execution
                            HandoffValidator.validate_researcher_to_writer(findings)
                            handoff_tracker.record(
                                source="researcher",
                                target="writer",
                                payload_type="research_findings",
                                payload=findings.model_dump(),
                                iteration=current_iteration,
                            )

                            with self.tracer.start_as_current_span(f"iteration_{current_iteration}.writer") as wrt_span:
                                draft = await self.writer.execute(
                                    task=task,
                                    research_findings=findings,
                                    state=state,
                                    revision_instructions=revision_instructions,
                                )
                                wrt_span.set_attribute("draft.title", draft.title)
                                wrt_span.set_attribute("draft.confidence", draft.confidence_score)

                            # Step C: Judge Agent Handoff & Execution
                            HandoffValidator.validate_writer_to_judge(findings, draft)
                            handoff_tracker.record(
                                source="writer",
                                target="judge",
                                payload_type="draft_report",
                                payload=draft.model_dump(),
                                iteration=current_iteration,
                            )

                            with self.tracer.start_as_current_span(f"iteration_{current_iteration}.judge") as jdg_span:
                                feedback = await self.judge.execute(
                                    task=task,
                                    research_findings=findings,
                                    draft_report=draft,
                                    state=state,
                                )
                                jdg_span.set_attribute("judge.status", feedback.status)
                                jdg_span.set_attribute("judge.overall_score", feedback.overall_score)
                                jdg_span.set_attribute("judge.hallucination_score", feedback.hallucination_score)

                            # Step D: Inspect Judge Evaluation
                            if feedback.status == "pass":
                                logger.info(f"Iteration {current_iteration}: Judge APPROVED the draft (score: {feedback.overall_score})")
                                state.record_step(
                                    agent_name="orchestrator",
                                    action="approve",
                                    canonical_event="orchestrator.approve",
                                    details={"score": feedback.overall_score, "iteration": current_iteration}
                                )
                                approved = True
                                state.status = "approved"
                                break
                            else:
                                logger.warning(
                                    f"Iteration {current_iteration}: Judge REJECTED the draft. "
                                    f"Hallucination: {feedback.hallucination_score}. Instructions: {feedback.revision_instructions}"
                                )
                                state.record_step(
                                    agent_name="orchestrator",
                                    action="reject",
                                    canonical_event="orchestrator.reject",
                                    details={
                                        "rejection_reasons": feedback.unsupported_claims or [feedback.revision_instructions],
                                        "iteration": current_iteration,
                                    }
                                )

                                state.revision_history.append({
                                    "iteration": current_iteration,
                                    "hallucination_score": feedback.hallucination_score,
                                    "unsupported_claims": feedback.unsupported_claims,
                                    "missing_information": feedback.missing_information,
                                    "instructions": feedback.revision_instructions,
                                })

                                revision_instructions = feedback.revision_instructions

                                if current_iteration < max_iters:
                                    with self.tracer.start_as_current_span(f"iteration_{current_iteration}.revision_cycle"):
                                        state.record_step(
                                            agent_name="orchestrator",
                                            action="revision_cycle",
                                            canonical_event="orchestrator.revision_cycle",
                                            details={"target_iteration": current_iteration + 1}
                                        )

                    except asyncio.TimeoutError:
                        logger.error(f"Iteration {current_iteration} timed out after {self.settings.agent_timeout_seconds}s")
                        state.error_message = f"Iteration {current_iteration} exceeded timeout."
                        break
                    except Exception as e:
                        logger.error(f"Error in orchestrator iteration {current_iteration}: {e}", exc_info=True)
                        state.error_message = str(e)
                        break

                    current_iteration += 1

                # Final loop termination resolution
                if not approved:
                    if state.current_iteration >= max_iters:
                        state.status = "max_iterations_reached"
                        state.record_step(
                            agent_name="orchestrator",
                            action="max_iterations",
                            canonical_event="orchestrator.max_iterations",
                            details={"total_iterations": state.current_iteration}
                        )
                    else:
                        state.status = "failed"

            # Final response synthesis with telemetry span
            with self.tracer.start_as_current_span("workflow.final_response"):
                if state.draft_report:
                    state.final_response = (
                        f"# {state.draft_report.title}\n\n"
                        f"**Status**: {'Approved' if approved else 'Best Effort (Max Iterations Reached)'} "
                        f"(Iterations: {state.current_iteration}/{max_iters})\n\n"
                        f"## Executive Summary\n{state.draft_report.executive_summary}\n\n"
                        f"## Detailed Analysis\n{state.draft_report.detailed_analysis}\n\n"
                        f"## Cited Sources\n" + "\n".join(f"- `{s}`" for s in state.draft_report.cited_sources)
                    )
                else:
                    state.final_response = "Workflow completed with no draft generated."

            total_elapsed = (time.perf_counter() - start_time) * 1000.0
            state.total_latency_ms = round(total_elapsed, 2)
            state.handoff_envelopes = [e.model_dump() for e in handoff_tracker.envelopes]
            from datetime import datetime, timezone
            state.completed_at = datetime.now(timezone.utc).isoformat()

            # Record final request metrics on root span
            TelemetryContext.annotate_llm_metrics(
                span=root_span,
                input_tokens=state.total_input_tokens,
                output_tokens=state.total_output_tokens,
                latency_ms=state.total_latency_ms,
                model_name=self.settings.gemini_model,
                cost_usd=state.total_cost_usd,
                is_mock=state.is_mock,
            )

        logger.info(
            f"Workflow finished: status={state.status}, tokens={state.total_tokens}, "
            f"latency={state.total_latency_ms:.1f}ms, cost=${state.total_cost_usd or 0:.6f}"
        )
        return state
