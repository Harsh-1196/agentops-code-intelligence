# AgentOps Architecture & System Design

AgentOps is an end-to-end multi-agent research, evaluation, and observability platform designed for deep codebase and technical inquiry. It orchestrates collaborating agents using the Google Agent Development Kit (ADK), queries the Gemini Developer API through Google AI Studio (requiring zero Vertex AI credentials or cloud billing), traces every step with OpenTelemetry and Jaeger, and runs a comprehensive independent evaluation harness (Golden Dataset, Trajectory Exact-Match, Precision/Recall/F1, and LLM-as-a-Judge for Hallucination & Response Match).

---

## 1. System Topology

```
+---------------------------------------------------------------------------------+
|                                 USER / CLIENT                                   |
|                  React TypeScript Web UI  |  REST API Client                    |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                          FASTAPI APPLICATION LAYER                              |
|           /api/orchestrate | /api/evaluate | /api/history | /api/metrics        |
+---------------------------------------------------------------------------------+
                                       |
           +---------------------------+---------------------------+
           |                                                       |
           v                                                       v
+--------------------------------------+   +--------------------------------------+
|       GOOGLE ADK ORCHESTRATOR        |   |       OPENTELEMETRY TRACER           |
|                                      |   |                                      |
|  +--------------------------------+  |   |  * Span: Request                     |
|  | Request Classifier             |  |   |    * Span: Orchestrator              |
|  +--------------------------------+  |   |      * Span: Researcher              |
|                 |                    |   |        * Span: Research Tools        |
|                 v                    |   |      * Span: Writer                  |
|  +--------------------------------+  |   |      * Span: Judge                   |
|  | Bounded Refinement Loop        |  |   |        * Span: LLM Inference         |
|  |  (LoopAgent, max_iterations=3) |  |   |      * Span: Revision Cycle (if rej) |
|  |                                |  |   |    * Span: Final Response            |
|  |  [Researcher] -> [Writer]      |  |   |                                      |
|  |        ^             |         |  |   |  * Token Accounting & Model Cost     |
|  |        |             v         |  |   |  * Latency Metrics (P50/P95)         |
|  |   (Reject) <----+ [Judge]      |  |   |  * Export to OTLP / Jaeger           |
|  |                 |              |  |   +--------------------------------------+
|  |                 v (Approve)    |  |                           |
|  |           [Escalation]         |  |                           v
|  +--------------------------------+  |              +--------------------------+
|                 |                    |              |   Jaeger All-In-One UI   |
|                 v                    |              |     localhost:16686      |
|         Approved Final Output        |              +--------------------------+
+--------------------------------------+
           |
           v
+---------------------------------------------------------------------------------+
|                         INDEPENDENT EVALUATION HARNESS                          |
|                                                                                 |
|  1. Golden Dataset: 30+ verified multi-category benchmark cases                 |
|  2. Trajectory Exact Match: Sequence alignment and canonical state diffing      |
|  3. Classification Metrics: Finding-level Precision, Recall, and F1             |
|  4. LLM-as-a-Judge: Hallucination detection & Response Match (Gemini Studio)    |
|  5. Automated reporting: Markdown, JSON, and CSV exports                        |
+---------------------------------------------------------------------------------+
           |
           v
+---------------------------------------------------------------------------------+
|                        PERSISTENCE & STORAGE (SQLite)                           |
|         Requests, Trajectories, Spans, Metrics, and Benchmark Results          |
+---------------------------------------------------------------------------------+
```

---

## 2. Multi-Agent Team Responsibilities

### A. Researcher Agent
- **Goal**: Gathers factual, verified evidence regarding the target research or repository topic.
- **Tools**:
  - `inspect_codebase`: Analyzes repository structure, file content, and documentation.
  - `web_knowledge_retriever`: Retrieves factual background context and documentation.
- **Output Schema**: Structured `ResearchFindings` containing:
  - `summary`: High-level topic synthesis.
  - `identified_facts`: List of discrete, atomic factual statements.
  - `supporting_evidence`: List of specific citations, excerpts, and tool outputs.
  - `source_references`: Document names, URLs, or repository file paths.
  - `uncertainties`: Known unknowns or unverified assumptions.
  - `unanswered_questions`: Items requiring clarification.

### B. Writer Agent
- **Goal**: Synthesizes verified research into a coherent, structured report.
- **Constraints**:
  - Must strictly ground every claim in `ResearchFindings`.
  - Must not hallucinate or extrapolate beyond provided facts.
  - Explicitly separates verified findings from unverified possibilities.
- **Output Schema**: Structured `DraftReport` containing:
  - `title`: Report headline.
  - `executive_summary`: Concise summary.
  - `detailed_analysis`: Deep dive with inline citations `[Source X]`.
  - `cited_sources`: References strictly mapped to research findings.
  - `confidence_score`: Self-assessed grounding confidence (0.0 to 1.0).

### C. Judge Agent
- **Goal**: Acts as an independent fact-checker and quality gate before release.
- **Evaluation Dimensions**:
  - `factual_accuracy`: Are all claims grounded in evidence?
  - `hallucinated_claims`: Identifies any statements without supporting evidence.
  - `completeness`: Did the writer answer the original user prompt completely?
  - `relevance`: Did the writer avoid tangential or irrelevant filler?
- **Output Schema**: Structured `JudgeFeedback` containing:
  - `status`: `"pass"` (Approve) or `"fail"` (Reject).
  - `overall_score`: 0 to 5.
  - `hallucination_score`: 0 (severe hallucination) to 5 (flawless grounding).
  - `response_match_score`: 0 to 5.
  - `unsupported_claims`: List of flagged sentences with explanations.
  - `missing_information`: Critical aspects omitted from the draft.
  - `revision_instructions`: Concrete, actionable feedback for the next iteration.

### D. Orchestrator Agent
- **Goal**: Coordinates the multi-agent lifecycle with deterministic state management.
- **Loop Strategy**:
  - Powered by Google ADK `LoopAgent` or bounded asynchronous iteration graph.
  - Hard upper bound: `max_iterations = 3`.
  - Termination condition: `status == "pass"` OR `iteration >= max_iterations`.
  - Stores all revision cycle reasons, iteration counters, and state transitions.

---

## 3. Observability & Telemetry

### Trace Hierarchy
Every request is assigned a `request_id` and OpenTelemetry `trace_id`. The span hierarchy tracks:
1. `request.root`
   - `agent.orchestrator`
     - `iteration_{i}.researcher`
       - `tool.{tool_name}`
       - `llm.gemini_call`
     - `iteration_{i}.writer`
       - `llm.gemini_call`
     - `iteration_{i}.judge`
       - `llm.gemini_call`
     - `iteration_{i}.revision` (recorded if judge rejected)
   - `persistence.save_run`

### Metrics & Cost Accounting
- **Tokens**: Captured directly from Gemini `usage_metadata` (`prompt_token_count`, `candidates_token_count`, `total_token_count`).
- **Cost Formula**:
  $$\text{Cost} = \frac{\text{Input Tokens} \times \text{Input Price} + \text{Output Tokens} \times \text{Output Price}}{1{,}000{,}000}$$
- **Latency**: Sub-millisecond tracking for each agent span, LLM roundtrip, and total workflow, reporting average, P50, and P95.

---

## 4. Evaluation Methodology

1. **Golden Dataset**: 30+ categorized cases with ground-truth facts, expected trajectories, and reference answers.
2. **Trajectory Exact Match**:
   $$\text{TEM} = \frac{\text{Number of Exactly Matched Trajectories}}{\text{Total Evaluated Trajectories}}$$
   Compares normalized canonical event sequences (e.g., `["orchestrator.classify", "researcher.execute", "researcher.tool_call", "writer.generate", "judge.evaluate", "orchestrator.approve"]`).
3. **Classification Metrics**:
   - Finding-level Precision: $\frac{TP}{TP + FP}$
   - Finding-level Recall: $\frac{TP}{TP + FN}$
   - F1 Score: $\frac{2 \times P \times R}{P + R}$
4. **LLM-as-a-Judge**: Evaluates Hallucination and Response Match using documented 0-5 rubrics, structured Pydantic schemas, and external Gemini calls.

---

## 5. Licensing & Attribution

- **Architectural Reference**: Inspired by `vladkol/agent-evaluation-lab` (Apache-2.0 License).
- **Independent Implementation**:
  - Independent ADK multi-agent pipeline replacing Vertex AI dependencies with Google AI Studio Gemini API keys.
  - Completely custom evaluation harness eliminating Vertex AI Evaluation Service and Cloud Run.
  - Custom FastAPI + React + SQLite architecture containerized with Docker Compose and local Jaeger.
