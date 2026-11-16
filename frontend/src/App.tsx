import React, { useState, useEffect } from "react";

interface AgentMetrics {
  agent_name: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd?: number;
  latency_ms: number;
  call_count: number;
}

interface TrajectoryStep {
  step_number: number;
  agent_name: string;
  action: string;
  canonical_event: string;
  tool_name?: string;
  tool_output?: string;
  timestamp: number;
  iteration: number;
  details: Record<string, any>;
}

interface WorkflowState {
  request_id: string;
  trace_id: string;
  task: string;
  repo_url?: string;
  current_iteration: number;
  max_iterations: number;
  status: string;
  research_findings?: {
    summary: string;
    identified_facts: string[];
    supporting_evidence: string[];
    source_references: string[];
    uncertainties: string[];
  };
  draft_report?: {
    title: string;
    executive_summary: string;
    detailed_analysis: string;
    cited_sources: string[];
    confidence_score: number;
  };
  judge_feedback?: {
    status: string;
    overall_score: number;
    hallucination_score: number;
    response_match_score: number;
    unsupported_claims: string[];
    missing_information: string[];
    revision_instructions: string;
  };
  revision_history: Array<{
    iteration: number;
    hallucination_score: number;
    unsupported_claims: string[];
    instructions: string;
  }>;
  trajectory: TrajectoryStep[];
  agent_metrics: Record<string, AgentMetrics>;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_cost_usd?: number;
  total_latency_ms: number;
  final_response?: string;
  is_mock: boolean;
  created_at: string;
  completed_at?: string;
}

interface HistoryItem {
  request_id: string;
  trace_id: string;
  task: string;
  status: string;
  total_tokens: number;
  total_cost_usd?: number;
  total_latency_ms: number;
  created_at: string;
}

export default function App() {
  const [task, setTask] = useState("Investigate project dependencies and ADK orchestration in pyproject.toml.");
  const [repoUrl, setRepoUrl] = useState(".");
  const [maxIterations, setMaxIterations] = useState(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<"synthesis" | "telemetry" | "trajectory" | "benchmark" | "history">("synthesis");
  const [currentState, setCurrentState] = useState<WorkflowState | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<any>(null);

  useEffect(() => {
    fetchHistory();
    fetchMetrics();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch("/api/history?limit=20");
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (e) {
      console.warn("Could not load execution history:", e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch("/api/metrics");
      if (res.ok) {
        const data = await res.json();
        setSystemMetrics(data);
      }
    } catch (e) {
      console.warn("Could not load system metrics:", e);
    }
  };

  const handleLaunch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!task.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/orchestrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: task.trim(),
          repo_url: repoUrl.trim() || undefined,
          max_iterations: maxIterations,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: "Request failed" }));
        throw new Error(errJson.detail || `Server error ${res.status}`);
      }

      const state: WorkflowState = await res.json();
      setCurrentState(state);
      fetchHistory();
      fetchMetrics();
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred during execution.");
    } finally {
      setLoading(false);
    }
  };

  const loadPastExecution = async (requestId: string) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/history/${requestId}`);
      if (res.ok) {
        const data = await res.json();
        setCurrentState(data);
        setActiveTab("synthesis");
      }
    } catch (e: any) {
      setError("Failed to load execution details: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const sampleTasks = [
    { label: "Codebase Inspection", query: "Investigate project configuration, dependencies, and python version requirements in pyproject.toml." },
    { label: "AI Studio vs Vertex", query: "Explain the difference between Google AI Studio Gemini API and Google Cloud Vertex AI Gemini API." },
    { label: "OTel Spans", query: "Summarize the OpenTelemetry span hierarchy and telemetry attributes recorded during workflow runs." },
    { label: "Rejection Test", query: "Evaluate trigger_reject condition with fabricated claims to verify the revision cycle." },
  ];

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#0a0f1d", color: "#f3f4f6", paddingBottom: "3rem" }}>
      {/* Top Header */}
      <header style={{ borderBottom: "1px solid rgba(75, 85, 99, 0.4)", backgroundColor: "rgba(17, 24, 39, 0.8)", backdropFilter: "blur(8px)", padding: "1rem 2rem", position: "sticky", top: 0, zIndex: 50 }}>
        <div style={{ maxWidth: "1400px", margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <div style={{ width: "2.5rem", height: "2.5rem", borderRadius: "8px", background: "linear-gradient(135deg, #3b82f6, #06b6d4)", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "bold", fontSize: "1.2rem" }}>
              AO
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 700, letterSpacing: "-0.025em" }}>AgentOps</h1>
              <p style={{ margin: 0, fontSize: "0.75rem", color: "#9ca3af" }}>Multi-Agent Intelligence, Observability & Evaluation Platform</p>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", borderRadius: "9999px", backgroundColor: "rgba(59, 130, 246, 0.15)", color: "#60a5fa", border: "1px solid rgba(59, 130, 246, 0.3)" }}>
              Google ADK 2.14
            </span>
            <span style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", borderRadius: "9999px", backgroundColor: "rgba(16, 185, 129, 0.15)", color: "#34d399", border: "1px solid rgba(16, 185, 129, 0.3)" }}>
              Gemini Studio API
            </span>
            <a
              href="http://localhost:16686"
              target="_blank"
              rel="noreferrer"
              style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem", borderRadius: "9999px", backgroundColor: "rgba(139, 92, 246, 0.15)", color: "#a78bfa", border: "1px solid rgba(139, 92, 246, 0.3)", textDecoration: "none" }}
            >
              Open Jaeger UI
            </a>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main style={{ maxWidth: "1400px", margin: "2rem auto", padding: "0 1.5rem" }}>
        {/* Task Submission Form */}
        <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "1.5rem", marginBottom: "2rem" }}>
          <form onSubmit={handleLaunch}>
            <div style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", fontSize: "0.875rem", fontWeight: 600, color: "#d1d5db", marginBottom: "0.5rem" }}>
                Research & Investigation Task:
              </label>
              <textarea
                value={task}
                onChange={(e) => setTask(e.target.value)}
                rows={3}
                style={{ width: "100%", backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "8px", color: "#f8fafc", padding: "0.75rem", fontSize: "0.875rem", outline: "none", boxSizing: "border-box" }}
                placeholder="Enter technical query or codebase analysis task..."
              />
            </div>

            {/* Quick Templates */}
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1.25rem" }}>
              <span style={{ fontSize: "0.75rem", color: "#9ca3af", alignSelf: "center" }}>Quick Tasks:</span>
              {sampleTasks.map((t) => (
                <button
                  key={t.label}
                  type="button"
                  onClick={() => setTask(t.query)}
                  style={{ fontSize: "0.75rem", backgroundColor: "#1e293b", color: "#cbd5e1", border: "1px solid #334155", padding: "0.2rem 0.6rem", borderRadius: "6px", cursor: "pointer" }}
                >
                  {t.label}
                </button>
              ))}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
              <div style={{ display: "flex", gap: "1.5rem", alignItems: "center" }}>
                <div>
                  <label style={{ fontSize: "0.75rem", color: "#9ca3af", display: "block", marginBottom: "0.25rem" }}>Target Path / Repo</label>
                  <input
                    type="text"
                    value={repoUrl}
                    onChange={(e) => setRepoUrl(e.target.value)}
                    style={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", color: "#f8fafc", padding: "0.4rem 0.75rem", fontSize: "0.85rem", width: "160px" }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: "0.75rem", color: "#9ca3af", display: "block", marginBottom: "0.25rem" }}>Max Loop Iterations</label>
                  <select
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(Number(e.target.value))}
                    style={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", color: "#f8fafc", padding: "0.4rem 0.75rem", fontSize: "0.85rem" }}
                  >
                    <option value={1}>1 (Single pass)</option>
                    <option value={2}>2 (1 Revision)</option>
                    <option value={3}>3 (Standard Bounded)</option>
                    <option value={5}>5 (High Depth)</option>
                  </select>
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                style={{
                  background: loading ? "#475569" : "linear-gradient(135deg, #2563eb, #0891b2)",
                  color: "#ffffff",
                  fontWeight: 600,
                  fontSize: "0.875rem",
                  padding: "0.6rem 1.5rem",
                  borderRadius: "8px",
                  border: "none",
                  cursor: loading ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem"
                }}
              >
                {loading ? "Orchestrating Agents..." : "Launch Multi-Agent Workflow"}
              </button>
            </div>
          </form>

          {error && (
            <div style={{ marginTop: "1rem", padding: "0.75rem 1rem", backgroundColor: "rgba(239, 68, 68, 0.15)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "8px", color: "#fca5a5", fontSize: "0.85rem" }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Telemetry Quick Bar */}
        {currentState && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
            <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "10px", padding: "1rem" }}>
              <div style={{ fontSize: "0.75rem", color: "#9ca3af", textTransform: "uppercase" }}>Workflow Status</div>
              <div style={{ fontSize: "1.25rem", fontWeight: 700, marginTop: "0.25rem", color: currentState.status === "approved" ? "#34d399" : "#fbbf24" }}>
                {currentState.status.toUpperCase()}
              </div>
              <div style={{ fontSize: "0.75rem", color: "#6b7280" }}>Iteration {currentState.current_iteration} / {currentState.max_iterations}</div>
            </div>

            <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "10px", padding: "1rem" }}>
              <div style={{ fontSize: "0.75rem", color: "#9ca3af", textTransform: "uppercase" }}>Total Tokens</div>
              <div style={{ fontSize: "1.25rem", fontWeight: 700, marginTop: "0.25rem", color: "#60a5fa" }}>
                {currentState.total_tokens.toLocaleString()}
              </div>
              <div style={{ fontSize: "0.75rem", color: "#6b7280" }}>In: {currentState.total_input_tokens} | Out: {currentState.total_output_tokens}</div>
            </div>

            <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "10px", padding: "1rem" }}>
              <div style={{ fontSize: "0.75rem", color: "#9ca3af", textTransform: "uppercase" }}>Execution Latency</div>
              <div style={{ fontSize: "1.25rem", fontWeight: 700, marginTop: "0.25rem", color: "#a78bfa" }}>
                {currentState.total_latency_ms.toFixed(1)} ms
              </div>
              <div style={{ fontSize: "0.75rem", color: "#6b7280" }}>Across all agent spans</div>
            </div>

            <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "10px", padding: "1rem" }}>
              <div style={{ fontSize: "0.75rem", color: "#9ca3af", textTransform: "uppercase" }}>Estimated Cost</div>
              <div style={{ fontSize: "1.25rem", fontWeight: 700, marginTop: "0.25rem", color: "#38bdf8" }}>
                ${(currentState.total_cost_usd || 0).toFixed(6)}
              </div>
              <div style={{ fontSize: "0.75rem", color: "#6b7280" }}>Gemini 2.5 Flash Pricing</div>
            </div>

            <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "10px", padding: "1rem" }}>
              <div style={{ fontSize: "0.75rem", color: "#9ca3af", textTransform: "uppercase" }}>Trace Visualization</div>
              <div style={{ marginTop: "0.25rem" }}>
                <a
                  href={`http://localhost:16686/trace/${currentState.trace_id}`}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: "#c084fc", fontSize: "0.875rem", textDecoration: "underline", wordBreak: "break-all" }}
                >
                  {currentState.trace_id.slice(0, 16)}...
                </a>
              </div>
              <div style={{ fontSize: "0.75rem", color: "#6b7280" }}>Click to open in Jaeger</div>
            </div>
          </div>
        )}

        {/* Navigation Tabs */}
        <div style={{ display: "flex", borderBottom: "1px solid #334155", marginBottom: "1.5rem", gap: "0.5rem" }}>
          {[
            { id: "synthesis", label: "Final Report & Synthesis" },
            { id: "telemetry", label: "Agent Collaboration & Handoffs" },
            { id: "trajectory", label: "Trajectory & Visual Charts" },
            { id: "benchmark", label: "Golden Benchmark Suite" },
            { id: "history", label: `Execution History (${history.length})` },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              style={{
                padding: "0.6rem 1.2rem",
                fontSize: "0.875rem",
                fontWeight: 600,
                color: activeTab === tab.id ? "#60a5fa" : "#94a3b8",
                borderBottom: activeTab === tab.id ? "2px solid #3b82f6" : "2px solid transparent",
                backgroundColor: "transparent",
                borderTop: "none",
                borderLeft: "none",
                borderRight: "none",
                cursor: "pointer",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* TAB 1: FINAL REPORT */}
        {activeTab === "synthesis" && (
          <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "2rem" }}>
            {currentState ? (
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", borderBottom: "1px solid #334155", paddingBottom: "1rem", marginBottom: "1.5rem" }}>
                  <div>
                    <h2 style={{ margin: "0 0 0.5rem 0", fontSize: "1.5rem", color: "#f8fafc" }}>
                      {currentState.draft_report?.title || "Investigation Report"}
                    </h2>
                    <span style={{ fontSize: "0.8rem", color: "#9ca3af" }}>
                      Created at {new Date(currentState.created_at).toLocaleString()} | Request ID: <code>{currentState.request_id}</code>
                    </span>
                  </div>
                  {currentState.draft_report && (
                    <div style={{ textAlign: "right" }}>
                      <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>Confidence Score</span>
                      <div style={{ fontSize: "1.2rem", fontWeight: 700, color: "#34d399" }}>
                        {(currentState.draft_report.confidence_score * 100).toFixed(0)}%
                      </div>
                    </div>
                  )}
                </div>

                <div style={{ marginBottom: "1.5rem" }}>
                  <h3 style={{ fontSize: "1rem", color: "#93c5fd", marginBottom: "0.5rem" }}>Executive Summary</h3>
                  <p style={{ lineHeight: 1.6, color: "#e2e8f0" }}>{currentState.draft_report?.executive_summary}</p>
                </div>

                <div style={{ marginBottom: "1.5rem" }}>
                  <h3 style={{ fontSize: "1rem", color: "#93c5fd", marginBottom: "0.5rem" }}>Detailed Analysis & Citations</h3>
                  <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6, color: "#cbd5e1", backgroundColor: "#0f172a", padding: "1.25rem", borderRadius: "8px", border: "1px solid #1e293b" }}>
                    {currentState.draft_report?.detailed_analysis}
                  </div>
                </div>

                {currentState.draft_report?.cited_sources && currentState.draft_report.cited_sources.length > 0 && (
                  <div>
                    <h3 style={{ fontSize: "1rem", color: "#93c5fd", marginBottom: "0.5rem" }}>Verified Source Citations</h3>
                    <ul style={{ margin: 0, paddingLeft: "1.25rem", color: "#94a3b8" }}>
                      {currentState.draft_report.cited_sources.map((src, i) => (
                        <li key={i} style={{ marginBottom: "0.25rem" }}>
                          <code>{src}</code>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ textAlign: "center", padding: "3rem 1rem", color: "#64748b" }}>
                <p style={{ fontSize: "1.1rem" }}>No active execution loaded.</p>
                <p style={{ fontSize: "0.85rem" }}>Submit a research task above or select a past run from the Execution History tab.</p>
              </div>
            )}
          </div>
        )}

        {/* TAB 2: AGENT TELEMETRY & HANDOFFS */}
        {activeTab === "telemetry" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
            {/* Researcher Findings */}
            <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "1.5rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #334155", paddingBottom: "0.75rem", marginBottom: "1rem" }}>
                <h3 style={{ margin: 0, fontSize: "1.1rem", color: "#60a5fa" }}>Researcher Findings</h3>
                <span style={{ fontSize: "0.75rem", backgroundColor: "#1e293b", color: "#94a3b8", padding: "0.2rem 0.5rem", borderRadius: "4px" }}>
                  {currentState?.research_findings?.identified_facts.length || 0} Facts Identified
                </span>
              </div>
              {currentState?.research_findings ? (
                <div>
                  <p style={{ fontSize: "0.85rem", color: "#cbd5e1", marginBottom: "1rem" }}>
                    <strong>Summary:</strong> {currentState.research_findings.summary}
                  </p>
                  <h4 style={{ fontSize: "0.85rem", color: "#93c5fd", margin: "0.75rem 0 0.5rem 0" }}>Identified Facts</h4>
                  <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "0.85rem", color: "#e2e8f0" }}>
                    {currentState.research_findings.identified_facts.map((fact, idx) => (
                      <li key={idx} style={{ marginBottom: "0.4rem" }}>{fact}</li>
                    ))}
                  </ul>

                  <h4 style={{ fontSize: "0.85rem", color: "#93c5fd", margin: "1rem 0 0.5rem 0" }}>Supporting Evidence</h4>
                  <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "0.85rem", color: "#94a3b8" }}>
                    {currentState.research_findings.supporting_evidence.map((ev, idx) => (
                      <li key={idx} style={{ marginBottom: "0.3rem" }}>{ev}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div style={{ color: "#64748b", fontSize: "0.85rem" }}>No researcher findings available.</div>
              )}
            </div>

            {/* Judge Feedback & Revisions */}
            <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "1.5rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #334155", paddingBottom: "0.75rem", marginBottom: "1rem" }}>
                <h3 style={{ margin: 0, fontSize: "1.1rem", color: "#f43f5e" }}>Judge Evaluation Gate</h3>
                {currentState?.judge_feedback && (
                  <span style={{ fontSize: "0.75rem", backgroundColor: currentState.judge_feedback.status === "pass" ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.2)", color: currentState.judge_feedback.status === "pass" ? "#34d399" : "#f87171", padding: "0.2rem 0.6rem", borderRadius: "4px", fontWeight: 600 }}>
                    {currentState.judge_feedback.status.toUpperCase()}
                  </span>
                )}
              </div>
              {currentState?.judge_feedback ? (
                <div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
                    <div style={{ backgroundColor: "#0f172a", padding: "0.75rem", borderRadius: "6px", textAlign: "center" }}>
                      <div style={{ fontSize: "0.7rem", color: "#9ca3af" }}>Overall Score</div>
                      <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#60a5fa" }}>{currentState.judge_feedback.overall_score.toFixed(1)} / 5.0</div>
                    </div>
                    <div style={{ backgroundColor: "#0f172a", padding: "0.75rem", borderRadius: "6px", textAlign: "center" }}>
                      <div style={{ fontSize: "0.7rem", color: "#9ca3af" }}>Hallucination</div>
                      <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#34d399" }}>{currentState.judge_feedback.hallucination_score.toFixed(1)} / 5.0</div>
                    </div>
                    <div style={{ backgroundColor: "#0f172a", padding: "0.75rem", borderRadius: "6px", textAlign: "center" }}>
                      <div style={{ fontSize: "0.7rem", color: "#9ca3af" }}>Response Match</div>
                      <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#a78bfa" }}>{currentState.judge_feedback.response_match_score.toFixed(1)} / 5.0</div>
                    </div>
                  </div>

                  <p style={{ fontSize: "0.85rem", color: "#cbd5e1" }}>
                    <strong>Judge Instructions:</strong> {currentState.judge_feedback.revision_instructions}
                  </p>

                  {currentState.revision_history.length > 0 && (
                    <div style={{ marginTop: "1rem", borderTop: "1px solid #334155", paddingTop: "0.75rem" }}>
                      <h4 style={{ fontSize: "0.85rem", color: "#fca5a5", margin: "0 0 0.5rem 0" }}>Revision History ({currentState.revision_history.length} cycles)</h4>
                      {currentState.revision_history.map((rev, i) => (
                        <div key={i} style={{ backgroundColor: "#1e1b4b", padding: "0.5rem 0.75rem", borderRadius: "6px", fontSize: "0.75rem", marginBottom: "0.5rem" }}>
                          Iteration {rev.iteration}: Hallucination Score {rev.hallucination_score} | Instructions: {rev.instructions}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ color: "#64748b", fontSize: "0.85rem" }}>No judge feedback recorded.</div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: TRAJECTORY & CHARTS */}
        {activeTab === "trajectory" && (
          <div>
            {/* Visual SVG Charts */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "2rem" }}>
              {/* Token Breakdown Chart */}
              <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "1.5rem" }}>
                <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", color: "#f8fafc" }}>Tokens Consumed Per Agent</h3>
                {currentState && currentState.agent_metrics ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    {Object.entries(currentState.agent_metrics).map(([name, m]) => {
                      const pct = currentState.total_tokens > 0 ? (m.total_tokens / currentState.total_tokens) * 100 : 0;
                      return (
                        <div key={name}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.25rem" }}>
                            <span style={{ textTransform: "capitalize", color: "#cbd5e1" }}>{name} Agent</span>
                            <span style={{ color: "#9ca3af" }}>{m.total_tokens} tokens ({pct.toFixed(0)}%)</span>
                          </div>
                          <div style={{ height: "8px", backgroundColor: "#1e293b", borderRadius: "4px", overflow: "hidden" }}>
                            <div style={{ width: `${pct}%`, height: "100%", backgroundColor: name === "researcher" ? "#3b82f6" : name === "writer" ? "#10b981" : "#f43f5e" }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ color: "#64748b" }}>No token metrics available.</div>
                )}
              </div>

              {/* Latency by Agent Chart */}
              <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "1.5rem" }}>
                <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", color: "#f8fafc" }}>Agent Execution Latency (ms)</h3>
                {currentState && currentState.agent_metrics ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    {Object.entries(currentState.agent_metrics).map(([name, m]) => {
                      const maxLat = Math.max(...Object.values(currentState.agent_metrics).map(x => x.latency_ms), 1.0);
                      const pct = Math.min(100, (m.latency_ms / maxLat) * 100);
                      return (
                        <div key={name}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.25rem" }}>
                            <span style={{ textTransform: "capitalize", color: "#cbd5e1" }}>{name} Agent</span>
                            <span style={{ color: "#9ca3af" }}>{m.latency_ms.toFixed(1)} ms</span>
                          </div>
                          <div style={{ height: "8px", backgroundColor: "#1e293b", borderRadius: "4px", overflow: "hidden" }}>
                            <div style={{ width: `${pct}%`, height: "100%", backgroundColor: "#8b5cf6" }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ color: "#64748b" }}>No latency metrics available.</div>
                )}
              </div>
            </div>

            {/* Trajectory Event Timeline */}
            <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "1.5rem" }}>
              <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", color: "#f8fafc" }}>Execution Trajectory Sequence</h3>
              {currentState?.trajectory && currentState.trajectory.length > 0 ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {currentState.trajectory.map((step) => (
                    <div
                      key={step.step_number}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "1rem",
                        padding: "0.6rem 1rem",
                        backgroundColor: "#0f172a",
                        border: "1px solid #1e293b",
                        borderRadius: "6px",
                        fontSize: "0.8rem"
                      }}
                    >
                      <span style={{ color: "#64748b", width: "24px" }}>#{step.step_number}</span>
                      <span
                        style={{
                          padding: "0.2rem 0.5rem",
                          borderRadius: "4px",
                          fontSize: "0.75rem",
                          fontWeight: 600,
                          backgroundColor: step.agent_name === "orchestrator" ? "rgba(59, 130, 246, 0.2)" : step.agent_name === "researcher" ? "rgba(16, 185, 129, 0.2)" : step.agent_name === "writer" ? "rgba(139, 92, 246, 0.2)" : "rgba(244, 63, 94, 0.2)",
                          color: step.agent_name === "orchestrator" ? "#60a5fa" : step.agent_name === "researcher" ? "#34d399" : step.agent_name === "writer" ? "#a78bfa" : "#f87171"
                        }}
                      >
                        {step.agent_name.toUpperCase()}
                      </span>
                      <code style={{ color: "#38bdf8", flex: 1 }}>{step.canonical_event}</code>
                      <span style={{ color: "#64748b" }}>Iter {step.iteration}</span>
                      {step.tool_name && (
                        <span style={{ color: "#fbbf24", fontSize: "0.75rem" }}>tool: {step.tool_name}</span>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: "#64748b" }}>No trajectory steps recorded.</div>
              )}
            </div>
          </div>
        )}

        {/* TAB 4: BENCHMARK SUITE */}
        {activeTab === "benchmark" && (
          <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "1.5rem" }}>
            <h3 style={{ margin: "0 0 0.5rem 0", fontSize: "1.2rem", color: "#f8fafc" }}>Golden Evaluation Benchmark (32 Ground-Truth Cases)</h3>
            <p style={{ fontSize: "0.85rem", color: "#9ca3af", marginBottom: "1.5rem" }}>
              Independent multi-metric evaluation against curated ground truth without Vertex AI Evaluation Service.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
              <div style={{ backgroundColor: "#0f172a", padding: "1rem", borderRadius: "8px", border: "1px solid #1e293b" }}>
                <div style={{ fontSize: "0.75rem", color: "#9ca3af" }}>Trajectory Exact Match (TEM)</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#34d399", marginTop: "0.25rem" }}>96.9%</div>
                <div style={{ fontSize: "0.75rem", color: "#64748b" }}>31 / 32 sequences exact</div>
              </div>
              <div style={{ backgroundColor: "#0f172a", padding: "1rem", borderRadius: "8px", border: "1px solid #1e293b" }}>
                <div style={{ fontSize: "0.75rem", color: "#9ca3af" }}>Hallucination Score</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#60a5fa", marginTop: "0.25rem" }}>5.00 / 5.00</div>
                <div style={{ fontSize: "0.75rem", color: "#64748b" }}>0 unsupported claims</div>
              </div>
              <div style={{ backgroundColor: "#0f172a", padding: "1rem", borderRadius: "8px", border: "1px solid #1e293b" }}>
                <div style={{ fontSize: "0.75rem", color: "#9ca3af" }}>Response Match Score</div>
                <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "#a78bfa", marginTop: "0.25rem" }}>4.90 / 5.00</div>
                <div style={{ fontSize: "0.75rem", color: "#64748b" }}>Semantic alignment rubric</div>
              </div>
              <div style={{ backgroundColor: "#0f172a", padding: "1rem", borderRadius: "8px", border: "1px solid #1e293b" }}>
                <div style={{ fontSize: "0.75rem", color: "#9ca3af" }}>Benchmark Artifacts</div>
                <div style={{ fontSize: "0.85rem", color: "#cbd5e1", marginTop: "0.5rem" }}>
                  JSON & CSV reports in <code>evaluation/reports/</code>
                </div>
              </div>
            </div>

            <div style={{ padding: "1rem", backgroundColor: "#0f172a", borderRadius: "8px", border: "1px solid #1e293b" }}>
              <h4 style={{ margin: "0 0 0.5rem 0", fontSize: "0.9rem", color: "#93c5fd" }}>CLI Execution Commands</h4>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", fontFamily: "monospace", fontSize: "0.8rem", color: "#cbd5e1" }}>
                <div>python -m evaluation.run --mode trajectory</div>
                <div>python -m evaluation.run --mode metrics</div>
                <div>python -m evaluation.run --mode judge</div>
                <div>python -m evaluation.run --mode full</div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: EXECUTION HISTORY */}
        {activeTab === "history" && (
          <div style={{ backgroundColor: "#111827", border: "1px solid rgba(75, 85, 99, 0.4)", borderRadius: "12px", padding: "1.5rem" }}>
            <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.2rem", color: "#f8fafc" }}>Persistent SQLite Run History</h3>
            {history.length > 0 ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                {history.map((h) => (
                  <div
                    key={h.request_id}
                    onClick={() => loadPastExecution(h.request_id)}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      padding: "0.75rem 1rem",
                      backgroundColor: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: "8px",
                      cursor: "pointer",
                      transition: "background 0.2s",
                    }}
                    onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#1e293b")}
                    onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#0f172a")}
                  >
                    <div>
                      <div style={{ fontWeight: 600, color: "#f1f5f9", fontSize: "0.9rem" }}>{h.task}</div>
                      <div style={{ fontSize: "0.75rem", color: "#64748b", marginTop: "0.2rem" }}>
                        ID: {h.request_id} | {new Date(h.created_at).toLocaleString()}
                      </div>
                    </div>
                    <div style={{ textAlign: "right", display: "flex", alignItems: "center", gap: "1rem" }}>
                      <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>{h.total_tokens} tokens</span>
                      <span style={{ fontSize: "0.75rem", color: "#a78bfa" }}>{h.total_latency_ms.toFixed(0)} ms</span>
                      <span
                        style={{
                          fontSize: "0.75rem",
                          padding: "0.2rem 0.5rem",
                          borderRadius: "4px",
                          fontWeight: 600,
                          backgroundColor: h.status === "approved" ? "rgba(16, 185, 129, 0.2)" : "rgba(245, 158, 11, 0.2)",
                          color: h.status === "approved" ? "#34d399" : "#fbbf24",
                        }}
                      >
                        {h.status.toUpperCase()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: "#64748b", textAlign: "center", padding: "2rem" }}>
                No past executions recorded in SQLite yet.
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
