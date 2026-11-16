# AgentOps: Multi-Agent Research, Evaluation and Observability Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Google ADK](https://img.shields.io/badge/Google-ADK-green.svg)](https://github.com/google/agent-development-kit)
[![Gemini API](https://img.shields.io/badge/Gemini-Developer%20API-orange.svg)](https://aistudio.google.com/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Tracing-purple.svg)](https://opentelemetry.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

AgentOps is an enterprise-grade multi-agent research, evaluation, and observability platform built on the **Google Agent Development Kit (ADK)**. It coordinates a team of specialized agents to conduct rigorous research, synthesize structured reports, perform independent verification, and trace every execution with OpenTelemetry and Jaeger—all powered by the **Google AI Studio Gemini Developer API** without requiring Vertex AI credentials, Google Cloud billing, or cloud-only services.

---

## Architecture Overview

```
 [User Request] 
       │
       ▼
 ┌─────────────── Google ADK Orchestrator ────────────────┐
 │                                                         │
 │  ┌──────────────┐     ┌───────────┐     ┌────────────┐  │
 │  │  Researcher  │ ──► │   Writer  │ ──► │   Judge    │  │
 │  │ (Tools/Facts)│     │ (Synthesis│     │(Validation)│  │
 │  └──────────────┘     └───────────┘     └────────────┘  │
 │         ▲                                     │         │
 │         └──────── [Reject / Revise] ──────────┘         │
 │                          │ (Pass / Max Iterations)      │
 │                          ▼                              │
 │                 Approved Final Report                   │
 └─────────────────────────────────────────────────────────┘
       │                              │
       ▼                              ▼
 ┌──────────────────────┐      ┌─────────────────────────┐
 │   OpenTelemetry +    │      │ Independent Evaluation  │
 │  Jaeger Trace Graphs │      │  - Trajectory Match     │
 │  Token & Cost Audits │      │  - Precision/Recall/F1  │
 └──────────────────────┘      │  - LLM-as-a-Judge       │
                               └─────────────────────────┘
```

---

## Key Features

1. **Google ADK Multi-Agent Team**:
   - **Researcher Agent**: Performs tool-assisted repository inspection and factual evidence gathering.
   - **Writer Agent**: Strictly synthesizes research findings into evidence-grounded reports with inline citations.
   - **Judge Agent**: Independently fact-checks drafts, detects hallucinations, and assigns pass/fail scores.
   - **Orchestrator Agent**: Manages bounded revision loops (`max_iterations=3`) with structured handoffs.

2. **Zero-Vertex AI Dependency**:
   - Connects directly to Google AI Studio Gemini API using an API key (`GEMINI_API_KEY`).
   - Runs 100% locally or in standard Docker containers without GCP service accounts.

3. **Custom Evaluation Harness**:
   - **Golden Dataset**: 30+ categorized evaluation cases spanning factual research, code inspection, and adversarial prompts.
   - **Trajectory Exact Match**: Compares actual agent execution graphs against expected execution trajectories.
   - **Precision, Recall & F1**: Quantitative finding-level information retrieval metrics.
   - **LLM-as-a-Judge**: Model-based evaluation of Hallucination (0-5) and Response-Match (0-5).

4. **Deep Observability**:
   - End-to-end OpenTelemetry distributed tracing with Jaeger visualization.
   - Exact token tracking (input, output, per-agent, judge).
   - Configurable model pricing table and cost-per-request accounting.
   - P50, P95, and average latency breakdown.

5. **Modern Full-Stack Experience**:
   - High-performance FastAPI backend with streaming workflow updates.
   - Clean React + TypeScript dashboard.
   - Fully containerized with Docker and Docker Compose.
