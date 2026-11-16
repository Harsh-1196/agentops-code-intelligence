"""Research and inspection tools for agent analysis."""

import os
import re
from typing import List, Dict, Any


def inspect_repository_structure(path: str = ".") -> str:
    """Inspects directory layout, file hierarchy, and key manifest files."""
    if not os.path.exists(path):
        return f"Error: Path '{path}' does not exist."

    entries: List[str] = []
    manifests: Dict[str, str] = {}
    key_manifest_names = ["pyproject.toml", "package.json", "Dockerfile", "docker-compose.yml", "README.md", ".env.example"]

    try:
        for root, dirs, files in os.walk(path):
            # Exclude noisy dirs
            dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache", "dist", "build"]]
            rel_root = os.path.relpath(root, path)
            indent = "  " * (rel_root.count(os.sep) if rel_root != "." else 0)
            dir_name = os.path.basename(root) if rel_root != "." else path
            entries.append(f"{indent}[DIR] {dir_name}/")

            for f in sorted(files):
                sub_indent = indent + "  "
                entries.append(f"{sub_indent}[FILE] {f}")
                if f in key_manifest_names and f not in manifests:
                    f_path = os.path.join(root, f)
                    try:
                        with open(f_path, "r", encoding="utf-8", errors="ignore") as mf:
                            content_sample = mf.read(1000)
                            manifests[f] = content_sample
                    except Exception:
                        pass

        summary_lines = [
            "=== Repository Structure Overview ===",
            "\n".join(entries[:60]),  # cap lines
            "\n=== Detected Key Manifests ==="
        ]
        for name, sample in manifests.items():
            summary_lines.append(f"\n--- {name} (excerpt) ---\n{sample[:400]}")

        return "\n".join(summary_lines)
    except Exception as e:
        return f"Error inspecting repository structure: {e}"


def read_source_file(file_path: str, start_line: int = 1, end_line: int = 60) -> str:
    """Reads specific line ranges from a source code or documentation file."""
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        total = len(lines)
        s = max(1, start_line)
        e = min(total, end_line)
        selected = lines[s - 1:e]

        numbered = [f"{i}: {line.rstrip()}" for i, line in enumerate(selected, start=s)]
        return f"=== File: {file_path} (Lines {s}-{e} of {total}) ===\n" + "\n".join(numbered)
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"


def search_codebase(query: str, root_path: str = ".") -> str:
    """Searches for keyword or pattern across text files in the project."""
    if not os.path.exists(root_path):
        return f"Error: Path '{root_path}' does not exist."

    matches: List[str] = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    try:
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "__pycache__", ".venv", "dist", "build"]]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".py", ".ts", ".tsx", ".js", ".json", ".md", ".toml", ".yml", ".yaml", ".txt"]:
                    f_path = os.path.join(root, f)
                    try:
                        with open(f_path, "r", encoding="utf-8", errors="ignore") as fh:
                            for idx, line in enumerate(fh, start=1):
                                if pattern.search(line):
                                    rel = os.path.relpath(f_path, root_path)
                                    matches.append(f"{rel}#L{idx}: {line.strip()[:140]}")
                                    if len(matches) >= 25:
                                        break
                    except Exception:
                        continue
            if len(matches) >= 25:
                break

        if not matches:
            return f"No occurrences of '{query}' found in codebase."
        return f"=== Search Results for '{query}' ({len(matches)} matches) ===\n" + "\n".join(matches)
    except Exception as e:
        return f"Error searching codebase: {e}"


def retrieve_web_documentation(topic: str) -> str:
    """Retrieves standard documentation and specifications for AI, ADK, and cloud architectures."""
    knowledge_base = {
        "google_adk": (
            "Google Agent Development Kit (ADK) is an open-source framework for building multi-agent systems. "
            "It provides abstractions such as Agent (LlmAgent), LoopAgent, and SequentialAgent, "
            "supporting structured event handling, session state persistence, and inter-agent coordination."
        ),
        "gemini_api": (
            "Google AI Studio Gemini Developer API provides direct access to Gemini models (e.g. gemini-2.5-flash) "
            "using standard API keys (GEMINI_API_KEY). It eliminates dependencies on Google Cloud projects, "
            "Vertex AI credentials, and Cloud Run billing."
        ),
        "opentelemetry": (
            "OpenTelemetry provides vendor-neutral instrumentation for distributed tracing and metrics. "
            "OTLP exporters forward traces to local Jaeger collectors (port 4318 for HTTP, 4317 for gRPC) "
            "allowing visualization of hierarchical multi-agent spans and latency breakdowns."
        ),
        "evaluation": (
            "Agent evaluation involves Trajectory Exact Match (comparing canonical event sequences), "
            "Classification Metrics (finding-level Precision, Recall, and F1), and LLM-as-a-Judge for "
            "independent Hallucination and Response Match scoring on 0-5 rubrics."
        )
    }

    t_clean = topic.lower().replace(" ", "_").replace("-", "_")
    for k, v in knowledge_base.items():
        if k in t_clean or any(word in t_clean for word in k.split("_")):
            return f"=== Documentation for {topic} ===\n{v}"

    return (
        f"=== Technical Knowledge Reference: {topic} ===\n"
        f"Verified standard practices for {topic}: ensure explicit source citations, strictly "
        f"ground claims in empirical evidence, and maintain deterministic validation."
    )
