"""Gemini API Provider using Google AI Studio Developer API.

Zero Vertex AI credentials required. Directly utilizes Google AI Studio API keys.
Includes deterministic structured mock fallback for local testing without incurring quota/costs.
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

from agentops.config import Settings, get_settings

logger = logging.getLogger("agentops.provider")


class GenerationResult(BaseModel):
    """Result of an LLM generation call with token and latency telemetry."""
    text: str
    parsed_json: Optional[Dict[str, Any]] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: Optional[float] = None
    model: str
    is_mock: bool = False
    finish_reason: str = "STOP"


class GeminiProvider:
    """Manages Gemini Developer API inference and deterministic mock fallbacks."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.gemini_api_key
        self.default_model = self.settings.gemini_model
        self.client = None

        if self.api_key and self.api_key.strip():
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key.strip())
                logger.info(f"GeminiProvider initialized with live Google AI Studio client (model: {self.default_model}).")
            except Exception as e:
                logger.warning(f"Failed to initialize live Gemini client: {e}. Falling back to mock if allowed.")
                self.client = None
        else:
            logger.info("No GEMINI_API_KEY detected. Running in mock/fixture mode.")

    @property
    def is_live(self) -> bool:
        """True if live Gemini API client is configured and authenticated."""
        return self.client is not None

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        agent_role: str = "general",
    ) -> GenerationResult:
        """
        Generate completion using live Gemini Developer API or deterministic mock fallback.
        """
        target_model = model or self.default_model
        temp = temperature if temperature is not None else self.settings.gemini_temperature
        start_time = time.perf_counter()

        if self.client:
            return await self._generate_live(
                prompt=prompt,
                system_instruction=system_instruction,
                model=target_model,
                temperature=temp,
                response_schema=response_schema,
                agent_role=agent_role,
                start_time=start_time,
            )

        if not self.settings.allow_mock_fallback:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured and ALLOW_MOCK_FALLBACK is False. "
                "Configure GEMINI_API_KEY in .env to enable live inference."
            )

        return self._generate_mock(
            prompt=prompt,
            system_instruction=system_instruction,
            model=target_model,
            response_schema=response_schema,
            agent_role=agent_role,
            start_time=start_time,
        )

    async def _generate_live(
        self,
        prompt: str,
        system_instruction: Optional[str],
        model: str,
        temperature: float,
        response_schema: Optional[Type[BaseModel]],
        agent_role: str,
        start_time: float,
    ) -> GenerationResult:
        """Call official Google GenAI Developer API asynchronously."""
        from google.genai import types

        config_dict: Dict[str, Any] = {
            "temperature": temperature,
        }
        if system_instruction:
            config_dict["system_instruction"] = system_instruction
        if response_schema:
            config_dict["response_mime_type"] = "application/json"
            config_dict["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_dict)

        try:
            # google-genai supports client.aio.models.generate_content
            response = await self.client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            text = response.text or ""

            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or (input_tokens + output_tokens)

            cost_usd = self.settings.calculate_cost(input_tokens, output_tokens, model)

            parsed_json = None
            if response_schema:
                try:
                    parsed_json = json.loads(text)
                except Exception as parse_err:
                    logger.warning(f"Failed to parse live JSON response for schema {response_schema.__name__}: {parse_err}")

            return GenerationResult(
                text=text,
                parsed_json=parsed_json,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
                cost_usd=cost_usd,
                model=model,
                is_mock=False,
                finish_reason="STOP",
            )
        except Exception as e:
            logger.error(f"Error during live Gemini API call: {e}")
            if self.settings.allow_mock_fallback:
                logger.warning("Failing over to deterministic mock provider following API error.")
                return self._generate_mock(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model=model,
                    response_schema=response_schema,
                    agent_role=agent_role,
                    start_time=start_time,
                )
            raise

    def _generate_mock(
        self,
        prompt: str,
        system_instruction: Optional[str],
        model: str,
        response_schema: Optional[Type[BaseModel]],
        agent_role: str,
        start_time: float,
    ) -> GenerationResult:
        """Deterministic mock generator for offline development, CI, and test execution."""
        latency_ms = (time.perf_counter() - start_time) * 1000.0 + 15.0  # slight realistic delay
        prompt_len = len(prompt)
        input_tokens = max(10, prompt_len // 4)

        text = ""
        parsed_json: Optional[Dict[str, Any]] = None

        if agent_role == "researcher":
            # Extract task-specific context from prompt if available
            facts = [
                "Repository uses Google ADK for multi-agent workflows.",
                "Gemini Developer API is integrated with API key authentication.",
                "Bounded loop is guarded by max iterations and Judge feedback."
            ]
            if "pyproject.toml" in prompt.lower() or "dependencies" in prompt.lower():
                facts = [
                    "Project name is agentops-code-intelligence",
                    "Python requirement is >=3.10",
                    "Uses google-adk as agent framework",
                    "Uses google-genai for Gemini Developer API access"
                ]
            elif "difference" in prompt.lower() or "vertex" in prompt.lower() or "studio" in prompt.lower():
                facts = [
                    "AI Studio uses API key authentication",
                    "Vertex AI requires GCP project and IAM credentials",
                    "AI Studio operates without Vertex AI billing dependencies"
                ]

            data = {
                "summary": "Detailed research findings investigated from the source codebase.",
                "identified_facts": facts,
                "supporting_evidence": [
                    "pyproject.toml defines project configuration.",
                    "agentops/config.py reads GEMINI_API_KEY from environment.",
                    "Orchestrator tracks escalation conditions and termination limits."
                ],
                "source_references": [
                    "pyproject.toml#L1-L25",
                    "agentops/config.py#L30-L70",
                    "docs/architecture.md"
                ],
                "uncertainties": [
                    "Exact rate limits for live Google AI Studio API tier."
                ],
                "unanswered_questions": []
            }
            parsed_json = data
            text = json.dumps(data, indent=2)

        elif agent_role == "writer":
            data = {
                "title": "Architectural Synthesis: Multi-Agent Intelligence Engine",
                "executive_summary": "The platform implements a verified multi-agent research workflow powered by Google ADK.",
                "detailed_analysis": (
                    "### 1. Agent Architecture\n"
                    "The system divides responsibilities between Researcher, Writer, and Judge agents.\n"
                    "The Researcher collects concrete facts from the codebase [pyproject.toml#L1-L25].\n"
                    "### 2. Telemetry and Verification\n"
                    "All LLM operations are observed via OpenTelemetry and validated by the independent Judge [agentops/config.py#L30-L70]."
                ),
                "cited_sources": [
                    "pyproject.toml#L1-L25",
                    "agentops/config.py#L30-L70"
                ],
                "confidence_score": 0.95
            }
            parsed_json = data
            text = json.dumps(data, indent=2)

        elif agent_role == "judge":
            # Check if this is a test intended to reject (contains trigger words like "reject", "hallucination", "fail")
            should_reject = any(k in prompt.lower() for k in ["reject", "revise", "fabricated", "trigger_reject"])
            if should_reject:
                data = {
                    "status": "fail",
                    "overall_score": 2.5,
                    "hallucination_score": 2.0,
                    "response_match_score": 3.0,
                    "unsupported_claims": [
                        "Claim regarding unverified cloud service deployment is unsupported by evidence."
                    ],
                    "missing_information": [
                        "Missing explicit code citations for the orchestrator loop."
                    ],
                    "revision_instructions": "Remove unsupported deployment claims and cite orchestrator implementation directly."
                }
            else:
                data = {
                    "status": "pass",
                    "overall_score": 4.8,
                    "hallucination_score": 5.0,
                    "response_match_score": 4.8,
                    "unsupported_claims": [],
                    "missing_information": [],
                    "revision_instructions": "Draft satisfies all grounding and completeness criteria."
                }
            parsed_json = data
            text = json.dumps(data, indent=2)

        elif agent_role == "evaluator_hallucination":
            data = {
                "hallucination_score": 5.0,
                "unsupported_claim_count": 0,
                "unsupported_claims": [],
                "evidence_coverage": 1.0,
                "evaluation_rationale": "Every factual statement corresponds directly to provided source reference material."
            }
            parsed_json = data
            text = json.dumps(data, indent=2)

        elif agent_role == "evaluator_response_match":
            data = {
                "response_match_score": 4.9,
                "missing_information": [],
                "irrelevant_information": [],
                "contradictions": [],
                "evaluation_explanation": "Response completely and accurately answers all facets of the query."
            }
            parsed_json = data
            text = json.dumps(data, indent=2)

        else:
            text = "Mock LLM output for: " + prompt[:100]

        output_tokens = max(15, len(text) // 4)
        total_tokens = input_tokens + output_tokens
        cost_usd = self.settings.calculate_cost(input_tokens, output_tokens, model)

        return GenerationResult(
            text=text,
            parsed_json=parsed_json,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=cost_usd,
            model=model,
            is_mock=True,
            finish_reason="STOP",
        )
