"""Independent LLM-as-a-Judge Hallucination Evaluator.

Evaluates factual grounding against provided context, identifying unsupported claims,
fabricated references, and unverified assumptions using an explicit 0-5 rubric.
Tracks its own Gemini token consumption and cost.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agentops.config import Settings, get_settings
from agentops.providers.gemini_provider import GeminiProvider

logger = logging.getLogger("agentops.evaluator.hallucination")

PROMPT_VERSION = "hallucination-rubric-v1.0.0"

HALLUCINATION_RUBRIC_INSTRUCTION = """You are an independent, adversarial LLM evaluation judge specializing in factual grounding.
Your task is to detect and grade hallucinations in the candidate response when compared against the provided reference context.

Scoring Rubric (0.0 to 5.0):
- 5.0 (Perfect Grounding): Every factual claim is directly supported by the context. Zero hallucinations or invented details.
- 4.0 (Minor Omission / Trivial extrapolation): Minor conversational filler or trivial phrasing that introduces no factual distortions.
- 3.0 (Ambiguous / Unsubstantiated): Contains claims presented as fact that cannot be verified from the context.
- 2.0 (Significant Hallucination): Includes clear factual claims that are unsupported or fabricated numbers/files.
- 1.0 (Severe Hallucination): The majority of factual assertions contradict or invent information beyond the context.
- 0.0 (Complete Fabrication): The entire output is fabricated and completely untethered from reality.

Evaluation Requirements:
1. Extract and count all unsupported claims.
2. Calculate evidence coverage (fraction of candidate facts grounded in context, 0.0 to 1.0).
3. Provide an explicit explanation justifying the score.

Return your evaluation strictly as JSON conforming to this schema:
{
  "hallucination_score": 4.8,
  "unsupported_claim_count": 0,
  "unsupported_claims": [],
  "evidence_coverage": 1.0,
  "evaluation_rationale": "Detailed explanation citing specific claims"
}
"""


class HallucinationEvaluationResult(BaseModel):
    """Result of an LLM-as-a-Judge hallucination evaluation."""
    case_id: str
    hallucination_score: float = Field(ge=0.0, le=5.0)
    unsupported_claim_count: int
    unsupported_claims: List[str] = Field(default_factory=list)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    evaluation_rationale: str
    prompt_version: str = PROMPT_VERSION
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None
    latency_ms: float = 0.0
    is_mock: bool = False


class HallucinationEvaluator:
    """Evaluates candidate agent responses for factual grounding and hallucinations."""

    def __init__(self, settings: Optional[Settings] = None, provider: Optional[GeminiProvider] = None):
        self.settings = settings or get_settings()
        self.provider = provider or GeminiProvider(settings=self.settings)

    async def evaluate(
        self,
        candidate_response: str,
        reference_context: str,
        case_id: str = "CASE-N/A",
    ) -> HallucinationEvaluationResult:
        """
        Runs independent LLM-as-a-Judge hallucination evaluation.
        """
        start_time = time.perf_counter()

        prompt = (
            f"Reference Context (Ground Truth Source Material):\n"
            f"```\n{reference_context}\n```\n\n"
            f"Candidate Response Under Evaluation:\n"
            f"```\n{candidate_response}\n```\n\n"
            f"Perform the evaluation according to the 0-5 hallucination rubric."
        )

        generation = await self.provider.generate(
            prompt=prompt,
            system_instruction=HALLUCINATION_RUBRIC_INSTRUCTION,
            agent_role="evaluator_hallucination",
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        score = 5.0
        claim_count = 0
        claims = []
        coverage = 1.0
        rationale = "Candidate response is supported by reference context."

        if generation.parsed_json and isinstance(generation.parsed_json, dict):
            try:
                score = float(generation.parsed_json.get("hallucination_score", 5.0))
                claim_count = int(generation.parsed_json.get("unsupported_claim_count", 0))
                claims = list(generation.parsed_json.get("unsupported_claims", []))
                coverage = float(generation.parsed_json.get("evidence_coverage", 1.0))
                rationale = str(generation.parsed_json.get("evaluation_rationale", rationale))
            except Exception as e:
                logger.warning(f"Error extracting hallucination fields: {e}")

        return HallucinationEvaluationResult(
            case_id=case_id,
            hallucination_score=max(0.0, min(5.0, score)),
            unsupported_claim_count=claim_count,
            unsupported_claims=claims,
            evidence_coverage=max(0.0, min(1.0, coverage)),
            evaluation_rationale=rationale,
            prompt_version=PROMPT_VERSION,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            total_tokens=generation.total_tokens,
            cost_usd=generation.cost_usd,
            latency_ms=round(elapsed_ms, 2),
            is_mock=generation.is_mock,
        )
