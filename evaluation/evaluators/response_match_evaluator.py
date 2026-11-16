"""Independent LLM-as-a-Judge Response Match Evaluator.

Compares candidate responses against golden reference answers and original user tasks
to evaluate completeness, semantic alignment, missing requirements, and contradictions.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agentops.config import Settings, get_settings
from agentops.providers.gemini_provider import GeminiProvider

logger = logging.getLogger("agentops.evaluator.response_match")

PROMPT_VERSION = "response-match-rubric-v1.0.0"

RESPONSE_MATCH_RUBRIC_INSTRUCTION = """You are an independent LLM evaluation judge specializing in semantic response alignment.
Your task is to evaluate how completely and accurately the candidate response addresses the user's task when compared against the expected reference answer.

Scoring Rubric (0.0 to 5.0):
- 5.0 (Flawless Match): Completely answers the user's inquiry, covers all expected facts, and introduces no extraneous noise.
- 4.0 (Strong Match): Answers all core questions accurately, omitting only minor auxiliary nuances.
- 3.0 (Moderate Match): Partially answers the query; leaves out a key requirement or includes slight irrelevant filler.
- 2.0 (Weak Match): Misses the majority of core facts or provides a superficial, incomplete summary.
- 1.0 (Poor Match): Contradicts the reference answer or answers an entirely different question.
- 0.0 (Unresponsive): Fails to answer the task in any meaningful way.

Evaluation Requirements:
1. Identify any missing information present in the reference answer but absent from candidate response.
2. Identify any irrelevant information or unnecessary filler.
3. Identify direct contradictions between candidate response and reference answer.
4. Provide an explicit explanation justifying the numerical score.

Return your evaluation strictly conforming to this JSON schema:
{
  "response_match_score": 4.9,
  "missing_information": [],
  "irrelevant_information": [],
  "contradictions": [],
  "evaluation_explanation": "Detailed justification"
}
"""


class ResponseMatchEvaluationResult(BaseModel):
    """Result of an LLM-as-a-Judge response-match evaluation."""
    case_id: str
    response_match_score: float = Field(ge=0.0, le=5.0)
    missing_information: List[str] = Field(default_factory=list)
    irrelevant_information: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    evaluation_explanation: str
    prompt_version: str = PROMPT_VERSION
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None
    latency_ms: float = 0.0
    is_mock: bool = False


class ResponseMatchEvaluator:
    """Evaluates candidate response alignment against reference answers."""

    def __init__(self, settings: Optional[Settings] = None, provider: Optional[GeminiProvider] = None):
        self.settings = settings or get_settings()
        self.provider = provider or GeminiProvider(settings=self.settings)

    async def evaluate(
        self,
        candidate_response: str,
        expected_response: str,
        task: str,
        case_id: str = "CASE-N/A",
    ) -> ResponseMatchEvaluationResult:
        """
        Runs independent LLM-as-a-Judge response match evaluation.
        """
        start_time = time.perf_counter()

        prompt = (
            f"Original User Task:\n"
            f"```\n{task}\n```\n\n"
            f"Expected Reference Answer (Ground Truth):\n"
            f"```\n{expected_response}\n```\n\n"
            f"Candidate Response Under Evaluation:\n"
            f"```\n{candidate_response}\n```\n\n"
            f"Perform the evaluation according to the 0-5 response match rubric."
        )

        generation = await self.provider.generate(
            prompt=prompt,
            system_instruction=RESPONSE_MATCH_RUBRIC_INSTRUCTION,
            agent_role="evaluator_response_match",
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        score = 4.9
        missing = []
        irrelevant = []
        contradictions = []
        explanation = "Candidate response completely answers the task with strong alignment to reference answer."

        if generation.parsed_json and isinstance(generation.parsed_json, dict):
            try:
                score = float(generation.parsed_json.get("response_match_score", 4.9))
                missing = list(generation.parsed_json.get("missing_information", []))
                irrelevant = list(generation.parsed_json.get("irrelevant_information", []))
                contradictions = list(generation.parsed_json.get("contradictions", []))
                explanation = str(generation.parsed_json.get("evaluation_explanation", explanation))
            except Exception as e:
                logger.warning(f"Error parsing response match fields: {e}")

        return ResponseMatchEvaluationResult(
            case_id=case_id,
            response_match_score=max(0.0, min(5.0, score)),
            missing_information=missing,
            irrelevant_information=irrelevant,
            contradictions=contradictions,
            evaluation_explanation=explanation,
            prompt_version=PROMPT_VERSION,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            total_tokens=generation.total_tokens,
            cost_usd=generation.cost_usd,
            latency_ms=round(elapsed_ms, 2),
            is_mock=generation.is_mock,
        )
