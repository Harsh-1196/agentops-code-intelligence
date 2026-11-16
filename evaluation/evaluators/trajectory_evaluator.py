"""Trajectory exact-match evaluator for multi-agent workflows.

Compares actual recorded agent execution trajectories against expected
golden trajectories using canonical normalization and sequence alignment.
"""

from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field

from agentops.schemas import AgentTrajectoryStep


class TrajectoryEvaluationResult(BaseModel):
    """Result of evaluating a single execution trajectory against golden benchmark."""
    case_id: str
    is_exact_match: bool
    actual_trajectory: List[str]
    expected_trajectory: List[str]
    missing_steps: List[str] = Field(default_factory=list)
    unexpected_steps: List[str] = Field(default_factory=list)
    sequence_length_diff: int = 0
    alignment_ratio: float = 0.0


class TrajectoryBatchResult(BaseModel):
    """Aggregated batch trajectory exact-match metrics."""
    total_evaluated: int
    exact_matches: int
    mismatches: int
    trajectory_exact_match: float = Field(
        description="Exact matching trajectories / total evaluated trajectories (0.0 to 1.0)"
    )
    results: List[TrajectoryEvaluationResult] = Field(default_factory=list)


class TrajectoryExactMatchEvaluator:
    """Evaluates agent trajectories against golden expected sequences."""

    @staticmethod
    def normalize_event(event: Union[str, AgentTrajectoryStep, Dict[str, Any]]) -> str:
        """Converts raw step or event into canonical normalized string identifier."""
        if isinstance(event, AgentTrajectoryStep):
            return event.canonical_event.strip().lower()
        if isinstance(event, dict):
            return str(event.get("canonical_event") or event.get("action") or "").strip().lower()

        raw = str(event).strip().lower()
        # Canonical alias mapping
        alias_map = {
            "classify": "orchestrator.classify",
            "research": "researcher.execute",
            "write": "writer.generate",
            "judge": "judge.evaluate",
            "approve": "orchestrator.approve",
            "reject": "orchestrator.reject",
        }
        return alias_map.get(raw, raw)

    @classmethod
    def normalize_sequence(cls, seq: List[Union[str, AgentTrajectoryStep, Dict[str, Any]]]) -> List[str]:
        """Normalizes an entire sequence of trajectory events."""
        return [cls.normalize_event(e) for e in seq if cls.normalize_event(e)]

    def evaluate_single(
        self,
        actual: List[Union[str, AgentTrajectoryStep, Dict[str, Any]]],
        expected: List[str],
        case_id: str = "CASE-N/A",
    ) -> TrajectoryEvaluationResult:
        """
        Evaluate whether actual normalized trajectory exactly matches expected sequence.
        Also identifies missing and unexpected steps.
        """
        norm_actual = self.normalize_sequence(actual)
        norm_expected = self.normalize_sequence(expected)

        is_exact = (norm_actual == norm_expected)

        # Compute missing steps (in expected but not in actual)
        actual_copy = list(norm_actual)
        missing: List[str] = []
        for step in norm_expected:
            if step in actual_copy:
                actual_copy.remove(step)
            else:
                missing.append(step)

        # Compute unexpected steps (in actual but not in expected)
        expected_copy = list(norm_expected)
        unexpected: List[str] = []
        for step in norm_actual:
            if step in expected_copy:
                expected_copy.remove(step)
            else:
                unexpected.append(step)

        # Sequence alignment similarity (Levenshtein-based ratio)
        import difflib
        matcher = difflib.SequenceMatcher(None, norm_actual, norm_expected)
        alignment_ratio = round(matcher.ratio(), 4)

        return TrajectoryEvaluationResult(
            case_id=case_id,
            is_exact_match=is_exact,
            actual_trajectory=norm_actual,
            expected_trajectory=norm_expected,
            missing_steps=missing,
            unexpected_steps=unexpected,
            sequence_length_diff=len(norm_actual) - len(norm_expected),
            alignment_ratio=alignment_ratio,
        )

    def evaluate_batch(
        self,
        test_pairs: List[Dict[str, Any]],
    ) -> TrajectoryBatchResult:
        """
        Evaluates a batch of (actual, expected) trajectory pairs.
        Calculates:
        trajectory_exact_match = exact matching trajectories / total evaluated trajectories
        """
        total = len(test_pairs)
        if total == 0:
            return TrajectoryBatchResult(
                total_evaluated=0,
                exact_matches=0,
                mismatches=0,
                trajectory_exact_match=0.0,
                results=[],
            )

        results: List[TrajectoryEvaluationResult] = []
        exact_count = 0

        for pair in test_pairs:
            case_id = pair.get("case_id", "CASE-UNKNOWN")
            actual = pair.get("actual", [])
            expected = pair.get("expected", [])

            res = self.evaluate_single(actual=actual, expected=expected, case_id=case_id)
            results.append(res)
            if res.is_exact_match:
                exact_count += 1

        tem_rate = round(exact_count / total, 4)

        return TrajectoryBatchResult(
            total_evaluated=total,
            exact_matches=exact_count,
            mismatches=total - exact_count,
            trajectory_exact_match=tem_rate,
            results=results,
        )
