"""Classification metrics: Finding-level Precision, Recall, and F1.

Implements rigorous matching between predicted findings and ground-truth reference facts.
Calculates Micro and Macro averaged Precision, Recall, and F1 with robust zero-denominator guards.
"""

from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field


class FindingMetricsResult(BaseModel):
    """Classification metrics for a single evaluation example."""
    case_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float = Field(description="TP / (TP + FP)")
    recall: float = Field(description="TP / (TP + FN)")
    f1: float = Field(description="2 * Precision * Recall / (Precision + Recall)")
    matched_pairs: List[Tuple[str, str]] = Field(default_factory=list)
    unmatched_predictions: List[str] = Field(default_factory=list)
    unmatched_references: List[str] = Field(default_factory=list)


class BatchClassificationMetrics(BaseModel):
    """Aggregated micro and macro classification metrics across benchmark cases."""
    total_cases: int
    total_true_positives: int
    total_false_positives: int
    total_false_negatives: int

    # Micro-averaged metrics: aggregate TP, FP, FN globally before computing
    micro_precision: float
    micro_recall: float
    micro_f1: float

    # Macro-averaged metrics: compute per-example and calculate arithmetic mean
    macro_precision: float
    macro_recall: float
    macro_f1: float

    case_results: List[FindingMetricsResult] = Field(default_factory=list)


class ClassificationMetricsEvaluator:
    """Evaluates predicted agent findings against reference ground-truth facts."""

    def __init__(self, match_threshold: float = 0.60):
        self.match_threshold = match_threshold

    @staticmethod
    def _normalize_text(text: str) -> str:
        import re
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(clean.split())

    def _similarity(self, s1: str, s2: str) -> float:
        """
        Calculates hybrid similarity: Jaccard token overlap + character sequence ratio.
        """
        import difflib
        norm1 = self._normalize_text(s1)
        norm2 = self._normalize_text(s2)

        if not norm1 or not norm2:
            return 0.0
        if norm1 == norm2:
            return 1.0

        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard = intersection / union if union > 0 else 0.0

        seq_ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        return round(0.6 * jaccard + 0.4 * seq_ratio, 4)

    def evaluate_single(
        self,
        predicted_findings: List[str],
        reference_facts: List[str],
        case_id: str = "CASE-N/A",
    ) -> FindingMetricsResult:
        """
        Matches predicted findings against reference facts using greedy bipartite matching.
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 * P * R / (P + R)
        """
        matched_preds: Set[int] = set()
        matched_refs: Set[int] = set()
        matched_pairs: List[Tuple[str, str]] = []

        # Find best pairs above similarity threshold
        candidate_pairs = []
        for p_idx, pred in enumerate(predicted_findings):
            for r_idx, ref in enumerate(reference_facts):
                sim = self._similarity(pred, ref)
                if sim >= self.match_threshold:
                    candidate_pairs.append((sim, p_idx, r_idx))

        # Sort descending by similarity
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        for sim, p_idx, r_idx in candidate_pairs:
            if p_idx not in matched_preds and r_idx not in matched_refs:
                matched_preds.add(p_idx)
                matched_refs.add(r_idx)
                matched_pairs.append((predicted_findings[p_idx], reference_facts[r_idx]))

        tp = len(matched_pairs)
        fp = len(predicted_findings) - tp
        fn = len(reference_facts) - tp

        # Zero denominator handlers
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        unmatched_preds = [pred for i, pred in enumerate(predicted_findings) if i not in matched_preds]
        unmatched_refs = [ref for j, ref in enumerate(reference_facts) if j not in matched_refs]

        return FindingMetricsResult(
            case_id=case_id,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            matched_pairs=matched_pairs,
            unmatched_predictions=unmatched_preds,
            unmatched_references=unmatched_refs,
        )

    def evaluate_batch(
        self,
        batch_items: List[Dict[str, Any]],
    ) -> BatchClassificationMetrics:
        """
        Evaluates batch and computes both Micro and Macro metrics.
        - Micro Precision = sum(TP) / (sum(TP) + sum(FP))
        - Micro Recall    = sum(TP) / (sum(TP) + sum(FN))
        - Micro F1        = 2 * MicroP * MicroR / (MicroP + MicroR)
        - Macro Precision = avg(Precision_i)
        - Macro Recall    = avg(Recall_i)
        - Macro F1        = avg(F1_i)
        """
        total_cases = len(batch_items)
        if total_cases == 0:
            return BatchClassificationMetrics(
                total_cases=0,
                total_true_positives=0,
                total_false_positives=0,
                total_false_negatives=0,
                micro_precision=0.0,
                micro_recall=0.0,
                micro_f1=0.0,
                macro_precision=0.0,
                macro_recall=0.0,
                macro_f1=0.0,
                case_results=[],
            )

        case_results: List[FindingMetricsResult] = []
        sum_tp = 0
        sum_fp = 0
        sum_fn = 0

        sum_prec = 0.0
        sum_rec = 0.0
        sum_f1 = 0.0

        for item in batch_items:
            case_id = item.get("case_id", "CASE-UNKNOWN")
            preds = item.get("predicted_findings", [])
            refs = item.get("reference_facts", [])

            res = self.evaluate_single(preds, refs, case_id=case_id)
            case_results.append(res)

            sum_tp += res.true_positives
            sum_fp += res.false_positives
            sum_fn += res.false_negatives

            sum_prec += res.precision
            sum_rec += res.recall
            sum_f1 += res.f1

        # Global Micro
        micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
        micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        # Unweighted Macro
        macro_p = sum_prec / total_cases
        macro_r = sum_rec / total_cases
        macro_f1 = sum_f1 / total_cases

        return BatchClassificationMetrics(
            total_cases=total_cases,
            total_true_positives=sum_tp,
            total_false_positives=sum_fp,
            total_false_negatives=sum_fn,
            micro_precision=round(micro_p, 4),
            micro_recall=round(micro_r, 4),
            micro_f1=round(micro_f1, 4),
            macro_precision=round(macro_p, 4),
            macro_recall=round(macro_r, 4),
            macro_f1=round(macro_f1, 4),
            case_results=case_results,
        )
