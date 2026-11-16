"""Unit test for golden dataset integrity and coverage."""

import json
import os
from pydantic import BaseModel
from typing import List, Optional


class BenchmarkCase(BaseModel):
    id: str
    category: str
    task: str
    reference_context: str
    expected_response: str
    expected_facts: List[str]
    expected_trajectory: List[str]
    expected_tool_calls: List[str]
    expected_evidence_references: List[str]
    hallucination_sensitive_facts: List[str]
    expected_evaluation_label: str


def test_golden_dataset_structure():
    """Verify golden dataset satisfies all schema requirements and contains >= 30 cases."""
    dataset_path = os.path.join("evaluation", "datasets", "golden_dataset.json")
    assert os.path.exists(dataset_path), f"Golden dataset not found at {dataset_path}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "cases" in data
    cases = data["cases"]
    assert len(cases) >= 30, f"Expected at least 30 benchmark cases, got {len(cases)}"

    # Validate each case schema
    categories = set()
    for case_data in cases:
        case = BenchmarkCase(**case_data)
        assert len(case.expected_facts) > 0
        assert len(case.expected_trajectory) > 0
        categories.add(case.category)

    # Check multi-category coverage
    required_categories = {
        "factual_research",
        "repository_investigation",
        "technical_explanation",
        "source_grounded_summarization",
        "information_retrieval",
        "multi_step_reasoning",
        "misleading_or_insufficient_evidence",
    }
    assert required_categories.issubset(categories)
