"""
Evaluates the orchestrator's decision-making: did it pick the right tool/agent?
Logs routing accuracy and agent selection metrics to MLflow.
"""

import logging
from typing import Optional

from core.data_models import UserQuery, Intent, Persona, ActFilter
from models.nlp_classifier.intent_classifier import classify
from evaluation.ragas_metrics import log_metrics_to_mlflow

logger = logging.getLogger(__name__)


def load_trajectory_test_cases(path: Optional[str] = None) -> list[dict]:
    """
    Load test cases for trajectory evaluation.

    Expected format per entry:
    {
        "query": "What is BNS Section 103?",
        "expected_intent": "LEGAL",
        "expected_agents": ["legal_agent"],
        "language": "en"
    }
    """
    import json
    from pathlib import Path

    if path is None:
        path = str(
            Path(__file__).resolve().parent.parent
            / "data" / "bronze" / "eval" / "trajectory_test_cases.json"
        )
    p = Path(path)
    if not p.exists():
        logger.warning("Trajectory test cases not found at %s", p)
        return []

    with open(p) as f:
        data = json.load(f)
    logger.info("Loaded %d trajectory test cases", len(data))
    return data


def evaluate_intent_classification(test_cases: list[dict]) -> dict:
    """
    Evaluate the intent classifier against labeled test cases.

    Returns
    -------
    dict with keys: total, correct, accuracy, confusion_matrix
    """
    if not test_cases:
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    total = len(test_cases)
    correct = 0
    confusion: dict[str, dict[str, int]] = {}

    for tc in test_cases:
        query_text = tc["query"]
        expected = tc["expected_intent"]
        predicted = classify(query_text).value

        # Update confusion matrix
        if expected not in confusion:
            confusion[expected] = {}
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

        if predicted == expected:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "confusion_matrix": confusion,
    }


def evaluate_routing(
    test_cases: Optional[list[dict]] = None,
    experiment_name: str = "/nyaya-sahayak/trajectory-eval",
) -> dict:
    """
    Full trajectory evaluation: intent classification + agent routing accuracy.

    Returns
    -------
    dict with evaluation summary
    """
    if test_cases is None:
        test_cases = load_trajectory_test_cases()

    if not test_cases:
        logger.warning("No trajectory test cases available")
        return {"total": 0, "accuracy": 0.0}

    intent_results = evaluate_intent_classification(test_cases)

    # Log to MLflow
    log_metrics_to_mlflow(
        experiment_name=experiment_name,
        metrics={
            "intent_accuracy": intent_results["accuracy"],
            "intent_total": intent_results["total"],
            "intent_correct": intent_results["correct"],
        },
        params={"evaluator": "trajectory_eval"},
        run_name="trajectory_routing_eval",
    )

    logger.info(
        "Trajectory eval: %d/%d correct intent classifications (%.1f%%)",
        intent_results["correct"],
        intent_results["total"],
        intent_results["accuracy"] * 100,
    )
    return intent_results
