"""
Script to run BhashaBench-Legal datasets through the RAG pipeline.
Evaluates accuracy on MCQ-format legal questions across languages.
"""

import logging
import json
from pathlib import Path
from typing import Optional

from core.data_models import UserQuery, Persona, ActFilter
from agents.orchestrator import handle as orchestrator_handle
from evaluation.ragas_metrics import log_metrics_to_mlflow

logger = logging.getLogger(__name__)


def load_bhashabench_questions(path: Optional[str] = None) -> list[dict]:
    """
    Load BhashaBench-Legal MCQ dataset.

    Expected format per entry:
    {
        "id": "bb_001",
        "question_text": "...",
        "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
        "correct_option": "B",
        "language": "en",
        "domain": "criminal",
        "difficulty": "medium"
    }
    """
    if path is None:
        path = str(
            Path(__file__).resolve().parent.parent
            / "data" / "bronze" / "eval" / "bhashabench_legal.json"
        )

    p = Path(path)
    if not p.exists():
        logger.warning("BhashaBench dataset not found at %s", p)
        return []

    with open(p) as f:
        data = json.load(f)
    logger.info("Loaded %d BhashaBench-Legal questions", len(data))
    return data


def _extract_option_letter(answer_text: str, options: list[str]) -> Optional[str]:
    """
    Attempt to map the model's free-form answer to one of the MCQ options.

    Strategy:
    1. Look for explicit option letter (A, B, C, D) in the answer.
    2. Compute token overlap between answer and each option text.
    """
    answer_lower = answer_text.lower().strip()

    # Direct letter match
    for letter in ["a", "b", "c", "d"]:
        patterns = [
            f"answer is {letter}",
            f"option {letter}",
            f"correct answer: {letter}",
            f"({letter})",
        ]
        for pat in patterns:
            if pat in answer_lower:
                return letter.upper()

    # Fallback: best token overlap with option text
    best_letter = None
    best_score = 0.0
    answer_tokens = set(answer_lower.split())

    for opt in options:
        opt_lower = opt.lower()
        # Extract letter prefix
        letter = opt_lower.strip()[0].upper() if opt_lower.strip() else None
        # Remove "A) " prefix for comparison
        opt_text = opt_lower[2:].strip() if len(opt_lower) > 2 else opt_lower
        opt_tokens = set(opt_text.split())

        if not opt_tokens:
            continue
        overlap = len(answer_tokens & opt_tokens) / len(opt_tokens)
        if overlap > best_score:
            best_score = overlap
            best_letter = letter

    if best_score > 0.3:
        return best_letter
    return None


def run_evaluation(
    questions: Optional[list[dict]] = None,
    model_id: str = "param1",
    language_filter: Optional[str] = None,
    max_questions: Optional[int] = None,
    experiment_name: str = "/nyaya-sahayak/bhashabench-eval",
) -> dict:
    """
    Run the BhashaBench-Legal evaluation suite.

    Parameters
    ----------
    questions : list[dict] or None
        Pre-loaded questions; loads from default path if None.
    model_id : str
        Model to evaluate.
    language_filter : str or None
        Filter questions by language code (e.g., "en", "hi").
    max_questions : int or None
        Limit number of questions for quick testing.
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    dict with keys: total, correct, accuracy, per_domain, per_difficulty
    """
    if questions is None:
        questions = load_bhashabench_questions()

    if not questions:
        logger.warning("No questions to evaluate")
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    if language_filter:
        questions = [q for q in questions if q.get("language") == language_filter]

    if max_questions:
        questions = questions[:max_questions]

    total = len(questions)
    correct = 0
    per_domain: dict[str, dict] = {}
    per_difficulty: dict[str, dict] = {}
    results_log = []

    for i, q in enumerate(questions):
        logger.info("Evaluating question %d/%d: %s", i + 1, total, q.get("id", "?"))

        query = UserQuery(
            text=q["question_text"],
            user_lang=q.get("language", "en"),
            persona=Persona.CITIZEN,
            act_filter=ActFilter.ALL,
        )

        try:
            response = orchestrator_handle(query)
            predicted_letter = _extract_option_letter(
                response.answer_text, q.get("options", [])
            )
        except Exception as exc:
            logger.error("Error on question %s: %s", q.get("id"), exc)
            predicted_letter = None

        is_correct = (predicted_letter == q.get("correct_option"))
        if is_correct:
            correct += 1

        # Track per-domain
        domain = q.get("domain", "unknown")
        if domain not in per_domain:
            per_domain[domain] = {"total": 0, "correct": 0}
        per_domain[domain]["total"] += 1
        if is_correct:
            per_domain[domain]["correct"] += 1

        # Track per-difficulty
        diff = q.get("difficulty", "unknown")
        if diff not in per_difficulty:
            per_difficulty[diff] = {"total": 0, "correct": 0}
        per_difficulty[diff]["total"] += 1
        if is_correct:
            per_difficulty[diff]["correct"] += 1

        results_log.append({
            "id": q.get("id"),
            "predicted": predicted_letter,
            "correct": q.get("correct_option"),
            "is_correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0

    # Compute per-domain and per-difficulty accuracies
    for d in per_domain.values():
        d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0
    for d in per_difficulty.values():
        d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_domain": per_domain,
        "per_difficulty": per_difficulty,
        "model_id": model_id,
    }

    # Log to MLflow
    metrics = {"accuracy": accuracy, "total_questions": total, "correct_answers": correct}
    for domain, stats in per_domain.items():
        metrics[f"accuracy_{domain}"] = stats["accuracy"]

    log_metrics_to_mlflow(
        experiment_name=experiment_name,
        metrics=metrics,
        params={"model_id": model_id, "language": language_filter or "all"},
        run_name=f"bhashabench_{model_id}",
    )

    logger.info("BhashaBench evaluation complete: %d/%d correct (%.1f%%)", correct, total, accuracy * 100)
    return summary
