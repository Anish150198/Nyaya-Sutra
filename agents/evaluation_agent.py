"""
Evaluation Agent – runs BhashaBench-Legal subsets or custom evaluation sets
through the RAG pipeline and logs results to MLflow.
Provides summary stats for a "Model performance" view in the UI.
"""

import logging
from typing import Optional

from evaluation.bhashabench_eval import run_evaluation, load_bhashabench_questions
from evaluation.ragas_metrics import log_metrics_to_mlflow
from core.config import MLFLOW_EXPERIMENT_NAME

logger = logging.getLogger(__name__)


def handle(
    model_id: str = "param1",
    language_filter: Optional[str] = None,
    max_questions: Optional[int] = 10,
    models_to_compare: Optional[list[str]] = None,
) -> dict:
    """
    Run evaluation and return summary stats.

    Parameters
    ----------
    model_id : str
        Primary model to evaluate.
    language_filter : str or None
        Only evaluate questions in this language.
    max_questions : int or None
        Cap the number of questions for quick evaluation.
    models_to_compare : list[str] or None
        If provided, evaluate each model and return comparative results.

    Returns
    -------
    dict with per-model evaluation summaries.
    """
    logger.info(
        "Evaluation agent: model=%s, lang=%s, max_q=%s",
        model_id, language_filter, max_questions,
    )

    questions = load_bhashabench_questions()
    if not questions:
        return {"error": "No BhashaBench-Legal dataset found. Place it in data/bronze/eval/"}

    models = models_to_compare or [model_id]
    results = {}

    for mid in models:
        logger.info("Evaluating model: %s", mid)
        summary = run_evaluation(
            questions=questions,
            model_id=mid,
            language_filter=language_filter,
            max_questions=max_questions,
            experiment_name=MLFLOW_EXPERIMENT_NAME,
        )
        results[mid] = summary

    # If comparing, add a comparative summary
    if len(results) > 1:
        best_model = max(results, key=lambda m: results[m].get("accuracy", 0))
        results["_comparison"] = {
            "best_model": best_model,
            "best_accuracy": results[best_model].get("accuracy", 0),
            "models_evaluated": list(results.keys()),
        }

    return results
