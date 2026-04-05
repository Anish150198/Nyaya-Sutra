"""
MLflow RAGAS scorers: Faithfulness, AnswerAccuracy, ContextPrecision.
Designed for evaluating the RAG pipeline's output quality.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_faithfulness(answer: str, context: str) -> float:
    """
    Compute faithfulness score: how well the answer is grounded in the context.
    Uses RAGAS faithfulness metric when available, falls back to simple overlap.

    Returns
    -------
    float  Score between 0.0 and 1.0
    """
    try:
        from ragas.metrics import faithfulness
        from ragas import evaluate
        from datasets import Dataset

        ds = Dataset.from_dict({
            "question": [""],
            "answer": [answer],
            "contexts": [[context]],
        })
        result = evaluate(ds, metrics=[faithfulness])
        return result["faithfulness"]
    except ImportError:
        logger.warning("RAGAS not available, using simple overlap scorer")
        return _simple_overlap_score(answer, context)
    except Exception as exc:
        logger.error("Faithfulness computation failed: %s", exc)
        return _simple_overlap_score(answer, context)


def compute_answer_accuracy(answer: str, ground_truth: str) -> float:
    """
    Compute answer accuracy against a ground truth reference.

    Returns
    -------
    float  Score between 0.0 and 1.0
    """
    try:
        from ragas.metrics import answer_correctness
        from ragas import evaluate
        from datasets import Dataset

        ds = Dataset.from_dict({
            "question": [""],
            "answer": [answer],
            "ground_truth": [ground_truth],
            "contexts": [[""]],
        })
        result = evaluate(ds, metrics=[answer_correctness])
        return result["answer_correctness"]
    except ImportError:
        logger.warning("RAGAS not available, using simple match scorer")
        return _simple_match_score(answer, ground_truth)
    except Exception as exc:
        logger.error("Answer accuracy computation failed: %s", exc)
        return _simple_match_score(answer, ground_truth)


def compute_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """
    Compute context precision: are the retrieved contexts relevant to the question?

    Returns
    -------
    float  Score between 0.0 and 1.0
    """
    try:
        from ragas.metrics import context_precision
        from ragas import evaluate
        from datasets import Dataset

        ds = Dataset.from_dict({
            "question": [question],
            "answer": [""],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        })
        result = evaluate(ds, metrics=[context_precision])
        return result["context_precision"]
    except ImportError:
        logger.warning("RAGAS not available, returning default score")
        return 0.5
    except Exception as exc:
        logger.error("Context precision computation failed: %s", exc)
        return 0.5


def log_metrics_to_mlflow(
    experiment_name: str,
    metrics: dict,
    params: Optional[dict] = None,
    run_name: Optional[str] = None,
):
    """
    Log evaluation metrics to MLflow.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    metrics : dict
        Metric name → value pairs.
    params : dict or None
        Parameter name → value pairs.
    run_name : str or None
        Human-readable run name.
    """
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            logger.info("Logged metrics to MLflow: %s", metrics)
    except ImportError:
        logger.warning("MLflow not available, skipping metric logging")
    except Exception as exc:
        logger.error("MLflow logging failed: %s", exc)


# ── Fallback scorers (no external deps) ────────────────────────────────

def _simple_overlap_score(answer: str, context: str) -> float:
    """Token overlap between answer and context as a crude faithfulness proxy."""
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


def _simple_match_score(answer: str, ground_truth: str) -> float:
    """Token-level F1 between answer and ground truth."""
    a_tokens = set(answer.lower().split())
    g_tokens = set(ground_truth.lower().split())
    if not a_tokens or not g_tokens:
        return 0.0
    common = a_tokens & g_tokens
    precision = len(common) / len(a_tokens)
    recall = len(common) / len(g_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
