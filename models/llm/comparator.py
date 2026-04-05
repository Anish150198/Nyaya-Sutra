"""
Multi-LLM comparison module.
Runs the same prompt through multiple models and optionally uses a referee step.
"""

import logging
from typing import Optional

from models.llm.router import run_model

logger = logging.getLogger(__name__)


def run_dual_models(
    prompt: str,
    models: list[str] | None = None,
    referee: bool = True,
    **gen_kwargs,
) -> dict:
    """
    Run a prompt through two (or more) models and return a comparison dict.

    Parameters
    ----------
    prompt : str
        The assembled prompt (context + question).
    models : list[str]
        Model IDs to compare. Defaults to ["openai"] (single model with self-comparison).
    referee : bool
        If True, use the first model with a meta-prompt to judge answers.

    Returns
    -------
    dict  {model_id: answer_dict, "comparison": analysis_str | None}
    """
    if models is None:
        models = ["openai"]

    results: dict = {}
    for mid in models:
        try:
            results[mid] = run_model(mid, prompt, **gen_kwargs)
        except Exception as exc:
            logger.error("Model %s failed: %s", mid, exc)
            results[mid] = {"text": f"[Error: {exc}]", "model_id": mid}

    comparison = None
    if referee and len(results) >= 2:
        comparison = _referee_compare(prompt, results)

    results["comparison"] = comparison
    return results


def _referee_compare(original_prompt: str, results: dict) -> Optional[str]:
    """Use the first model as a referee to compare answers."""
    model_ids = [k for k in results if k != "comparison"]
    if len(model_ids) < 2:
        return None

    answers_block = "\n\n".join(
        f"### Answer from {mid}:\n{results[mid].get('text', '')}"
        for mid in model_ids
    )

    referee_prompt = (
        "You are an expert Indian legal analyst. "
        "Given the following legal question and two AI-generated answers, "
        "compare them for accuracy, completeness, and citation quality. "
        "State which answer is better and why in 3-4 sentences.\n\n"
        f"### Question:\n{original_prompt[:500]}\n\n"
        f"{answers_block}\n\n"
        "### Your comparison:"
    )

    try:
        ref_out = run_model(model_ids[0], referee_prompt, max_tokens=256)
        return ref_out.get("text", "")
    except Exception as exc:
        logger.error("Referee comparison failed: %s", exc)
        return None
