"""
CPU inference wrapper for Param-1 2.9B (GGUF Q8_0) using llama-cpp-python.
"""

import logging
import time
from typing import Optional

from core.config import PARAM1_MODEL_PATH, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS, LLM_TEMPERATURE
from models.llm.model_loader import load_llama_model, estimate_memory_gb

logger = logging.getLogger(__name__)

_model = None


def get_model():
    """Lazy-load the Param-1 model singleton."""
    global _model
    if _model is None:
        _model = load_llama_model(
            model_path=PARAM1_MODEL_PATH,
            n_ctx=MAX_CONTEXT_TOKENS,
        )
    return _model


def generate(
    prompt: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = LLM_TEMPERATURE,
    top_p: float = 0.95,
    stop: Optional[list[str]] = None,
) -> dict:
    """
    Generate text from Param-1.

    Returns
    -------
    dict with keys: text, tokens_generated, ttft_ms, total_ms
    """
    model = get_model()
    stop = stop or ["\n\n\n", "###"]

    t0 = time.perf_counter()
    output = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        echo=False,
    )
    total_ms = (time.perf_counter() - t0) * 1000

    text = output["choices"][0]["text"].strip()
    tokens_generated = output["usage"].get("completion_tokens", 0)

    return {
        "text": text,
        "tokens_generated": tokens_generated,
        "total_ms": round(total_ms, 2),
        "model_id": "param1",
    }


def estimate_memory() -> float:
    """Return estimated memory in GB for Param-1 GGUF."""
    return estimate_memory_gb(PARAM1_MODEL_PATH)
