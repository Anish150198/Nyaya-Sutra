"""
Local GGUF LLM runner via llama-cpp-python. CPU-only.
"""

import logging
from core.config import LOCAL_LLM_PATH, MAX_OUTPUT_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    if not LOCAL_LLM_PATH:
        raise RuntimeError("LOCAL_LLM_PATH not set in .env")
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("llama-cpp-python required for local LLM. pip install llama-cpp-python")
    logger.info("Loading local GGUF model from %s (CPU-only)...", LOCAL_LLM_PATH)
    _llm = Llama(model_path=LOCAL_LLM_PATH, n_ctx=2048, n_threads=4, verbose=False)
    logger.info("Local LLM loaded.")
    return _llm


def generate(prompt: str, **kwargs) -> dict:
    """Generate text with a local GGUF model on CPU."""
    llm = _get_llm()
    try:
        output = llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", MAX_OUTPUT_TOKENS),
            temperature=kwargs.get("temperature", LLM_TEMPERATURE),
            stop=["\n\n\n", "###"],
            echo=False,
        )
        text = output["choices"][0]["text"].strip()
        return {"text": text, "model_id": "local/gguf", "tokens_used": output.get("usage", {}).get("total_tokens", 0)}
    except Exception as e:
        logger.error("Local LLM generation failed: %s", e)
        return {"text": f"[Local LLM Error: {e}]", "model_id": "local/gguf", "tokens_used": 0}
