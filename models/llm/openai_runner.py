"""
OpenAI API wrapper for LLM generation. CPU-only (API call, no local GPU needed).
Supports OpenAI, Azure OpenAI, and any OpenAI-compatible endpoint.
"""

import logging
from core.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL, MAX_OUTPUT_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. pip install openai")

    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    _client = OpenAI(**kwargs)
    logger.info("OpenAI client initialized (model=%s)", OPENAI_MODEL)
    return _client


def generate(prompt: str, model: str = None, **kwargs) -> dict:
    """
    Generate text using OpenAI API.

    Parameters
    ----------
    prompt : str
        The full prompt (system + context + question).
    model : str or None
        Override the model name.

    Returns
    -------
    dict with keys: text, model_id, tokens_used
    """
    client = _get_client()
    model_name = model or OPENAI_MODEL

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", MAX_OUTPUT_TOKENS),
            temperature=kwargs.get("temperature", LLM_TEMPERATURE),
        )
        text = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0

        return {
            "text": text,
            "model_id": f"openai/{model_name}",
            "tokens_used": tokens,
        }
    except Exception as e:
        logger.error("OpenAI generation failed: %s", e)
        return {
            "text": f"[LLM Error: {e}]",
            "model_id": f"openai/{model_name}",
            "tokens_used": 0,
        }
