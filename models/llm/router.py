"""
Model router — selects which LLM to use.
Uses OpenAI as the primary (and only required) provider.
Local GGUF models (param1/airavata) are disabled — OpenAI is used instead.
"""

import logging

from core.config import LLM_PROVIDER

logger = logging.getLogger(__name__)


def select_model(
    canonical_lang: str = "en",
    persona: str = "citizen",
    complexity: str = "normal",
) -> str:
    """
    Return a model identifier string.

    With OpenAI as provider, always returns "openai".
    If LLM_PROVIDER is set to "local", returns "local" (requires llama-cpp-python).
    """
    if LLM_PROVIDER == "openai":
        logger.info("Router selected: openai")
        return "openai"
    elif LLM_PROVIDER == "local":
        logger.info("Router selected: local (param1)")
        return "param1"
    else:
        logger.warning("Unknown LLM_PROVIDER=%s, defaulting to openai", LLM_PROVIDER)
        return "openai"


def run_model(model_id: str, prompt: str, **kwargs) -> dict:
    """Dispatch generation to the appropriate runner."""
    if model_id == "openai":
        from models.llm.openai_runner import generate
        return generate(prompt, **kwargs)
    elif model_id == "param1":
        try:
            from models.llm.param1_runner import generate
            return generate(prompt, **kwargs)
        except ImportError:
            logger.warning("param1 runner not available (llama-cpp-python missing), falling back to openai")
            from models.llm.openai_runner import generate
            return generate(prompt, **kwargs)
    elif model_id == "airavata":
        try:
            from models.llm.airavata_runner import generate
            return generate(prompt, **kwargs)
        except ImportError:
            logger.warning("Airavata runner not available, falling back to openai")
            from models.llm.openai_runner import generate
            return generate(prompt, **kwargs)
    else:
        # Fallback to openai for any unknown model id
        logger.warning("Unknown model_id: %s, using openai", model_id)
        from models.llm.openai_runner import generate
        return generate(prompt, **kwargs)
