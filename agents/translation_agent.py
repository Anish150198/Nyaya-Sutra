"""
Translation Agent – wraps IndicTrans2 for bidirectional translation.
The orchestrator always interacts with canonical text (English) and
translates back to the user's language before returning.
"""

import logging
from typing import Optional

from models.translation.indictrans2_runner import to_canonical, from_canonical, translate, LANG_CODE_MAP

logger = logging.getLogger(__name__)


def handle_to_canonical(text: str, src_lang: str = "auto") -> tuple[str, str]:
    """
    Translate user input to canonical language (English).

    Parameters
    ----------
    text : str
        Raw user input.
    src_lang : str
        Detected or specified source language code. "auto" for auto-detect.

    Returns
    -------
    (canonical_text, canonical_lang)
    """
    logger.info("Translating to canonical (src=%s)", src_lang)
    try:
        canonical_text, canonical_lang = to_canonical(text, src_lang)
        return canonical_text, canonical_lang
    except Exception as exc:
        logger.error("Translation to canonical failed: %s. Returning original.", exc)
        return text, src_lang if src_lang != "auto" else "en"


def handle_from_canonical(text: str, target_lang: str) -> str:
    """
    Translate canonical (English) text back to user's language.

    Parameters
    ----------
    text : str
        Canonical English text (the AI's answer).
    target_lang : str
        Target language code (e.g., "hi", "ta", "bn").

    Returns
    -------
    str  Translated text.
    """
    if target_lang == "en":
        return text

    logger.info("Translating from canonical to %s", target_lang)
    try:
        return from_canonical(text, target_lang)
    except Exception as exc:
        logger.error("Translation from canonical failed: %s. Returning English.", exc)
        return text


def get_supported_languages() -> list[str]:
    """Return list of supported language codes."""
    return list(LANG_CODE_MAP.keys())
