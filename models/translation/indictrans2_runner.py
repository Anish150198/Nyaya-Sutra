"""
Translation runner for Nyaya-Sahayak.
Uses OpenAI GPT for translation when non-English input is detected.
For English input, returns text unchanged (zero latency, zero cost).

Replaces the heavy CTranslate2/IndicTrans2 dependency.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Language code mapping (for reference / UI display)
LANG_CODE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
}

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali",
    "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu",
}


def _openai_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text using OpenAI ChatCompletion."""
    try:
        from openai import OpenAI
        from core.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
    except ImportError:
        logger.warning("openai not installed, returning original text")
        return text

    src_name = LANG_NAMES.get(src_lang, src_lang)
    tgt_name = LANG_NAMES.get(tgt_lang, "English")

    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    client = OpenAI(**kwargs)
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an expert Indian legal translator. "
                    f"Translate the following text from {src_name} to {tgt_name}. "
                    "STRICTLY PRESERVE these exact legal acronyms without expanding or translating them: "
                    "BNS, BNSS, BSA, IPC, CrPC, IEA. For example, if the user says 'बीएनएस धारा 300', translate it to 'BNS section 300'. "
                    "Return ONLY the translated text in the target language. Do not add anything else."
                ),
            },
                {"role": "user", "content": text},
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        prompt = response.choices[0].message.content.strip()

        # Post-translation normalization (Defensive check)
        # If the LLM expanded BNS to "Bharatiya Nyaya Sanhita", we want to make sure 
        # the downstream regex has the best chance, though we expanded the regex 
        # too, this helps with consistency.
        norm_map = {
            "Bharatiya Nyaya Sanhita": "BNS",
            "Indian Penal Code": "BNS",
            "Bharatiya Nagarik Suraksha Sanhita": "BNSS",
            "Code of Criminal Procedure": "BNSS",
            "Bharatiya Sakshya Adhiniyam": "BSA",
            "Indian Evidence Act": "BSA",
        }
        for full, acr in norm_map.items():
            prompt = prompt.replace(full, acr)

        return prompt
    except Exception as exc:
        logger.error("OpenAI translation failed: %s. Returning original.", exc)
        return text


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text between supported languages.

    Parameters
    ----------
    text : str
    src_lang : str  Source language code (e.g., "hi")
    tgt_lang : str  Target language code (e.g., "en")

    Returns
    -------
    str  Translated text.
    """
    if src_lang == tgt_lang:
        return text
    if not text.strip():
        return text

    logger.info("Translating %s → %s via OpenAI (%d chars)", src_lang, tgt_lang, len(text))
    return _openai_translate(text, src_lang, tgt_lang)


def to_canonical(text: str, src_lang: str = "auto") -> tuple[str, str]:
    """
    Translate to canonical language (English).
    If already English, return as-is.

    Returns
    -------
    (canonical_text, detected_canonical_lang)
    """
    if src_lang == "en":
        return text, "en"

    if src_lang == "auto":
        # Heuristic: if mostly ASCII, assume English
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio > 0.8:
            return text, "en"
        # Default to Hindi for Devanagari-heavy text
        src_lang = "hi"

    canonical = translate(text, src_lang=src_lang, tgt_lang="en")
    return canonical, "en"


def from_canonical(text: str, target_lang: str) -> str:
    """Translate from English to the target language."""
    if target_lang == "en":
        return text
    return translate(text, src_lang="en", tgt_lang=target_lang)
