"""
Intent classifier – determines whether a user query is LEGAL, WELFARE, MIXED, or GENERIC.
Uses keyword heuristics + optional embedding similarity for lightweight CPU classification.
"""

import logging
import re
from core.data_models import Intent

logger = logging.getLogger(__name__)

# Keyword banks for heuristic classification
LEGAL_KEYWORDS = [
    "bns", "bnss", "bsa", "ipc", "crpc", "constitution", "section", "article",
    "punishment", "offence", "offense", "bail", "fir", "arrest", "trial",
    "court", "judge", "advocate", "law", "legal", "crime", "criminal",
    "murder", "theft", "fraud", "defamation", "assault", "dowry",
    "divorce", "custody", "maintenance", "rights", "fundamental",
    "directive", "amendment", "act", "statute", "penalty", "sentence",
    "summons", "warrant", "evidence", "witness", "prosecution",
    "complaint", "appeal", "petition", "writ", "habeas corpus",
    "भारतीय न्याय संहिता", "धारा", "कानून", "अपराध", "सज़ा", "गिरफ्तारी",
]

WELFARE_KEYWORDS = [
    "scheme", "yojana", "benefit", "subsidy", "pension", "scholarship",
    "housing", "employment", "ration", "bpl", "apl", "income",
    "pmay", "mgnrega", "nrega", "jan dhan", "ayushman", "mudra",
    "kisan", "fasal", "ujjwala", "sukanya", "atal", "maternity",
    "widow", "disability", "sc/st", "obc", "minority", "tribal",
    "eligible", "eligibility", "apply", "registration",
    "योजना", "लाभ", "सब्सिडी", "पेंशन", "छात्रवृत्ति", "आवास",
    "legal aid", "nalsa", "tele-law", "nyaya bandhu", "free lawyer",
]


def classify(text: str) -> Intent:
    """
    Classify the intent of a user query.

    Returns
    -------
    Intent enum value: LEGAL, WELFARE, MIXED, or GENERIC
    """
    text_lower = text.lower()

    legal_score = sum(1 for kw in LEGAL_KEYWORDS if kw in text_lower)
    welfare_score = sum(1 for kw in WELFARE_KEYWORDS if kw in text_lower)

    # Check for IPC section patterns like "IPC 302" or "section 420"
    if re.search(r'\b(ipc|bns|bnss|bsa|crpc)\s*\d+', text_lower):
        legal_score += 3
    if re.search(r'\bsection\s+\d+', text_lower):
        legal_score += 2
    if re.search(r'\barticle\s+\d+', text_lower):
        legal_score += 2

    logger.debug("Intent scores – legal: %d, welfare: %d", legal_score, welfare_score)

    if legal_score > 0 and welfare_score > 0:
        return Intent.MIXED
    elif legal_score > 0:
        return Intent.LEGAL
    elif welfare_score > 0:
        return Intent.WELFARE
    else:
        return Intent.GENERIC
