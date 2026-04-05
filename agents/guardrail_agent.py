"""
Guardrail Agent – ensures the AI does not offer definitive legal counsel,
blocks harmful queries, and inserts appropriate disclaimers.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns that indicate harmful or out-of-scope queries
HARMFUL_PATTERNS = [
    r"\bhow\s+to\s+(commit|do|perform|execute)\s+(murder|theft|fraud|crime)",
    r"\bhow\s+to\s+(evade|escape|avoid)\s+(police|law|arrest|prosecution)",
    r"\bhow\s+to\s+(forge|fake|fabricate)\s+(documents?|evidence|id|identity)",
    r"\bhow\s+to\s+(hack|break\s+into)",
    r"\bhow\s+to\s+(bribe|corrupt)",
    r"\bhelp\s+me\s+(kill|attack|assault|rob|steal)",
]

PERSONAL_ADVICE_PATTERNS = [
    r"\bshould\s+i\s+(file|go\s+to\s+court|sue|divorce|arrest)",
    r"\bwill\s+i\s+(win|go\s+to\s+jail|be\s+arrested|be\s+convicted)",
    r"\bmy\s+case\b.*\bwhat\s+should",
    r"\badvise\s+me\s+on\s+my\s+specific",
]

STANDARD_DISCLAIMER = (
    "⚠️ **Disclaimer**: This information is for general educational purposes only "
    "and does NOT constitute legal advice. For specific legal matters, please consult "
    "a qualified advocate or contact your nearest Legal Services Authority. "
    "You can also reach **Tele-Law** at 1516 for free telephonic legal guidance."
)

HARMFUL_RESPONSE = (
    "🚫 I cannot assist with this request. This query appears to seek information "
    "about illegal activities. If you are in danger or need help, please contact:\n"
    "- **Police**: 100\n"
    "- **Women Helpline**: 181\n"
    "- **Legal Aid**: 1516 (Tele-Law)"
)

PERSONAL_ADVICE_RESPONSE = (
    "⚖️ I can provide general legal information, but I cannot give specific legal advice "
    "for your personal situation. For personalized guidance, please:\n"
    "- Contact a qualified advocate\n"
    "- Visit your nearest **District Legal Services Authority (DLSA)**\n"
    "- Call **Tele-Law**: 1516 for free consultation\n"
    "- Visit **nyayabandhu.in** for pro-bono lawyer matching"
)


class GuardrailResult:
    """Result of guardrail check."""
    def __init__(self, safe: bool, category: str, message: Optional[str] = None):
        self.safe = safe
        self.category = category  # SAFE, HARMFUL, PERSONAL_ADVICE, OUT_OF_SCOPE
        self.message = message


def check(query: str) -> GuardrailResult:
    """
    Run guardrail checks on a user query.

    Returns
    -------
    GuardrailResult with safe=True if query is okay, or safe=False with a redirect message.
    """
    query_lower = query.lower().strip()

    # Check for harmful content
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning("GUARDRAIL: Harmful query detected")
            return GuardrailResult(safe=False, category="HARMFUL", message=HARMFUL_RESPONSE)

    # Check for personal legal advice requests
    for pattern in PERSONAL_ADVICE_PATTERNS:
        if re.search(pattern, query_lower):
            logger.info("GUARDRAIL: Personal advice query detected")
            return GuardrailResult(safe=False, category="PERSONAL_ADVICE", message=PERSONAL_ADVICE_RESPONSE)

    return GuardrailResult(safe=True, category="SAFE")


def get_disclaimer() -> str:
    """Return the standard legal disclaimer to append to all responses."""
    return STANDARD_DISCLAIMER
