"""
Welfare Agent – handles government scheme recommendations and legal aid eligibility.
Supports both conversational queries and wizard-based demographic filtering.
"""

import logging
from typing import Optional

from core.data_models import WelfareAnswer, UserProfile
from rag.pipeline import run_welfare_rag
from agents.tools.sql_welfare_tool import filter_schemes_local
from agents.tools.vector_legal_tool import search_scheme_docs

logger = logging.getLogger(__name__)

# Cached legal aid rules (loaded once)
_legal_aid_rules: list[dict] | None = None


def load_legal_aid_rules(path: Optional[str] = None) -> list[dict]:
    """Load legal aid eligibility rules from file."""
    global _legal_aid_rules
    if _legal_aid_rules is not None:
        return _legal_aid_rules

    import json
    from pathlib import Path

    if path is None:
        default = Path(__file__).resolve().parent.parent / "data" / "bronze" / "legal_aid" / "legal_aid_rules.json"
        path = str(default)

    p = Path(path)
    if p.exists():
        with open(p) as f:
            _legal_aid_rules = json.load(f)
        logger.info("Loaded %d legal aid rules", len(_legal_aid_rules))
    else:
        logger.warning("Legal aid rules not found at %s", path)
        _legal_aid_rules = []
    return _legal_aid_rules


def check_legal_aid_eligibility(profile: Optional[UserProfile]) -> tuple[bool, str]:
    """
    Check if a user is eligible for free legal aid based on NALSA guidelines.

    Returns
    -------
    (eligible: bool, info: str)
    """
    if profile is None:
        return False, ""

    rules = load_legal_aid_rules()
    if not rules:
        # Fallback: basic NALSA eligibility criteria
        eligible_categories = [
            "sc", "st", "obc", "minority", "woman", "child",
            "disabled", "industrial_workman", "victim_trafficking",
        ]
        reasons = []

        if profile.caste and profile.caste.lower() in eligible_categories:
            reasons.append(f"Belongs to {profile.caste} category")
        if profile.gender and profile.gender.lower() == "female":
            reasons.append("Woman (eligible under Section 12 of Legal Services Authorities Act)")
        if profile.income and profile.income < 300000:
            reasons.append(f"Annual income below ₹3,00,000")
        if profile.disability:
            reasons.append("Person with disability")

        if reasons:
            info = (
                "You may be eligible for FREE legal aid through:\n"
                "- **Tele-Law**: Call 1516 for free telephonic legal advice\n"
                "- **Nyaya Bandhu**: Pro-bono lawyer matching at nyayabandhu.in\n"
                "- **NALSA**: Visit your nearest District Legal Services Authority\n\n"
                f"Reasons: {'; '.join(reasons)}"
            )
            return True, info

    return False, ""


def handle(
    query: str,
    user_profile: Optional[UserProfile] = None,
    canonical_lang: str = "en",
    model_id: Optional[str] = None,
) -> WelfareAnswer:
    """
    Main entry point for the Welfare Agent.

    Parameters
    ----------
    query : str
        Canonical (English) question text.
    user_profile : UserProfile or None
        Demographics from the Scheme Wizard.
    canonical_lang : str
        Language of the canonical query.
    model_id : str or None
        Override model selection.

    Returns
    -------
    WelfareAnswer
    """
    logger.info("Welfare agent handling query (profile=%s)", "provided" if user_profile else "none")

    # Build user profile string for the prompt
    profile_str = "Not provided"
    if user_profile:
        parts = []
        if user_profile.age:
            parts.append(f"Age: {user_profile.age}")
        if user_profile.gender:
            parts.append(f"Gender: {user_profile.gender}")
        if user_profile.state_code:
            parts.append(f"State: {user_profile.state_code}")
        if user_profile.income:
            parts.append(f"Income: ₹{user_profile.income:,.0f}")
        if user_profile.caste:
            parts.append(f"Category: {user_profile.caste}")
        if user_profile.occupation:
            parts.append(f"Occupation: {user_profile.occupation}")
        if user_profile.disability:
            parts.append("Disability: Yes")
        profile_str = ", ".join(parts) if parts else "Not provided"

    # Run RAG pipeline for schemes
    rag_result = run_welfare_rag(
        question=query,
        user_profile_str=profile_str,
        canonical_lang=canonical_lang,
        model_id=model_id,
    )

    # Check legal aid eligibility
    eligible, legal_aid_info = check_legal_aid_eligibility(user_profile)

    return WelfareAnswer(
        answer_text=rag_result["answer_text"],
        schemes=rag_result["schemes"],
        legal_aid_eligible=eligible,
        legal_aid_info=legal_aid_info if eligible else None,
        model_id=rag_result.get("model_id", "param1"),
    )
