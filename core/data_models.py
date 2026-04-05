"""
Pydantic data models shared across the entire pipeline.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Intent(str, Enum):
    LEGAL = "LEGAL"
    WELFARE = "WELFARE"
    MIXED = "MIXED"
    GENERIC = "GENERIC"


class Persona(str, Enum):
    CITIZEN = "citizen"
    JUNIOR_LAWYER = "junior_lawyer"


class ActFilter(str, Enum):
    ALL = "ALL"
    BNS = "BNS"
    BNSS = "BNSS"
    BSA = "BSA"
    CONST = "CONST"


class UserQuery(BaseModel):
    """Incoming query from the UI."""
    text: str
    user_lang: str = "en"
    persona: Persona = Persona.CITIZEN
    act_filter: ActFilter = ActFilter.ALL
    compare_models: bool = False
    user_profile: Optional[UserProfile] = None


class UserProfile(BaseModel):
    """Demographics collected by the Scheme Wizard."""
    age: Optional[int] = None
    gender: Optional[str] = None
    state_code: Optional[str] = None
    income: Optional[float] = None
    caste: Optional[str] = None
    occupation: Optional[str] = None
    disability: Optional[bool] = False


class Citation(BaseModel):
    """A reference to a specific legal section or scheme."""
    code: Optional[str] = None          # e.g. "BNS", "BNSS", "scheme"
    section_no: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    similarity_score: Optional[float] = None


class LegalAnswer(BaseModel):
    """Structured output from the Legal Agent."""
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    ipc_bns_mapping: Optional[dict] = None
    confidence: str = "medium"          # high / medium / low
    model_id: str = "param1"


class WelfareAnswer(BaseModel):
    """Structured output from the Welfare Agent."""
    answer_text: str
    schemes: list[dict] = Field(default_factory=list)
    legal_aid_eligible: bool = False
    legal_aid_info: Optional[str] = None
    confidence: str = "medium"
    model_id: str = "param1"


class OrchestratorResponse(BaseModel):
    """Final response returned to the UI."""
    answer_text: str
    intent: Intent
    citations: list[Citation] = Field(default_factory=list)
    schemes: list[dict] = Field(default_factory=list)
    confidence: str = "medium"
    model_ids_used: list[str] = Field(default_factory=list)
    comparison: Optional[dict] = None   # populated when compare_models=True
    disclaimer: Optional[str] = None
    original_lang: str = "en"
