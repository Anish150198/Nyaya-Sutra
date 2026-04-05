"""
Orchestrator Agent – core routing agent that classifies intent and
delegates to specific tools/agents. This is the single entry point
called by the UI layer.
"""

import logging
from typing import Optional

from core.data_models import (
    UserQuery, OrchestratorResponse, Intent, LegalAnswer, WelfareAnswer,
)
from models.nlp_classifier.intent_classifier import classify
from agents import translation_agent, legal_agent, welfare_agent, guardrail_agent
from models.llm.comparator import run_dual_models

logger = logging.getLogger(__name__)


def handle(query: UserQuery) -> OrchestratorResponse:
    """
    Main entry point for every user query.

    Flow
    ----
    1. Guardrail check
    2. Translate to canonical language (if needed)
    3. Classify intent
    4. Route to legal / welfare / both
    5. Optionally run multi-LLM comparison
    6. Translate answer back to user language
    7. Return structured OrchestratorResponse
    """

    # ── 1. Guardrail check ──────────────────────────────────────────────
    guard_result = guardrail_agent.check(query.text)
    if not guard_result.safe:
        return OrchestratorResponse(
            answer_text=guard_result.message or "Query blocked.",
            intent=Intent.GENERIC,
            confidence="high",
            disclaimer=guardrail_agent.get_disclaimer(),
            original_lang=query.user_lang,
        )

    # ── 2. Translation to canonical ─────────────────────────────────────
    if query.user_lang == "en":
        canonical_text = query.text
        canonical_lang = "en"
    else:
        canonical_text, canonical_lang = translation_agent.handle_to_canonical(
            query.text, src_lang=query.user_lang
        )
    logger.info("User Query: '%s' (lang=%s)", query.text, query.user_lang)
    logger.info("Canonical (English) Query: '%s'", canonical_text)

    # ── 3. Intent classification ────────────────────────────────────────
    intent = classify(canonical_text)
    logger.info("Classified intent: %s", intent)

    # ── 4. Route to agents ──────────────────────────────────────────────
    legal_answer: Optional[LegalAnswer] = None
    welfare_answer: Optional[WelfareAnswer] = None

    if intent in (Intent.LEGAL, Intent.MIXED):
        legal_answer = legal_agent.handle(
            query=canonical_text,
            persona=query.persona.value,
            act_filter=query.act_filter.value,
            canonical_lang=canonical_lang,
        )

    if intent in (Intent.WELFARE, Intent.MIXED):
        welfare_answer = welfare_agent.handle(
            query=canonical_text,
            user_profile=query.user_profile,
            canonical_lang=canonical_lang,
        )

    if intent == Intent.GENERIC:
        # Fallback: try legal RAG with relaxed filters
        legal_answer = legal_agent.handle(
            query=canonical_text,
            persona=query.persona.value,
            act_filter="ALL",
            canonical_lang=canonical_lang,
        )

    # ── 5. Multi-LLM comparison (optional) ──────────────────────────────
    comparison = None
    if query.compare_models and legal_answer:
        from rag.retriever import search_legal, format_legal_context
        from rag.prompts import build_legal_prompt

        results = search_legal(canonical_text, act_filter=query.act_filter.value)
        context = format_legal_context(results)
        prompt = build_legal_prompt(canonical_text, context, query.persona.value)
        comparison = run_dual_models(prompt, models=["openai"])


    # ── 6. Assemble response ────────────────────────────────────────────
    answer_parts = []
    citations = []
    schemes = []
    model_ids = []

    if legal_answer:
        answer_parts.append(legal_answer.answer_text)
        citations = legal_answer.citations
        model_ids.append(legal_answer.model_id)

    if welfare_answer:
        if answer_parts:
            answer_parts.append("\n\n---\n**Welfare Schemes & Legal Aid:**\n")
        answer_parts.append(welfare_answer.answer_text)
        schemes = welfare_answer.schemes
        model_ids.append(welfare_answer.model_id)

        if welfare_answer.legal_aid_eligible and welfare_answer.legal_aid_info:
            answer_parts.append(f"\n\n{welfare_answer.legal_aid_info}")

    answer_text = "\n".join(answer_parts) if answer_parts else "I couldn't find relevant information for your query."

    # ── 7. Translate back to user language ──────────────────────────────
    if query.user_lang not in ("en", canonical_lang):
        answer_text = translation_agent.handle_from_canonical(answer_text, query.user_lang)

    # Confidence
    confidence = "medium"
    if legal_answer:
        confidence = legal_answer.confidence
    elif welfare_answer:
        confidence = welfare_answer.confidence

    return OrchestratorResponse(
        answer_text=answer_text,
        intent=intent,
        citations=citations,
        schemes=schemes,
        confidence=confidence,
        model_ids_used=list(set(model_ids)),
        comparison=comparison,
        disclaimer=guardrail_agent.get_disclaimer(),
        original_lang=query.user_lang,
    )
