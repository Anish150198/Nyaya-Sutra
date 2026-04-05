"""
End-to-end RAG pipeline: retrieve → build prompt → generate → structure answer.
"""

import logging
import re
from typing import Optional

from core.data_models import LegalAnswer, Citation
from models.llm.router import select_model, run_model
from rag.vector_db import search_acts, get_available_acts, get_section
from rag.prompts import build_legal_prompt, build_welfare_prompt

logger = logging.getLogger(__name__)

# Instead of a single regex, we will use a function with multiple patterns
_ACT_ALIAS = {
    # Acronyms
    "bns": "BNS", "bnss": "BNSS", "bsa": "BSA",
    "ipc": "IPC",
    "crpc": "BNSS",
    "iea": "BSA",
    "constitution": "CONSTITUTION",
    "constitution of india": "CONSTITUTION",
    "acts": "OTHER_ACTS",
    
    # Full names (handles translation expansion)
    "bharatiya nyaya sanhita": "BNS",
    "bharatiya nagarik suraksha sanhita": "BNSS",
    "bharatiya sakshya adhiniyam": "BSA",
    "indian penal code": "IPC",
    "code of criminal procedure": "BNSS",
    "indian evidence act": "BSA",
}



def _extract_section_refs(query: str) -> list[tuple[str, str]]:
    """
    Extract (act_code, section_number) pairs from the query.
    Handles multiple natural language formulations.
    """
    # Build a massive regex for all act names and aliases
    all_act_terms = sorted(_ACT_ALIAS.keys(), key=len, reverse=True)
    act_pattern = rf"({'|'.join(re.escape(t) for t in all_act_terms)})"
    
    sec_pattern = r'(?:section|secs?|§|s\.|dhara|dhara-)'
    num_pattern = r'(\d+[a-zA-Z]?)'
    
    # Optional connecting words
    conn_pattern = r'(?:(?:of|under|in|from)(?:\s+the)?\s+)?'
    
    # Patterns to match across the string:
    # 1. "BNS section 300"
    p1 = re.compile(rf'\b{act_pattern}\s+(?:{sec_pattern}\s+)?{num_pattern}\b', re.IGNORECASE)
    # 2. "section 300 of the BNS"
    p2 = re.compile(rf'\b(?:{sec_pattern}\s+)?{num_pattern}\s+{conn_pattern}{act_pattern}\b', re.IGNORECASE)
    # 3. "BNS 300 section"
    p3 = re.compile(rf'\b{act_pattern}\s+{num_pattern}\s+{sec_pattern}\b', re.IGNORECASE)

    refs = []
    seen = set()

    def add_ref(act_raw, num_raw):
        act_raw = (act_raw or "").lower().strip()
        act_code = _ACT_ALIAS.get(act_raw, act_raw.upper()) if act_raw else ""
        if (act_code, num_raw) not in seen:
            seen.add((act_code, num_raw))
            refs.append((act_code, num_raw))

    for m in p1.finditer(query):
        add_ref(m.group(1), m.group(2))
    for m in p2.finditer(query):
        add_ref(m.group(2), m.group(1))
    for m in p3.finditer(query):
        add_ref(m.group(1), m.group(2))

    if refs:
        logger.info("Extracted section references: %s", refs)
    return refs


def _lookup_sections(section_refs: list[tuple[str, str]], available_acts: list[str]) -> list[dict]:
    """Look up exact section chunks for each (act, section_num) pair."""
    results = []
    for act_code, sec_num in section_refs:
        if act_code:
            # Specific act requested
            hits = get_section(act_code, sec_num)
            results.extend(hits)
        else:
            # No act specified → search all available acts
            for a in available_acts:
                hits = get_section(a, sec_num)
                results.extend(hits)
    return results


def run_legal_rag(
    question: str,
    persona: str = "citizen",
    act_filter: str = "ALL",
    canonical_lang: str = "en",
    model_id: Optional[str] = None,
) -> LegalAnswer:
    """
    Full legal RAG: retrieve chunks → build prompt → generate answer.

    Strategy:
    1. If query contains explicit section numbers → exact metadata lookup in ChromaDB
    2. Always also do semantic search
    3. Merge & deduplicate results
    """
    from models.embeddings.embedder import embed_query

    # 1. Embed query
    query_embedding = embed_query(question)

    # 2. Determine which acts to search
    if act_filter and act_filter.upper() not in ("ALL", ""):
        available_acts = [act_filter.upper()]
    else:
        available_acts = get_available_acts()

    results = []

    # 3a. Exact section lookup (when query mentions a section number)
    section_refs = _extract_section_refs(question)
    if section_refs:
        logger.info("Section refs detected: %s — doing exact lookup", section_refs)
        exact_hits = _lookup_sections(section_refs, available_acts)
        if exact_hits:
            logger.info("Exact section lookup returned %d chunks", len(exact_hits))
            results.extend(exact_hits)

    # 3b. Semantic search (always run)
    if available_acts:
        sem_results = search_acts(available_acts, query_embedding, top_k=5)
        # Merge: add semantic results that aren't already in results (by chunk_id)
        seen_ids = {r.get("chunk_id") for r in results}
        for r in sem_results:
            if r.get("chunk_id") not in seen_ids:
                results.append(r)
                seen_ids.add(r.get("chunk_id"))
    else:
        logger.warning("No indexed acts found in vector DB — LLM will use training knowledge")

    # 4. Build context string
    if results:
        context = _format_legal_context(results[:8])  # cap at 8 chunks
    else:
        context = (
            "No pre-indexed context available for this query. "
            "Please answer from your training knowledge about Indian law "
            "(BNS - Bharatiya Nyaya Sanhita 2023, BNSS - Bharatiya Nagarik Suraksha Sanhita 2023, "
            "BSA - Bharatiya Sakshya Adhiniyam 2023). "
            "Be specific about section numbers, punishments, and legal provisions."
        )

    # 5. Build prompt
    prompt = build_legal_prompt(question, context, persona)

    # 6. Select model and generate
    if model_id is None:
        model_id = select_model(canonical_lang, persona)
    gen_output = run_model(model_id, prompt)

    # 7. Build citations
    citations = []
    seen_cit = set()
    for r in results:
        act = r.get("act", r.get("act_code", ""))
        section = r.get("section_number", "")
        key = f"{act}_{section}"
        if key in seen_cit:
            continue
        seen_cit.add(key)
        citations.append(Citation(
            code=act,
            section_no=section,
            title=r.get("title", ""),
            snippet=r.get("text", "")[:200],
            similarity_score=r.get("score"),
        ))

    # 8. Confidence
    if results:
        # Exact matches get score 1.0, semantic matches get cosine score
        scores = [r.get("score", 1.0) for r in results]
        max_score = max(scores)
        confidence = "high" if max_score > 0.75 else "medium" if max_score > 0.4 else "low"
    else:
        confidence = "low"

    return LegalAnswer(
        answer_text=gen_output["text"],
        citations=citations,
        confidence=confidence,
        model_id=gen_output.get("model_id", model_id),
    )


def _format_legal_context(results: list[dict]) -> str:
    """Format retrieved legal chunks into a context string for the prompt."""
    if not results:
        return "No relevant legal provisions found."

    parts = []
    for i, r in enumerate(results, 1):
        act = r.get("act", r.get("act_code", ""))
        section = r.get("section_number", "")
        title = r.get("title", "")
        text = r.get("text", r.get("chunk_text", ""))
        score_str = f" (relevance: {r['score']:.3f})" if r.get("score") else ""
        parts.append(
            f"[{i}] {act} §{section}{' – ' + title if title else ''}{score_str}\n{text}"
        )
    return "\n\n".join(parts)


def run_welfare_rag(
    question: str,
    user_profile_str: str = "Not provided",
    canonical_lang: str = "en",
    model_id: Optional[str] = None,
) -> dict:
    """
    Full welfare RAG: build prompt → generate answer.
    Scheme vector index is not yet implemented; LLM answers from training knowledge.
    """
    context = (
        "Government Scheme Information: "
        "Please provide information about relevant Indian government welfare schemes "
        "based on the user profile and query. Include scheme names, eligibility, "
        "benefits, and how to apply."
    )
    prompt = build_welfare_prompt(question, context, user_profile_str)

    if model_id is None:
        model_id = select_model(canonical_lang, "citizen")
    gen_output = run_model(model_id, prompt)

    return {
        "answer_text": gen_output["text"],
        "schemes": [],
        "model_id": gen_output.get("model_id", model_id),
    }
