"""
RAG retriever for Nyaya-Sahayak.
Wraps the unified vector_db abstraction (ChromaDB local/remote or Databricks VS).
NO FAISS dependency.

This replaces the old FAISS-only retriever.py.
"""

import logging
from typing import Optional

import numpy as np

from core.config import TOP_K_RETRIEVAL
from models.embeddings.embedder import embed_query
from rag.vector_db import search_acts, get_available_acts, search_act

logger = logging.getLogger(__name__)


def search_legal(
    query: str,
    top_k: int = TOP_K_RETRIEVAL,
    act_filter: Optional[str] = None,
) -> list[dict]:
    """
    Search the legal vector store (ChromaDB or Databricks VS).

    Parameters
    ----------
    query : str
        The user query in canonical language.
    top_k : int
        Number of results to return.
    act_filter : str or None
        If set (e.g., "BNS"), only return chunks for that act.

    Returns
    -------
    list[dict]  Each dict has keys: chunk_id, act/act_code, section_number, title, text, score
    """
    query_vec = embed_query(query)

    if act_filter and act_filter.upper() != "ALL":
        acts_to_search = [act_filter.upper()]
    else:
        acts_to_search = get_available_acts()

    if not acts_to_search:
        logger.warning("No vector DB acts available, returning empty results")
        return []

    results = search_acts(acts_to_search, query_vec, top_k)
    return results


def search_schemes(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Search scheme documents.
    NOTE: Scheme vector data is not indexed in this version.
    Returns an empty list — the LLM will answer from training knowledge.
    """
    logger.info("Scheme search called (no scheme vector index in this version): query=%s", query[:50])
    return []


def format_legal_context(results: list[dict]) -> str:
    """Format retrieved legal chunks into a context string for the prompt."""
    if not results:
        return "No relevant legal sections found."

    parts = []
    for i, r in enumerate(results, 1):
        act = r.get("act", r.get("act_code", ""))
        section = r.get("section_number", r.get("section_no", ""))
        title = r.get("title", "")
        text = r.get("text", r.get("chunk_text", ""))
        parts.append(
            f"[{i}] {act} Section {section}"
            f"{(' – ' + title) if title else ''}\n"
            f"{text}"
        )
    return "\n\n".join(parts)


def format_scheme_context(results: list[dict]) -> str:
    """Format retrieved schemes into a context string for the prompt."""
    if not results:
        return "No relevant schemes found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r.get('name', 'Unknown Scheme')}\n"
            f"Description: {r.get('description', 'N/A')}\n"
            f"Eligibility: {r.get('eligibility', r.get('eligibility_text', 'N/A'))}\n"
            f"Benefits: {r.get('benefits', r.get('benefits_text', 'N/A'))}\n"
            f"Apply: {r.get('apply_url', 'N/A')}"
        )
    return "\n\n".join(parts)
