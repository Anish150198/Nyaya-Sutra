"""
Agentic tool: performs similarity search using the configured vector DB
(ChromaDB or Databricks Vector Search) for legal document retrieval.
NO FAISS dependency.
"""

import logging
from typing import Optional

from rag.retriever import search_legal, search_schemes, format_legal_context, format_scheme_context
from rag.vector_db import search_acts, get_section, get_available_acts

logger = logging.getLogger(__name__)


def search_legal_docs(
    query: str,
    act_filter: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Tool interface for the orchestrator to call legal vector search.

    Returns
    -------
    dict with keys: results (list[dict]), context_str (formatted for prompt)
    """
    results = search_legal(query, top_k=top_k, act_filter=act_filter)
    context_str = format_legal_context(results)
    return {"results": results, "context_str": context_str}


def search_scheme_docs(query: str, top_k: int = 5) -> dict:
    """
    Tool interface for scheme search.
    Returns empty results (no scheme vector index in this version).
    """
    results = search_schemes(query, top_k=top_k)
    context_str = format_scheme_context(results)
    return {"results": results, "context_str": context_str}


def get_legal_section(act_code: str, section_number: str) -> dict:
    """
    Get all chunks for a specific section from the vector DB.

    Returns
    -------
    dict with keys: results (list[dict]), context_str (formatted for prompt)
    """
    results = get_section(act_code.upper(), section_number)
    context_str = format_legal_context(results)
    return {"results": results, "context_str": context_str}


def lookup_ipc_bns_mapping(ipc_section: str, mapping_data: list[dict]) -> list[dict]:
    """
    Look up BNS equivalents for an IPC section number from the mapping table.

    Parameters
    ----------
    ipc_section : str
        IPC section number (e.g., "302")
    mapping_data : list[dict]
        IPC→BNS mapping data

    Returns
    -------
    list[dict]
        List of matching BNS sections with their mappings
    """
    matches = [
        row for row in mapping_data
        if str(row.get("ipc_section", "")).strip() == ipc_section.strip()
    ]
    if not matches:
        logger.info("No IPC→BNS mapping found for IPC %s", ipc_section)
    return matches
