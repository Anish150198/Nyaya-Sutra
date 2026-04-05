"""
Agentic tool: performs similarity search using the configured vector DB
(ChromaDB, FAISS, or Databricks Vector Search) for legal document retrieval.
"""

import logging
from typing import Optional

from rag.vector_db import search_acts, get_section, get_available_acts

logger = logging.getLogger(__name__)


def search_legal_docs(
    query: str,
    act_filter: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Tool interface for the orchestrator to call legal vector search.
    
    Uses the configured vector DB backend (ChromaDB, FAISS, or Databricks VS).

    Returns
    -------
    dict with keys: results (list[dict]), context_str (formatted for prompt)
    """
    # Determine which acts to search
    if act_filter and act_filter.upper() != "ALL":
        acts_to_search = [act_filter.upper()]
    else:
        acts_to_search = get_available_acts()
    
    if not acts_to_search:
        logger.warning("No vector DB acts available")
        return {"results": [], "context_str": "No legal data available."}
    
    # Get query embedding (using the same embedder as during indexing)
    from models.embeddings.embedder import embed_query
    query_embedding = embed_query(query)
    
    # Search across acts
    results = search_acts(acts_to_search, query_embedding, top_k)
    
    # Format results for the prompt
    context_str = format_legal_context(results)
    return {"results": results, "context_str": context_str}


def search_scheme_docs(query: str, top_k: int = 5) -> dict:
    """
    Tool interface for scheme document search.
    NOTE: Scheme vector index is not yet implemented — returns empty results.
    The LLM will answer from training knowledge about government schemes.

    Returns
    -------
    dict with keys: results (list[dict]), context_str (formatted for prompt)
    """
    logger.info("Scheme search called (no scheme vector index yet): %s", query[:50])
    return {"results": [], "context_str": "No scheme data indexed yet."}


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


def format_legal_context(results: list[dict]) -> str:
    """
    Format legal search results into a readable context string for the LLM.
    
    Parameters
    ----------
    results : list[dict]
        List of search results from vector_db.search_* functions
        
    Returns
    -------
    str
        Formatted context with citations
    """
    if not results:
        return "No relevant legal provisions found."
    
    context_parts = []
    for r in results:
        act = r.get("act", r.get("act_code", "Unknown"))
        section = r.get("section_number", r.get("section", "Unknown"))
        title = r.get("title", "")
        text = r.get("text", r.get("chunk_text", ""))
        score = r.get("score", 0)
        
        context_parts.append(
            f"**{act} §{section} - {title}** (relevance: {score:.3f})\n{text}"
        )
    
    return "\n\n".join(context_parts)


def lookup_ipc_bns_mapping(ipc_section: str, mapping_data: list[dict]) -> list[dict]:
    """
    Look up BNS equivalents for an IPC section number from the mapping table.
    This is a stub that returns an empty list - the actual implementation
    would use the mapping data to find corresponding BNS sections.
    
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
    # Stub implementation - in a real scenario, this would search mapping_data
    # For now, return empty list as we don't have the mapping data loaded
    return []
