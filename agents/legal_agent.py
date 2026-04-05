"""
Legal Agent – RAG synthesizer for BNS/BNSS/BSA/Constitution queries.
Handles act-specific filtering and IPC→BNS mapping lookups.
"""

import logging
import re
from typing import Optional

from core.data_models import LegalAnswer, Citation
from rag.pipeline import run_legal_rag
from agents.tools.vector_legal_tool import search_legal_docs, get_legal_section, lookup_ipc_bns_mapping

logger = logging.getLogger(__name__)

# Cached IPC→BNS mapping data (loaded once)
_ipc_bns_map: list[dict] | None = None


def load_ipc_bns_map(path: Optional[str] = None) -> list[dict]:
    """Load IPC→BNS mapping from a JSON/CSV file or Delta table."""
    global _ipc_bns_map
    if _ipc_bns_map is not None:
        return _ipc_bns_map

    import json
    from pathlib import Path

    if path is None:
        # Default local path
        default = Path(__file__).resolve().parent.parent / "data" / "bronze" / "ipc_bns_mapping" / "ipc_bns_map.json"
        path = str(default)

    p = Path(path)
    if p.exists():
        with open(p) as f:
            _ipc_bns_map = json.load(f)
        logger.info("Loaded %d IPC→BNS mapping entries", len(_ipc_bns_map))
    else:
        logger.warning("IPC→BNS mapping file not found at %s", path)
        _ipc_bns_map = []
    return _ipc_bns_map


def _extract_ipc_section(query: str) -> Optional[str]:
    """Extract an IPC section number from the query text."""
    match = re.search(r'\bipc\s*(?:section\s*)?(\d+[a-zA-Z]?)', query.lower())
    if match:
        return match.group(1)
    return None


def handle(
    query: str,
    persona: str = "citizen",
    act_filter: str = "ALL",
    canonical_lang: str = "en",
    compare_models: bool = False,
    model_id: Optional[str] = None,
) -> LegalAnswer:
    """
    Main entry point for the Legal Agent.

    Parameters
    ----------
    query : str
        Canonical (English) question text.
    persona : str
        "citizen" or "junior_lawyer".
    act_filter : str
        Act code filter or "ALL".
    canonical_lang : str
        Language of the canonical query.
    compare_models : bool
        If True, comparison is handled upstream by orchestrator.
    model_id : str or None
        Override model selection.

    Returns
    -------
    LegalAnswer
    """
    logger.info("Legal agent handling query (persona=%s, act_filter=%s)", persona, act_filter)

    # Check for IPC→BNS mapping queries
    ipc_section = _extract_ipc_section(query)
    ipc_bns_mapping = None

    if ipc_section:
        mapping_data = load_ipc_bns_map()
        matches = lookup_ipc_bns_mapping(ipc_section, mapping_data)
        if matches:
            ipc_bns_mapping = {
                "ipc_section": ipc_section,
                "bns_mappings": matches,
            }
            logger.info("Found IPC %s → BNS mapping: %d entries", ipc_section, len(matches))

    # Run RAG pipeline
    answer = run_legal_rag(
        question=query,
        persona=persona,
        act_filter=act_filter,
        canonical_lang=canonical_lang,
        model_id=model_id,
    )

    # Attach IPC→BNS mapping if found
    if ipc_bns_mapping:
        answer.ipc_bns_mapping = ipc_bns_mapping

    return answer
