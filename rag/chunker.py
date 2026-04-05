"""
Load and chunk law data from JSON/TXT files per act.
Produces a list of dicts: {chunk_id, act, section_number, title, text, source_url}
"""

import json
import logging
from pathlib import Path
from typing import Optional

from core.config import ACTS, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def load_json_sections(directory: Path) -> list[dict]:
    """Load all .json section files from a directory (recursive)."""
    sections = []
    json_files = sorted(directory.rglob("*.json"))
    for fp in json_files:
        try:
            with open(fp) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue  # skip manifest.json or non-section files
            sec_num = data.get("section_number", "") or ""
            sections.append({
                "section_number": str(sec_num).strip() if sec_num else "",
                "title": data.get("title", fp.stem),
                "content": data.get("content", ""),
                "source_url": data.get("source_url", ""),
                "_filename": fp.stem,  # fallback for unique ID
            })
        except Exception as e:
            logger.warning("Failed to load %s: %s", fp, e)
    return sections


def load_txt_sections(directory: Path) -> list[dict]:
    """Load all .txt section files from a directory (recursive)."""
    sections = []
    txt_files = sorted(directory.rglob("*.txt"))
    for fp in txt_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
            parts = fp.stem.split("_", 1)
            sec_num = parts[0] if parts[0].isdigit() else ""
            title = parts[1].replace("-", " ").title() if len(parts) > 1 else fp.stem
            sections.append({
                "section_number": sec_num,
                "title": title,
                "content": text.strip(),
                "source_url": "",
            })
        except Exception as e:
            logger.warning("Failed to load %s: %s", fp, e)
    return sections


def load_act_sections(act_code: str) -> list[dict]:
    """
    Load all sections for a given act code (BNS, BNSS, BSA).
    Tries JSON first, falls back to TXT.
    """
    act_info = ACTS.get(act_code)
    if not act_info:
        raise ValueError(f"Unknown act: {act_code}. Valid: {list(ACTS.keys())}")

    data_dir = act_info["data_dir"]
    if data_dir is None:
        logger.warning("data_dir is None for act %s (Databricks mode?)", act_code)
        return []
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.warning("Data directory not found: %s", data_dir)
        return []

    sections = load_json_sections(data_dir)
    if not sections:
        sections = load_txt_sections(data_dir)

    logger.info("Loaded %d sections for %s from %s", len(sections), act_code, data_dir)
    return sections


def chunk_act(act_code: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> list[dict]:
    """
    Load and chunk all sections for an act.

    Returns list of dicts:
        {chunk_id, act, section_number, title, text, source_url}
    """
    cs = chunk_size or CHUNK_SIZE
    ov = overlap or CHUNK_OVERLAP

    sections = load_act_sections(act_code)
    chunks = []

    seen_ids = set()
    for idx, sec in enumerate(sections):
        content = sec["content"]
        if not content.strip():
            continue
        # Use section_number if available, else fallback to file stem or index
        sec_id = sec["section_number"]
        if not sec_id or sec_id == "None":
            sec_id = sec.get("_filename", f"idx{idx}")
        text_chunks = _split_text(content, cs, ov)
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{act_code}_s{sec_id}_{i}"
            # Ensure uniqueness
            while chunk_id in seen_ids:
                chunk_id += "_dup"
            seen_ids.add(chunk_id)
            chunks.append({
                "chunk_id": chunk_id,
                "act": act_code,
                "section_number": sec["section_number"] or sec_id,
                "title": sec["title"],
                "text": chunk_text,
                "source_url": sec.get("source_url", ""),
            })

    logger.info("Chunked %s: %d sections -> %d chunks", act_code, len(sections), len(chunks))
    return chunks


def chunk_all_acts() -> dict[str, list[dict]]:
    """Chunk all registered acts. Returns {act_code: [chunks]}."""
    result = {}
    for act_code in ACTS:
        result[act_code] = chunk_act(act_code)
    return result
