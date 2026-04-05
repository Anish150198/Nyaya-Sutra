"""
Silver Layer — Read Bronze Delta, chunk text, write Silver Delta table.

Schema:
  chunk_id | act_code | section_number | title | chunk_text | chunk_index | total_chunks

Usage:
  python -m pipelines.silver_chunk          # local (uses deltalake)
  # On Databricks: run notebooks/02_silver_chunk
"""

import logging
import sys
from pathlib import Path

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BRONZE_DELTA_PATH, SILVER_DELTA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("act_code", pa.string()),
    ("section_number", pa.string()),
    ("title", pa.string()),
    ("chunk_text", pa.string()),
    ("chunk_index", pa.int32()),
    ("total_chunks", pa.int32()),
])


def _split_text(text: str, max_len: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_len
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_len - overlap
    return chunks


def chunk_bronze() -> int:
    """Read Bronze Delta, chunk all sections, write Silver Delta."""
    logger.info("Reading Bronze Delta: %s", BRONZE_DELTA_PATH)
    dt = DeltaTable(BRONZE_DELTA_PATH)
    bronze_df = dt.to_pyarrow_table().to_pylist()
    logger.info("Bronze rows: %d", len(bronze_df))

    silver_rows = []
    seen_ids = set()

    for row in bronze_df:
        act_code = row["act_code"]
        sec_num = row["section_number"]
        title = row["title"]
        content = row["content"]

        if not content or not content.strip():
            continue

        chunks = _split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            # Build unique chunk_id
            sec_id = sec_num if sec_num else "x"
            chunk_id = f"{act_code}_s{sec_id}_{i}"
            while chunk_id in seen_ids:
                chunk_id += "_dup"
            seen_ids.add(chunk_id)

            silver_rows.append({
                "chunk_id": chunk_id,
                "act_code": act_code,
                "section_number": sec_num,
                "title": title,
                "chunk_text": chunk_text,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    if not silver_rows:
        logger.error("No chunks produced!")
        return 0

    table = pa.Table.from_pylist(silver_rows, schema=SCHEMA)

    logger.info("Writing %d chunks to Silver Delta: %s", len(silver_rows), SILVER_DELTA_PATH)
    write_deltalake(
        SILVER_DELTA_PATH,
        table,
        mode="overwrite",
    )

    dt_silver = DeltaTable(SILVER_DELTA_PATH)
    logger.info("✅ Silver Delta table: %d chunks, version %d", len(dt_silver.to_pyarrow_table()), dt_silver.version())

    # Per-act stats
    from collections import Counter
    counts = Counter(r["act_code"] for r in silver_rows)
    for act_code, count in sorted(counts.items()):
        logger.info("  %s: %d chunks", act_code, count)

    return len(silver_rows)


if __name__ == "__main__":
    chunk_bronze()
