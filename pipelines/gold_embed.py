"""
Gold Layer — Read Silver Delta, embed chunks, write Gold Delta + upsert to vector DB.

Schema:
  chunk_id | act_code | section_number | title | chunk_text | embedding | embedded_at

Usage:
  python -m pipelines.gold_embed              # local (uses deltalake)
  # On Databricks: run notebooks/03_gold_embed
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import SILVER_DELTA_PATH, GOLD_DELTA_PATH, ACTS, VECTOR_DB
from models.embeddings.embedder import embed_texts
from rag.vector_db import upsert_act

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA = pa.schema([
    ("chunk_id", pa.string()),
    ("act_code", pa.string()),
    ("section_number", pa.string()),
    ("title", pa.string()),
    ("chunk_text", pa.string()),
    ("embedding", pa.list_(pa.float32())),
    ("embedded_at", pa.string()),
])


def embed_and_store() -> int:
    """Read Silver, embed all chunks, write Gold Delta, upsert to vector DB."""
    logger.info("Reading Silver Delta: %s", SILVER_DELTA_PATH)
    dt = DeltaTable(SILVER_DELTA_PATH)
    silver_rows = dt.to_pyarrow_table().to_pylist()
    logger.info("Silver chunks: %d", len(silver_rows))

    if not silver_rows:
        logger.error("No chunks to embed!")
        return 0

    # Group by act_code for batch processing
    acts_data = {}
    for row in silver_rows:
        acts_data.setdefault(row["act_code"], []).append(row)

    gold_rows = []
    now = datetime.now(timezone.utc).isoformat()

    for act_code in ACTS:
        if act_code not in acts_data:
            logger.warning("No Silver chunks for %s, skipping", act_code)
            continue

        rows = acts_data[act_code]
        texts = [r["chunk_text"] for r in rows]
        logger.info("Embedding %d chunks for %s...", len(texts), act_code)

        # Embed in one call (batched internally)
        embeddings = embed_texts(texts, show_progress=True)

        # Build Gold rows
        for i, row in enumerate(rows):
            gold_rows.append({
                "chunk_id": row["chunk_id"],
                "act_code": row["act_code"],
                "section_number": row["section_number"],
                "title": row["title"],
                "chunk_text": row["chunk_text"],
                "embedding": embeddings[i].tolist(),
                "embedded_at": now,
            })

        # Upsert to vector DB
        logger.info("Upserting %d vectors to %s for %s...", len(rows), VECTOR_DB, act_code)
        chunks_for_db = [
            {
                "chunk_id": r["chunk_id"],
                "act": act_code,
                "section_number": r["section_number"],
                "title": r["title"],
                "text": r["chunk_text"],
            }
            for r in rows
        ]
        upsert_act(act_code, chunks_for_db, embeddings)
        logger.info("✓ %s: %d vectors upserted to %s", act_code, len(rows), VECTOR_DB)

    if not gold_rows:
        logger.error("No Gold rows produced!")
        return 0

    # Write Gold Delta table
    table = pa.Table.from_pylist(gold_rows, schema=SCHEMA)
    logger.info("Writing %d rows to Gold Delta: %s", len(gold_rows), GOLD_DELTA_PATH)
    write_deltalake(
        GOLD_DELTA_PATH,
        table,
        mode="overwrite",
    )

    dt_gold = DeltaTable(GOLD_DELTA_PATH)
    logger.info("✅ Gold Delta table: %d rows, version %d", len(dt_gold.to_pyarrow_table()), dt_gold.version())

    return len(gold_rows)


if __name__ == "__main__":
    embed_and_store()
