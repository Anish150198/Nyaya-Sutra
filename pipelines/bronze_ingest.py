"""
Bronze Layer — Ingest raw scraped JSON files into a Delta table.

Reads all JSON files from data/bronze/laws/{BNS,BNSS,BSA}/json/
and writes them into a single Bronze Delta table with schema:

  act_code | section_number | section_id | title | content | source_url | ingested_at

Usage:
  python -m pipelines.bronze_ingest          # local (uses deltalake)
  # On Databricks: run notebooks/01_bronze_ingest
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import ACTS, BRONZE_LAWS_DIR, BRONZE_DELTA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHEMA = pa.schema([
    ("act_code", pa.string()),
    ("section_number", pa.string()),
    ("section_id", pa.string()),
    ("title", pa.string()),
    ("content", pa.string()),
    ("source_url", pa.string()),
    ("ingested_at", pa.string()),
])


def load_json_files(act_code: str) -> list[dict]:
    """Load all JSON section files for one act."""
    json_dir = BRONZE_LAWS_DIR / act_code / "json"
    if not json_dir.exists():
        logger.warning("No JSON directory for %s at %s", act_code, json_dir)
        return []

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for f in sorted(json_dir.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            if not isinstance(data, dict):
                continue
            rows.append({
                "act_code": act_code,
                "section_number": str(data.get("section_number", "")),
                "section_id": str(data.get("section_id", "")),
                "title": data.get("title", ""),
                "content": data.get("content", ""),
                "source_url": data.get("source_url", ""),
                "ingested_at": now,
            })
        except Exception as e:
            logger.error("Failed to read %s: %s", f, e)

    logger.info("Loaded %d sections from %s", len(rows), act_code)
    return rows


def ingest_all() -> int:
    """Ingest all acts into the Bronze Delta table. Returns total row count."""
    all_rows = []
    for act_code in ACTS:
        all_rows.extend(load_json_files(act_code))

    if not all_rows:
        logger.error("No data to ingest!")
        return 0

    # Convert to PyArrow table
    table = pa.Table.from_pylist(all_rows, schema=SCHEMA)

    # Write to Delta (overwrite — full refresh each run)
    logger.info("Writing %d rows to Bronze Delta: %s", len(all_rows), BRONZE_DELTA_PATH)
    write_deltalake(
        BRONZE_DELTA_PATH,
        table,
        mode="overwrite",
    )

    # Verify
    dt = DeltaTable(BRONZE_DELTA_PATH)
    logger.info("✅ Bronze Delta table: %d rows, version %d", len(dt.to_pyarrow_table()), dt.version())

    # Per-act stats
    for act_code in ACTS:
        count = len([r for r in all_rows if r["act_code"] == act_code])
        logger.info("  %s: %d sections", act_code, count)

    return len(all_rows)


if __name__ == "__main__":
    ingest_all()
