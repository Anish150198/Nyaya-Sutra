"""
Run the full medallion pipeline: Bronze → Silver → Gold.

Usage:
  python -m pipelines.run_all                # all stages
  python -m pipelines.run_all --stage bronze # single stage
  python -m pipelines.run_all --stage silver
  python -m pipelines.run_all --stage gold
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_bronze():
    from pipelines.bronze_ingest import ingest_all
    logger.info("=" * 60)
    logger.info("STAGE 1: BRONZE — Ingest raw JSON → Delta")
    logger.info("=" * 60)
    t0 = time.time()
    count = ingest_all()
    logger.info("Bronze complete: %d rows (%.1fs)", count, time.time() - t0)
    return count


def run_silver():
    from pipelines.silver_chunk import chunk_bronze
    logger.info("=" * 60)
    logger.info("STAGE 2: SILVER — Chunk sections → Delta")
    logger.info("=" * 60)
    t0 = time.time()
    count = chunk_bronze()
    logger.info("Silver complete: %d chunks (%.1fs)", count, time.time() - t0)
    return count


def run_gold():
    from pipelines.gold_embed import embed_and_store
    logger.info("=" * 60)
    logger.info("STAGE 3: GOLD — Embed + Vector DB upsert → Delta")
    logger.info("=" * 60)
    t0 = time.time()
    count = embed_and_store()
    logger.info("Gold complete: %d embeddings (%.1fs)", count, time.time() - t0)
    return count


def main():
    parser = argparse.ArgumentParser(description="Nyaya-Sahayak Medallion Pipeline")
    parser.add_argument("--stage", choices=["bronze", "silver", "gold", "all"], default="all",
                        help="Which pipeline stage to run (default: all)")
    args = parser.parse_args()

    t_total = time.time()

    if args.stage in ("bronze", "all"):
        run_bronze()

    if args.stage in ("silver", "all"):
        run_silver()

    if args.stage in ("gold", "all"):
        run_gold()

    logger.info("=" * 60)
    logger.info("✅ Pipeline complete (%.1fs total)", time.time() - t_total)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
