"""
01_data_ingestion.py
====================
Uploads raw data to Databricks Volumes/DBFS and creates Bronze Delta tables.
Run this as a Databricks notebook or locally with databricks-connect.

Bronze tables created:
  - laws_raw
  - schemes_raw
  - ipc_bns_raw
  - legal_aid_raw
  - bb_legal_raw
"""

# COMMAND ----------
# %pip install databricks-connect pyspark

# COMMAND ----------

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path("__file__").resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from core.config import (
    BRONZE_SCHEMA, TABLE_LAWS_RAW, TABLE_SCHEMES_RAW, TABLE_IPC_BNS_RAW,
)

# COMMAND ----------
# Initialize Spark session
# On Databricks, `spark` is pre-initialized.
# Locally, use databricks-connect or local SparkSession.

try:
    spark  # noqa: F821 – available in Databricks notebooks
except NameError:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("NyayaSutra_DataIngestion") \
        .getOrCreate()

# COMMAND ----------
# Create Bronze schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {BRONZE_SCHEMA}")

# COMMAND ----------
# === 1. Ingest Law PDFs/JSONs ===
# Place raw law files in data/bronze/laws/ or upload to Volumes.

LAWS_DIR = Path(PROJECT_ROOT) / "data" / "bronze" / "laws"

if LAWS_DIR.exists():
    law_files = list(LAWS_DIR.glob("*.json"))
    if law_files:
        # Read JSON law files
        laws_df = spark.read.json([str(f) for f in law_files])
        laws_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_LAWS_RAW)
        print(f"✓ Ingested {laws_df.count()} law records into {TABLE_LAWS_RAW}")
    else:
        print(f"⚠ No JSON files found in {LAWS_DIR}. Place BNS/BNSS/BSA JSON files here.")
else:
    print(f"⚠ Directory {LAWS_DIR} does not exist. Create it and add raw law data.")

# COMMAND ----------
# === 2. Ingest IPC→BNS Mapping ===

MAPPING_DIR = Path(PROJECT_ROOT) / "data" / "bronze" / "ipc_bns_mapping"

if MAPPING_DIR.exists():
    csv_files = list(MAPPING_DIR.glob("*.csv"))
    json_files = list(MAPPING_DIR.glob("*.json"))

    if csv_files:
        mapping_df = spark.read.csv(
            [str(f) for f in csv_files],
            header=True, inferSchema=True,
        )
        mapping_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_IPC_BNS_RAW)
        print(f"✓ Ingested {mapping_df.count()} IPC→BNS mapping rows")
    elif json_files:
        mapping_df = spark.read.json([str(f) for f in json_files])
        mapping_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_IPC_BNS_RAW)
        print(f"✓ Ingested {mapping_df.count()} IPC→BNS mapping rows")
    else:
        print(f"⚠ No CSV/JSON files in {MAPPING_DIR}")
else:
    print(f"⚠ {MAPPING_DIR} not found.")

# COMMAND ----------
# === 3. Ingest Scheme Data ===

SCHEMES_DIR = Path(PROJECT_ROOT) / "data" / "bronze" / "schemes"

if SCHEMES_DIR.exists():
    scheme_files = list(SCHEMES_DIR.glob("*.json")) + list(SCHEMES_DIR.glob("*.csv"))
    if scheme_files:
        if scheme_files[0].suffix == ".csv":
            schemes_df = spark.read.csv([str(f) for f in scheme_files], header=True, inferSchema=True)
        else:
            schemes_df = spark.read.json([str(f) for f in scheme_files])
        schemes_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_SCHEMES_RAW)
        print(f"✓ Ingested {schemes_df.count()} scheme records into {TABLE_SCHEMES_RAW}")
    else:
        print(f"⚠ No data files in {SCHEMES_DIR}")
else:
    print(f"⚠ {SCHEMES_DIR} not found.")

# COMMAND ----------
print("\n=== Bronze Ingestion Summary ===")
for table in [TABLE_LAWS_RAW, TABLE_SCHEMES_RAW, TABLE_IPC_BNS_RAW]:
    try:
        count = spark.table(table).count()
        print(f"  {table}: {count} rows")
    except Exception:
        print(f"  {table}: NOT CREATED (missing source data)")
