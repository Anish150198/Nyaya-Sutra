# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# dependencies = [
#   "openai",
# ]
# ///
# MAGIC %md
# MAGIC # 03 — Gold Layer: Embed Chunks → Delta Table + Vector Index
# MAGIC
# MAGIC Reads Silver Delta, generates embeddings (OpenAI API or HuggingFace),
# MAGIC writes Gold Delta table with Z-ORDER, and optionally syncs to a vector DB.
# MAGIC
# MAGIC **Note**: On Community Edition, we call the OpenAI Embeddings API from the driver.
# MAGIC On paid tiers, you could use Databricks Vector Search for auto-sync.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Paths — adjust for your Databricks workspace
NYAYA_DATA_ROOT = "/Volumes/workspace/default/tes/data"
SILVER_DELTA = f"{NYAYA_DATA_ROOT}/silver/delta/law_chunks"
GOLD_DELTA = f"{NYAYA_DATA_ROOT}/gold/delta/law_embeddings"

CATALOG = "workspace"
SCHEMA = "default"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.gold_law_embeddings"

# Set your OpenAI key as a Databricks secret or notebook widget
# dbutils.widgets.text("openai_api_key", "", "OpenAI API Key")
# OPENAI_API_KEY = dbutils.widgets.get("openai_api_key")

# Or set directly (NOT recommended for production — use Databricks Secrets)
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64

ACTS = ["BNS", "BNSS", "BSA"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Silver

# COMMAND ----------

from pyspark.sql import functions as F

silver_df = spark.read.format("delta").load(SILVER_DELTA)
print(f"Silver chunks: {silver_df.count()}")
display(silver_df.groupBy("act_code").count().orderBy("act_code"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Embed using OpenAI API (driver-side batched)
# MAGIC
# MAGIC For Community Edition we collect to driver and call the API.
# MAGIC For larger datasets on paid tiers, use a Pandas UDF with `mapInPandas`.

# COMMAND ----------

import numpy as np
import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenAI API."""
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]

# Collect Silver data to driver (fine for ~2K chunks)
silver_rows = silver_df.select(
    "chunk_id", "act_code", "section_number", "title", "chunk_text"
).collect()

print(f"Embedding {len(silver_rows)} chunks...")

all_embeddings = []
for i in range(0, len(silver_rows), BATCH_SIZE):
    batch_texts = [row["chunk_text"] for row in silver_rows[i:i + BATCH_SIZE]]
    batch_emb = embed_batch(batch_texts)
    all_embeddings.extend(batch_emb)
    print(f"  Batch {i // BATCH_SIZE + 1}/{(len(silver_rows) + BATCH_SIZE - 1) // BATCH_SIZE} done")

print(f"✓ Embedded {len(all_embeddings)} chunks (dim={len(all_embeddings[0])})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build Gold DataFrame

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
from datetime import datetime, timezone

gold_schema = StructType([
    StructField("chunk_id", StringType()),
    StructField("act_code", StringType()),
    StructField("section_number", StringType()),
    StructField("title", StringType()),
    StructField("chunk_text", StringType()),
    StructField("embedding", ArrayType(FloatType())),
    StructField("embedded_at", StringType()),
])

now = datetime.now(timezone.utc).isoformat()
gold_data = []
for i, row in enumerate(silver_rows):
    gold_data.append({
        "chunk_id": row["chunk_id"],
        "act_code": row["act_code"],
        "section_number": row["section_number"],
        "title": row["title"],
        "chunk_text": row["chunk_text"],
        "embedding": all_embeddings[i],
        "embedded_at": now,
    })

gold_df = spark.createDataFrame(gold_data, schema=gold_schema)
print(f"Gold DataFrame: {gold_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write Gold Delta with Z-ORDER

# COMMAND ----------

(
    gold_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_DELTA)
)

# Z-ORDER for fast metadata filtering
spark.sql(f"OPTIMIZE delta.`{GOLD_DELTA}` ZORDER BY (act_code, section_number)")

print(f"✅ Gold Delta written to {GOLD_DELTA} (Z-ORDERed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify

# COMMAND ----------

verify_df = spark.read.format("delta").load(GOLD_DELTA)
print(f"Gold Delta rows: {verify_df.count()}")
print(f"Embedding dim: {len(verify_df.select('embedding').first()[0])}")
display(verify_df.groupBy("act_code").count().orderBy("act_code"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. (Optional) Register as table

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
(
    gold_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")  # Required for Vector Search sync
    .saveAsTable(FULL_TABLE_NAME)
)
print(f"✅ Registered as {FULL_TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test: Cosine similarity search from Gold Delta
# MAGIC
# MAGIC This shows how to do vector search directly on the Delta table.
# MAGIC On paid tiers, you'd use Databricks Vector Search instead.

# COMMAND ----------

def search_gold(query: str, act_filter: str = None, top_k: int = 5):
    """Search Gold Delta table using cosine similarity."""
    # Embed query
    q_emb = embed_batch([query])[0]
    q_vec = np.array(q_emb, dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # Read Gold table
    gdf = spark.read.format("delta").load(GOLD_DELTA)
    if act_filter:
        gdf = gdf.filter(F.col("act_code") == act_filter)

    rows = gdf.select("chunk_id", "act_code", "section_number", "title", "chunk_text", "embedding").collect()

    # Compute similarities
    results = []
    for row in rows:
        emb = np.array(row["embedding"], dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        score = float(np.dot(q_vec, emb))
        results.append({
            "act": row["act_code"],
            "section": row["section_number"],
            "title": row["title"],
            "score": round(score, 4),
            "text": row["chunk_text"][:150],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# Test queries
for q in ["What is the bail procedure?", "punishment for theft", "evidence admissibility"]:
    print(f"\n--- Query: {q} ---")
    for r in search_gold(q):
        print(f"  {r['act']} §{r['section']} — {r['title']} (score={r['score']:.3f})")