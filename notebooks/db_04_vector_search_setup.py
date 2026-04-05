# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# dependencies = [
#   "databricks-vectorsearch",
#   "openai",
# ]
# ///
# MAGIC %md
# MAGIC # 04 — Set Up Databricks Vector Search
# MAGIC
# MAGIC Creates a Vector Search **endpoint** and a **Delta Sync Index** on the Gold table.
# MAGIC The index auto-syncs whenever the Gold Delta table is updated.
# MAGIC
# MAGIC **Prerequisites**: Run `db_01`, `db_02`, `db_03` first so the Gold table exists.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Unity Catalog coordinates — must match what db_03 wrote
CATALOG = "workspace"
SCHEMA = "default"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_law_embeddings"

# Vector Search settings
VS_ENDPOINT_NAME = "nyaya_vs_endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.gold_vs_index"

# Embedding column + dimension (must match your embedding model)
EMBEDDING_COL = "embedding"
EMBEDDING_DIM = 1536  # text-embedding-3-small

# Primary key
PRIMARY_KEY = "chunk_id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create the Schema (if not exists)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"✓ Catalog/Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Verify Gold Table Exists

# COMMAND ----------

gold_df = spark.table(GOLD_TABLE)
row_count = gold_df.count()
print(f"✓ Gold table: {GOLD_TABLE}")
print(f"  Rows: {row_count}")
print(f"  Columns: {gold_df.columns}")
display(gold_df.groupBy("act_code").count().orderBy("act_code"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Vector Search Endpoint
# MAGIC
# MAGIC An endpoint is compute that serves your vector indexes.
# MAGIC Takes ~5-10 min to provision. You only need **one endpoint** for all indexes.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
VS_ENDPOINT_NAME='nyaya_vs_endpoint'
vsc = VectorSearchClient()

# Check if endpoint already exists
existing_endpoints = [ep["name"] for ep in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT_NAME in existing_endpoints:
    print(f"✓ Endpoint '{VS_ENDPOINT_NAME}' already exists")
else:
    print(f"Creating endpoint '{VS_ENDPOINT_NAME}'...")
    vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"✓ Endpoint creation initiated. Wait ~5-10 min for it to become ONLINE.")
    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Wait for Endpoint to be ONLINE
# MAGIC
# MAGIC Re-run this cell until status shows **ONLINE**.

# COMMAND ----------

import time

for i in range(30):
    ep = vsc.get_endpoint(VS_ENDPOINT_NAME)
    status = ep.get("endpoint_status", {}).get("state", "UNKNOWN")
    print(f"  [{i+1}] Endpoint status: {status}")
    if status == "ONLINE":
        print(f"✅ Endpoint '{VS_ENDPOINT_NAME}' is ONLINE!")
        break
    time.sleep(30)
else:
    print("⚠ Endpoint not online yet. Check Databricks UI → Compute → Vector Search Endpoints")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Create Delta Sync Vector Search Index
# MAGIC
# MAGIC This creates an index that **auto-syncs** from the Gold Delta table.
# MAGIC Whenever you re-run db_03, the index updates automatically.

# COMMAND ----------

# Check if index already exists
VS_INDEX_NAME='main.default.gold_vs_index'
try:
    existing_index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
    print(f"✓ Index '{VS_INDEX_NAME}' already exists")
    print(f"  Status: {existing_index.describe()}")
except Exception:
    print(f"Creating Delta Sync index '{VS_INDEX_NAME}'...")
    print(f"  Source table: {GOLD_TABLE}")
    print(f"  Embedding column: {EMBEDDING_COL} (dim={EMBEDDING_DIM})")

    vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=GOLD_TABLE,
        pipeline_type="TRIGGERED",       # sync on demand (or "CONTINUOUS" for auto)
        primary_key=PRIMARY_KEY,
        embedding_dimension=EMBEDDING_DIM,
        embedding_vector_column=EMBEDDING_COL,
    )
    print(f"✓ Index creation initiated. Syncing data...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Wait for Index Sync

# COMMAND ----------

for i in range(30):
    try:
        idx = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
        desc = idx.describe()
        status = desc.get("status", {}).get("detailed_state", "UNKNOWN")
        ready = desc.get("status", {}).get("ready", False)
        num_rows = desc.get("status", {}).get("num_rows_updated", 0)
        print(f"  [{i+1}] Index status: {status}, ready={ready}, rows_synced={num_rows}")
        if ready:
            print(f"✅ Index '{VS_INDEX_NAME}' is READY with {num_rows} vectors!")
            break
    except Exception as e:
        print(f"  [{i+1}] Waiting... ({e})")
    time.sleep(30)
else:
    print("⚠ Index not ready yet. Check UI → Compute → Vector Search → Indexes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test Vector Search

# COMMAND ----------

import openai, os, numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def embed_query(text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return resp.data[0].embedding

# Get the index
index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)

# Test queries
test_queries = [
    ("bail procedure", "BNSS"),
    ("punishment for murder", "BNS"),
    ("evidence admissibility", "BSA"),
    ("explain section 300", None),  # no filter
]

for query, act_filter in test_queries:
    q_emb = embed_query(query)
    kwargs = {
        "query_vector": q_emb,
        "columns": ["chunk_id", "act_code", "section_number", "title", "chunk_text"],
        "num_results": 3,
    }
    if act_filter:
        kwargs["filters"] = {"act_code": act_filter}

    results = index.similarity_search(**kwargs)
    print(f"\n--- Query: '{query}' (filter={act_filter}) ---")
    if results and "result" in results:
        for row in results["result"].get("data_array", []):
            score = row[5] if len(row) > 5 else "?"
            print(f"  {row[1]} §{row[2]} — {row[3]} (score={score})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Trigger Re-sync (after updating Gold table)
# MAGIC
# MAGIC If you re-run `db_03_gold_embed`, call this to sync the new data:

# COMMAND ----------

# index = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME)
# index.sync()
# print("✓ Sync triggered. Check status with index.describe()")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Variables for Your App
# MAGIC
# MAGIC Set these in your `.env` or Databricks App config:
# MAGIC ```
# MAGIC VECTOR_DB=databricks_vs
# MAGIC VS_ENDPOINT_NAME=nyaya_vs_endpoint
# MAGIC VS_INDEX_NAME=main.nyaya.gold_vs_index
# MAGIC VS_SOURCE_TABLE=main.nyaya.gold_law_embeddings
# MAGIC ```