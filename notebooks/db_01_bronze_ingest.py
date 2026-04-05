# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# dependencies = [
#   "-r /Workspace/Users/akashmaji945@gmail.com/nyaya-sahayak/requirements.txt",
# ]
# ///
# MAGIC %md
# MAGIC # 01 — Bronze Layer: Ingest Raw Law Data → Delta Table
# MAGIC
# MAGIC Reads scraped JSON files uploaded to DBFS and writes a Bronze Delta table.
# MAGIC
# MAGIC **Databricks Community Edition**: upload JSON folders to `/FileStore/nyaya/bronze/laws/{BNS,BNSS,BSA}/`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Paths — adjust for your Databricks workspace
NYAYA_DATA_ROOT = "/Volumes/workspace/default/tes/data"
DBFS_RAW_DIR = f"{NYAYA_DATA_ROOT}/bronze/laws"    # upload scraped JSONs here
BRONZE_DELTA = f"{NYAYA_DATA_ROOT}/bronze/delta/laws_raw"     # Bronze Delta output

CATALOG = "workspace"
SCHEMA = "default"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.laws_raw"

ACTS = ["BNS", "BNSS", "BSA", "IPC", "CONSTITUTION", "OTHER_ACTS"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read all JSON files into a DataFrame

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("section_number", StringType(), True),
    StructField("section_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("content", StringType(), True),
    StructField("source_url", StringType(), True),
])

frames = []
for act in ACTS:
    path = f"{DBFS_RAW_DIR}/{act}/json/"
    try:
        df = (
            spark.read
            .schema(schema)
            .option("multiLine", True)
            .json(path)
            .withColumn("act_code", F.lit(act))
            .withColumn("ingested_at", F.current_timestamp())
        )
        count = df.count()
        print(f"✓ {act}: {count} sections from {path}")
        frames.append(df)
    except Exception as e:
        print(f"✗ {act}: {e}")

if not frames:
    raise RuntimeError("No data loaded! Upload JSONs to DBFS first.")

bronze_df = frames[0]
for f in frames[1:]:
    bronze_df = bronze_df.unionByName(f)

print(f"\nTotal rows: {bronze_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Preview

# COMMAND ----------

display(bronze_df.select("act_code", "section_number", "title", "ingested_at").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write Bronze Delta Table

# COMMAND ----------

(
    bronze_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(BRONZE_DELTA)
)

print(f"✅ Bronze Delta written to {BRONZE_DELTA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify

# COMMAND ----------

verify_df = spark.read.format("delta").load(BRONZE_DELTA)
print(f"Bronze Delta rows: {verify_df.count()}")
display(verify_df.groupBy("act_code").count().orderBy("act_code"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. (Optional) Register as a table
# MAGIC
# MAGIC On Community Edition this creates a Hive metastore table.
# MAGIC On paid tiers with Unity Catalog, use `CREATE TABLE catalog.schema.bronze_laws_raw`.

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
verify_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(FULL_TABLE_NAME)
print(f"✅ Registered as {FULL_TABLE_NAME}")