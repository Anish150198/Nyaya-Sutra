# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver Layer: Chunk Bronze Sections → Delta Table
# MAGIC
# MAGIC Reads Bronze Delta, splits long sections into overlapping chunks,
# MAGIC writes a Silver Delta table with Z-ORDER on (act_code, section_number).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Paths — adjust for your Databricks workspace
NYAYA_DATA_ROOT = "/Volumes/workspace/default/tes/data"
BRONZE_DELTA = f"{NYAYA_DATA_ROOT}/bronze/delta/laws_raw"
SILVER_DELTA = f"{NYAYA_DATA_ROOT}/silver/delta/law_chunks"

CATALOG = "workspace"
SCHEMA = "default"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.law_sections"

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # overlap between chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read Bronze

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType

bronze_df = spark.read.format("delta").load(BRONZE_DELTA)
print(f"Bronze rows: {bronze_df.count()}")
display(bronze_df.groupBy("act_code").count().orderBy("act_code"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chunk using a PySpark UDF

# COMMAND ----------

from pyspark.sql.functions import udf, explode, col, monotonically_increasing_id

chunk_schema = ArrayType(StructType([
    StructField("chunk_text", StringType()),
    StructField("chunk_index", IntegerType()),
    StructField("total_chunks", IntegerType()),
]))

@udf(returnType=chunk_schema)
def split_into_chunks(content, max_len=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if not content or not content.strip():
        return []
    if len(content) <= max_len:
        return [{"chunk_text": content, "chunk_index": 0, "total_chunks": 1}]
    chunks = []
    start = 0
    while start < len(content):
        end = start + max_len
        chunk = content[start:end]
        chunks.append(chunk)
        start += max_len - overlap
    return [
        {"chunk_text": c, "chunk_index": i, "total_chunks": len(chunks)}
        for i, c in enumerate(chunks)
    ]

# Apply chunking
chunked_df = (
    bronze_df
    .withColumn("chunks", split_into_chunks(col("content")))
    .withColumn("chunk", explode(col("chunks")))
    .select(
        col("act_code"),
        col("section_number"),
        col("title"),
        col("chunk.chunk_text"),
        col("chunk.chunk_index"),
        col("chunk.total_chunks"),
    )
    .withColumn(
        "chunk_id",
        F.concat(
            col("act_code"), F.lit("_s"),
            F.coalesce(col("section_number"), F.lit("x")), F.lit("_"),
            col("chunk_index").cast("string"),
        )
    )
)

print(f"Total chunks: {chunked_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Preview

# COMMAND ----------

display(
    chunked_df
    .select("chunk_id", "act_code", "section_number", "title", "chunk_index", "total_chunks")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write Silver Delta with Z-ORDER

# COMMAND ----------

(
    chunked_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .save(SILVER_DELTA)
)

# Z-ORDER for fast lookups by act + section
spark.sql(f"OPTIMIZE delta.`{SILVER_DELTA}` ZORDER BY (act_code, section_number)")

print(f"✅ Silver Delta written to {SILVER_DELTA} (Z-ORDERed by act_code, section_number)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify

# COMMAND ----------

silver_df = spark.read.format("delta").load(SILVER_DELTA)
print(f"Silver Delta chunks: {silver_df.count()}")
display(silver_df.groupBy("act_code").count().orderBy("act_code"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. (Optional) Register as table

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(FULL_TABLE_NAME)
print(f"✅ Registered as {FULL_TABLE_NAME}")