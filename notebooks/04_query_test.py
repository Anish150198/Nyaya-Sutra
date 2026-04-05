# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — End-to-End Query Test
# MAGIC
# MAGIC Tests the full pipeline: classifier → agent routing → FAISS retrieval → LLM answer.
# MAGIC **Run notebooks 01-03 first** to build the indexes.

# COMMAND ----------

import sys
from pathlib import Path

ROOT = str(Path.cwd().parent) if Path.cwd().name == 'notebooks' else str(Path.cwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(f'Project root: {ROOT}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Check available indexes

# COMMAND ----------

from rag.retriever import get_loaded_acts
available = get_loaded_acts()
print(f'Available indexes: {available}')
if not available:
    print('⚠️  No indexes found! Run notebooks 01-03 first.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test the intent classifier

# COMMAND ----------

from models.nlp_classifier.intent_classifier import classify

test_queries = [
    'What is the punishment for murder under BNS?',
    'How to file an FIR under BNSS?',
    'What counts as electronic evidence?',
    'What are my rights if police arrest me?',
    'BNS section 103',
]

for q in test_queries:
    intent = classify(q)
    print(f'Q: {q}\n  → Intent: {intent}\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test single-act search

# COMMAND ----------

from rag.retriever import search

if 'BNS' in available:
    print('=== BNS: "punishment for theft" ===')
    for r in search('BNS', 'punishment for theft', top_k=3):
        print(f'  §{r["section_number"]} {r["title"]} (score={r["score"]:.3f})')
        print(f'  {r["text"][:200]}...\n')

if 'BNSS' in available:
    print('=== BNSS: "procedure for arrest" ===')
    for r in search('BNSS', 'procedure for arrest', top_k=3):
        print(f'  §{r["section_number"]} {r["title"]} (score={r["score"]:.3f})')
        print(f'  {r["text"][:200]}...\n')

if 'BSA' in available:
    print('=== BSA: "electronic evidence" ===')
    for r in search('BSA', 'electronic evidence', top_k=3):
        print(f'  §{r["section_number"]} {r["title"]} (score={r["score"]:.3f})')
        print(f'  {r["text"][:200]}...\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test cross-act search

# COMMAND ----------

from rag.retriever import search_multiple

q = 'What are my rights if I am arrested?'
print(f'Query: {q}\nSearching: {available}\n')
for r in search_multiple(available, q, top_k=5):
    print(f'  [{r["act"]}] §{r["section_number"]} {r["title"]} (score={r["score"]:.3f})')
    print(f'  {r["text"][:200]}...\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test full orchestrator pipeline

# COMMAND ----------

from agents.orchestrator import handle

queries = [
    'What is the punishment for murder under BNS?',
    'How do I get bail under BNSS?',
    'What is electronic evidence under BSA?',
    'What are my rights if police arrest me?',
]

for q in queries:
    print(f'\n{"="*70}\nQ: {q}\n{"="*70}')
    result = handle(q)
    print(f'Acts: {result.get("detected_acts", [])}')
    print(f'Model: {result.get("model_id", "?")}')
    print(f'Confidence: {result.get("confidence", "?")}')
    print(f'\nAnswer:\n{result["answer"][:600]}')
    print(f'\nSources: {len(result.get("results", []))} chunks')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Test with specific act filter

# COMMAND ----------

result = handle('What is the punishment for theft?', act_filter='BNS')
print(f'Acts: {result.get("detected_acts")}')
print(f'Model: {result.get("model_id")}')
print(f'\nAnswer:\n{result["answer"][:500]}')
print(f'\n--- Sources ---')
for r in result.get('results', []):
    print(f'  [{r["act"]}] §{r["section_number"]} {r["title"]} ({r["score"]:.3f})')