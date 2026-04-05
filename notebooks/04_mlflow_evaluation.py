"""
04_mlflow_evaluation.py
=======================
Runs the LLM-as-a-judge benchmarking suite.
Evaluates RAG quality (RAGAS), routing accuracy, and latency.
Logs all metrics to MLflow.
"""

# COMMAND ----------
# %pip install mlflow[databricks] ragas datasets

# COMMAND ----------

import sys
from pathlib import Path

PROJECT_ROOT = str(Path("__file__").resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from core.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# COMMAND ----------
# Configure MLflow
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"MLflow experiment:   {MLFLOW_EXPERIMENT_NAME}")

# COMMAND ----------
# ========================
# 1. BhashaBench-Legal Evaluation
# ========================

print("\n=== BhashaBench-Legal Evaluation ===")

from evaluation.bhashabench_eval import run_evaluation

# Run on a small subset first for quick validation
bb_results = run_evaluation(
    model_id="param1",
    language_filter="en",
    max_questions=10,  # increase for full eval
    experiment_name=MLFLOW_EXPERIMENT_NAME,
)

print(f"  Accuracy: {bb_results['accuracy']*100:.1f}%")
print(f"  Correct:  {bb_results['correct']}/{bb_results['total']}")
if bb_results.get("per_domain"):
    print("  Per-domain:")
    for domain, stats in bb_results["per_domain"].items():
        print(f"    {domain}: {stats['accuracy']*100:.1f}%")

# COMMAND ----------
# ========================
# 2. Trajectory / Routing Evaluation
# ========================

print("\n=== Trajectory / Routing Evaluation ===")

from evaluation.trajectory_eval import evaluate_routing

traj_results = evaluate_routing(experiment_name=MLFLOW_EXPERIMENT_NAME)
print(f"  Intent classification accuracy: {traj_results.get('accuracy', 0)*100:.1f}%")
if traj_results.get("confusion_matrix"):
    print("  Confusion matrix:")
    for expected, predictions in traj_results["confusion_matrix"].items():
        print(f"    {expected}: {predictions}")

# COMMAND ----------
# ========================
# 3. RAG Quality (RAGAS Metrics)
# ========================

print("\n=== RAGAS Quality Evaluation ===")

from evaluation.ragas_metrics import (
    compute_faithfulness,
    compute_context_precision,
    log_metrics_to_mlflow,
)
from rag.pipeline import run_legal_rag

# Test with a sample query
test_query = "What is the punishment for theft under BNS?"
legal_answer = run_legal_rag(test_query, persona="citizen", act_filter="BNS")

if legal_answer.citations:
    context = " ".join(c.snippet or "" for c in legal_answer.citations)
    faith = compute_faithfulness(legal_answer.answer_text, context)
    print(f"  Faithfulness score: {faith:.3f}")

    log_metrics_to_mlflow(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        metrics={"faithfulness": faith},
        params={"query": test_query, "model_id": legal_answer.model_id},
        run_name="ragas_quality_check",
    )
else:
    print("  ⚠ No citations retrieved – cannot compute RAGAS metrics.")
    print("    Ensure FAISS indexes are built (run notebook 02).")

# COMMAND ----------
# ========================
# 4. Latency Benchmark
# ========================

print("\n=== Latency Benchmark ===")

from evaluation.latency_tracker import benchmark_pipeline

latency_stats = benchmark_pipeline(
    query="What are the fundamental rights under the Indian Constitution?",
    n_runs=3,
)

print(f"  Avg: {latency_stats['avg_ms']:.0f} ms")
print(f"  P50: {latency_stats['p50_ms']:.0f} ms")
print(f"  P95: {latency_stats['p95_ms']:.0f} ms")

log_metrics_to_mlflow(
    experiment_name=MLFLOW_EXPERIMENT_NAME,
    metrics={
        "latency_avg_ms": latency_stats["avg_ms"],
        "latency_p50_ms": latency_stats["p50_ms"],
        "latency_p95_ms": latency_stats["p95_ms"],
    },
    params={"n_runs": str(latency_stats["n_runs"])},
    run_name="latency_benchmark",
)

# COMMAND ----------
print("\n=== Evaluation Complete ===")
print("View results in MLflow UI at your Databricks workspace.")
