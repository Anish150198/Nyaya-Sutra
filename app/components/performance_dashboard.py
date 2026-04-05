"""
UI panel to display MLflow traces, TTFT speeds, and RAGAS accuracy scores.
"""

import streamlit as st


def render_performance_dashboard():
    """Render the performance/evaluation dashboard tab."""
    st.subheader("📊 Performance Dashboard")
    st.markdown(
        "View model evaluation metrics, latency benchmarks, and RAGAS scores."
    )

    tab_metrics, tab_latency, tab_eval = st.tabs(
        ["RAGAS Metrics", "Latency", "BhashaBench Results"]
    )

    with tab_metrics:
        _render_ragas_section()

    with tab_latency:
        _render_latency_section()

    with tab_eval:
        _render_bhashabench_section()


def _render_ragas_section():
    """Display RAGAS evaluation metrics."""
    st.markdown("### RAG Quality Metrics")
    st.info("Run the evaluation pipeline to populate these metrics.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Faithfulness", "—", help="How well answers are grounded in retrieved context")
    with col2:
        st.metric("Answer Accuracy", "—", help="Correctness against ground truth")
    with col3:
        st.metric("Context Precision", "—", help="Relevance of retrieved documents")

    st.markdown("---")
    st.markdown(
        "To run evaluation:\n"
        "```python\n"
        "from evaluation.ragas_metrics import compute_faithfulness\n"
        "score = compute_faithfulness(answer, context)\n"
        "```"
    )

    # MLflow integration placeholder
    if st.button("Load from MLflow", key="load_ragas"):
        try:
            import mlflow
            from core.config import MLFLOW_EXPERIMENT_NAME
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
                if not runs.empty:
                    st.dataframe(runs[["run_id", "start_time", "metrics.accuracy"]].head())
                else:
                    st.warning("No MLflow runs found.")
            else:
                st.warning(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")
        except ImportError:
            st.warning("MLflow not installed. pip install mlflow[databricks]")
        except Exception as exc:
            st.error(f"Could not load MLflow data: {exc}")


def _render_latency_section():
    """Display latency benchmarks."""
    st.markdown("### Latency Benchmarks")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TTFT", "—", help="Time To First Token (ms)")
    with col2:
        st.metric("TPOT", "—", help="Time Per Output Token (ms)")
    with col3:
        st.metric("Total Gen", "—", help="Total generation time (ms)")
    with col4:
        st.metric("Pipeline E2E", "—", help="End-to-end pipeline latency (ms)")

    st.markdown("---")
    st.markdown(
        "To run latency benchmark:\n"
        "```python\n"
        "from evaluation.latency_tracker import benchmark_pipeline\n"
        "stats = benchmark_pipeline('What is BNS Section 103?', n_runs=5)\n"
        "```"
    )


def _render_bhashabench_section():
    """Display BhashaBench-Legal evaluation results."""
    st.markdown("### BhashaBench-Legal Evaluation")
    st.info("Run `evaluation/bhashabench_eval.py` to generate results.")

    st.markdown(
        "| Metric | Value |\n"
        "|--------|-------|\n"
        "| Total Questions | — |\n"
        "| Correct Answers | — |\n"
        "| Accuracy | — |\n"
    )

    if st.button("Run Quick Eval (5 questions)", key="quick_eval"):
        st.warning(
            "This will run the RAG pipeline on 5 BhashaBench questions. "
            "Ensure models are loaded."
        )
        try:
            from evaluation.bhashabench_eval import run_evaluation
            with st.spinner("Running evaluation..."):
                results = run_evaluation(max_questions=5)
            st.success(f"Accuracy: {results['accuracy']*100:.1f}% ({results['correct']}/{results['total']})")
            if results.get("per_domain"):
                st.json(results["per_domain"])
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
