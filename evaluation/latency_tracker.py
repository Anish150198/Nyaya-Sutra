"""
Latency tracking with MLflow tracing decorators.
Measures Time To First Token (TTFT) and Time Per Output Token (TPOT).
"""

import logging
import time
import functools
from typing import Callable, Any

logger = logging.getLogger(__name__)


def trace_latency(func: Callable) -> Callable:
    """
    Decorator that measures and logs execution latency.
    When MLflow is available, uses @mlflow.trace; otherwise logs to Python logger.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "[LATENCY] %s: %.2f ms",
            func.__qualname__,
            elapsed_ms,
        )

        # Try to log to MLflow
        try:
            import mlflow
            mlflow.log_metric(f"latency_{func.__name__}_ms", elapsed_ms)
        except Exception:
            pass

        return result
    return wrapper


def measure_llm_latency(generate_fn: Callable, prompt: str, **kwargs) -> dict:
    """
    Wrap an LLM generate call and measure TTFT and TPOT.

    Parameters
    ----------
    generate_fn : Callable
        The model's generate function (e.g., param1_runner.generate).
    prompt : str
        The input prompt.

    Returns
    -------
    dict with keys: text, tokens_generated, ttft_ms, tpot_ms, total_ms
    """
    t0 = time.perf_counter()
    result = generate_fn(prompt, **kwargs)
    total_ms = (time.perf_counter() - t0) * 1000

    tokens = result.get("tokens_generated", 1)
    ttft_ms = result.get("ttft_ms", total_ms / max(tokens, 1))
    tpot_ms = (total_ms - ttft_ms) / max(tokens - 1, 1) if tokens > 1 else 0

    metrics = {
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "total_ms": round(total_ms, 2),
        "tokens_generated": tokens,
    }

    logger.info(
        "[LLM LATENCY] TTFT=%.1fms  TPOT=%.1fms  Total=%.1fms  Tokens=%d",
        metrics["ttft_ms"],
        metrics["tpot_ms"],
        metrics["total_ms"],
        metrics["tokens_generated"],
    )

    # Log to MLflow
    try:
        import mlflow
        mlflow.log_metrics({
            "ttft_ms": metrics["ttft_ms"],
            "tpot_ms": metrics["tpot_ms"],
            "total_generation_ms": metrics["total_ms"],
            "tokens_generated": metrics["tokens_generated"],
        })
    except Exception:
        pass

    result.update(metrics)
    return result


def benchmark_pipeline(
    query: str,
    n_runs: int = 5,
) -> dict:
    """
    Run the full pipeline multiple times and compute average latency stats.

    Returns
    -------
    dict with avg_ms, min_ms, max_ms, p50_ms, p95_ms
    """
    from core.data_models import UserQuery
    from agents.orchestrator import handle

    latencies = []
    for i in range(n_runs):
        uq = UserQuery(text=query)
        t0 = time.perf_counter()
        _ = handle(uq)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        logger.info("Pipeline run %d/%d: %.1f ms", i + 1, n_runs, elapsed)

    latencies.sort()
    n = len(latencies)
    stats = {
        "avg_ms": round(sum(latencies) / n, 2),
        "min_ms": round(latencies[0], 2),
        "max_ms": round(latencies[-1], 2),
        "p50_ms": round(latencies[n // 2], 2),
        "p95_ms": round(latencies[int(n * 0.95)], 2),
        "n_runs": n,
    }

    logger.info("[BENCHMARK] %s", stats)
    return stats
