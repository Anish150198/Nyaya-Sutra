"""
Utility to load quantized GGUF models into RAM using llama-cpp-python.
Handles both local filesystem and Databricks DBFS/Volumes paths.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_model_path(path: str) -> str:
    """
    Resolve a model path.  If the path starts with /Volumes or /dbfs,
    assume Databricks environment and return as-is.
    Otherwise treat as a local filesystem path.
    """
    if path.startswith(("/Volumes", "/dbfs")):
        return path
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")
    return str(resolved)


def estimate_memory_gb(model_path: str) -> float:
    """Return an approximate memory footprint in GB for a GGUF file."""
    size_bytes = os.path.getsize(resolve_model_path(model_path))
    overhead_factor = 1.15  # ~15 % runtime overhead on top of file size
    return (size_bytes / (1024 ** 3)) * overhead_factor


def load_llama_model(
    model_path: str,
    n_ctx: int = 2048,
    n_threads: int | None = None,
    verbose: bool = False,
):
    """
    Load a GGUF model via llama-cpp-python and return the Llama instance.

    Parameters
    ----------
    model_path : str
        Path to the .gguf file (local or DBFS).
    n_ctx : int
        Context window size.
    n_threads : int or None
        Number of CPU threads; defaults to os.cpu_count().
    verbose : bool
        Enable llama.cpp verbose logging.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required. Install with: "
            "pip install llama-cpp-python"
        ) from exc

    resolved = resolve_model_path(model_path)
    mem_gb = estimate_memory_gb(resolved)
    logger.info("Loading model %s  (est. %.2f GB RAM)", resolved, mem_gb)

    if n_threads is None:
        n_threads = os.cpu_count() or 4

    model = Llama(
        model_path=resolved,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=verbose,
    )
    logger.info("Model loaded successfully.")
    return model
