"""
Embedding generator for Nyaya-Sahayak.
Supports two backends:
  - openai  : text-embedding-3-small via OpenAI API (default, no local GPU)
  - huggingface : sentence-transformers (optional, CPU)

Controlled by EMBEDDING_PROVIDER env var.
"""

import logging
from typing import Optional

import numpy as np

from core.config import (
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_EMBEDDING_DIM,
    HF_EMBEDDING_MODEL,
    OPENAI_BASE_URL,
)

logger = logging.getLogger(__name__)

_openai_client = None
_hf_model = None


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. pip install openai")
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    _openai_client = OpenAI(**kwargs)
    logger.info("OpenAI embedding client initialised (model=%s)", OPENAI_EMBEDDING_MODEL)
    return _openai_client


def _embed_openai(texts: list[str]) -> np.ndarray:
    """Embed texts using OpenAI Embeddings API. Returns (N, 1536) float32 array."""
    client = _get_openai_client()
    # OpenAI API limit: 2048 tokens per string, 2048 strings per request
    # For safety, batch in groups of 100
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        batch_embs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embs)
    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# HuggingFace fallback (optional)
# ---------------------------------------------------------------------------

def _get_hf_model():
    global _hf_model
    if _hf_model is not None:
        return _hf_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for HuggingFace embeddings. "
            "pip install sentence-transformers"
        )
    logger.info("Loading HuggingFace embedding model: %s", HF_EMBEDDING_MODEL)
    _hf_model = SentenceTransformer(HF_EMBEDDING_MODEL, device="cpu")
    return _hf_model


def _embed_hf(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    model = _get_hf_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Embed a list of texts into dense vectors.

    Returns
    -------
    np.ndarray of shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.zeros((0, get_embedding_dim()), dtype=np.float32)

    if EMBEDDING_PROVIDER == "openai":
        logger.info("Embedding %d texts via OpenAI (%s)...", len(texts), OPENAI_EMBEDDING_MODEL)
        return _embed_openai(texts)
    elif EMBEDDING_PROVIDER == "huggingface":
        return _embed_hf(texts, batch_size=batch_size, show_progress=show_progress)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}. "
            "Use 'openai' or 'huggingface'."
        )


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, dim)."""
    return embed_texts([query])


def get_embedding_dim() -> int:
    """Return the dimensionality of the configured embedding model."""
    if EMBEDDING_PROVIDER == "openai":
        return OPENAI_EMBEDDING_DIM  # 1536 for text-embedding-3-small
    elif EMBEDDING_PROVIDER == "huggingface":
        model = _get_hf_model()
        return model.get_sentence_embedding_dimension()
    return OPENAI_EMBEDDING_DIM
