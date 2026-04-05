"""
Vector DB abstraction layer.
Supports three backends:
  - chroma_local   : ChromaDB with persistent on-disk storage (default for local)
  - chroma_remote  : ChromaDB client connecting to a remote server
  - databricks_vs  : Databricks Vector Search (Delta Sync Index)

Each act (BNS, BNSS, BSA) gets its own collection/index.
NO FAISS dependency.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import (
    VECTOR_DB, ACTS, TOP_K_RETRIEVAL,
    CHROMA_LOCAL_DIR, CHROMA_HOST, CHROMA_PORT, CHROMA_AUTH_TOKEN, CHROMA_SSL,
    DATABRICKS_HOST, DATABRICKS_TOKEN,
    VS_ENDPOINT_NAME, VS_INDEX_NAME, VS_SOURCE_TABLE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChromaDB backend
# ---------------------------------------------------------------------------

_chroma_client = None


def _get_chroma_client():
    """Lazy-init ChromaDB client (local or remote)."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    try:
        import chromadb
    except ImportError:
        raise ImportError("chromadb is required. pip install chromadb")

    if VECTOR_DB == "chroma_remote":
        kwargs = {"host": CHROMA_HOST, "port": CHROMA_PORT}
        if CHROMA_SSL:
            kwargs["ssl"] = True
        if CHROMA_AUTH_TOKEN:
            kwargs["headers"] = {"Authorization": f"Bearer {CHROMA_AUTH_TOKEN}"}
        _chroma_client = chromadb.HttpClient(**kwargs)
        logger.info("ChromaDB remote client → %s:%s (ssl=%s)", CHROMA_HOST, CHROMA_PORT, CHROMA_SSL)
    else:
        Path(CHROMA_LOCAL_DIR).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_LOCAL_DIR)
        logger.info("ChromaDB local persistent → %s", CHROMA_LOCAL_DIR)

    return _chroma_client


def _collection_name(act_code: str) -> str:
    """Standardized collection name per act."""
    return f"nyaya_{act_code.lower()}"


def chroma_upsert(act_code: str, chunks: list[dict], embeddings: np.ndarray):
    """
    Upsert chunks + embeddings into a ChromaDB collection for one act.

    Parameters
    ----------
    act_code : str
    chunks : list[dict]  — each must have 'chunk_id', 'text', plus metadata keys
    embeddings : np.ndarray of shape (N, dim)
    """
    client = _get_chroma_client()
    col = client.get_or_create_collection(
        name=_collection_name(act_code),
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {k: str(v) for k, v in c.items() if k not in ("text", "chunk_id")}
        for c in chunks
    ]

    # ChromaDB accepts list of lists for embeddings
    emb_list = embeddings.tolist()

    # Upsert in batches of 500
    batch = 500
    for i in range(0, len(ids), batch):
        col.upsert(
            ids=ids[i:i+batch],
            documents=documents[i:i+batch],
            metadatas=metadatas[i:i+batch],
            embeddings=emb_list[i:i+batch],
        )

    logger.info("ChromaDB: upserted %d vectors into collection '%s'", len(ids), _collection_name(act_code))
    return col


def chroma_search(act_code: str, query_embedding, top_k: int = TOP_K_RETRIEVAL, where: dict = None) -> list[dict]:
    """Search a ChromaDB collection. Returns list of dicts with score."""
    client = _get_chroma_client()
    name = _collection_name(act_code)

    try:
        col = client.get_collection(name)
    except Exception:
        logger.warning("ChromaDB collection '%s' not found", name)
        return []

    count = col.count()
    if count == 0:
        return []

    query_kwargs = {
        "query_embeddings": [query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
        "n_results": min(top_k, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = col.query(**query_kwargs)

    out = []
    if results and results["ids"]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            entry = dict(meta)
            entry["chunk_id"] = doc_id
            entry["text"] = results["documents"][0][i] if results["documents"] else ""
            # ChromaDB returns distances (lower = better for cosine). Convert to similarity.
            dist = results["distances"][0][i] if results["distances"] else 1.0
            entry["score"] = round(1.0 - dist, 4)
            out.append(entry)
    return out


def chroma_get_by_section(act_code: str, section_number: str) -> list[dict]:
    """Fetch chunks by exact section_number metadata from ChromaDB."""
    client = _get_chroma_client()
    name = _collection_name(act_code)
    try:
        col = client.get_collection(name)
    except Exception:
        return []

    results = col.get(
        where={"section_number": section_number},
        include=["documents", "metadatas"],
    )
    out = []
    if results and results["ids"]:
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            entry = dict(meta)
            entry["chunk_id"] = doc_id
            entry["text"] = results["documents"][i] if results["documents"] else ""
            entry["score"] = 1.0  # exact match
            out.append(entry)
    return out


def chroma_collection_exists(act_code: str) -> bool:
    """Check if a ChromaDB collection exists for this act."""
    try:
        client = _get_chroma_client()
        col = client.get_collection(_collection_name(act_code))
        count = col.count()
        logger.debug("ChromaDB collection '%s': %d vectors", _collection_name(act_code), count)
        return count > 0
    except Exception as e:
        logger.debug("ChromaDB collection '%s' not available: %s", _collection_name(act_code), e)
        return False


def chroma_get_stats(act_code: str) -> dict:
    """Get stats about a ChromaDB collection."""
    try:
        client = _get_chroma_client()
        col = client.get_collection(_collection_name(act_code))
        return {"name": _collection_name(act_code), "count": col.count()}
    except Exception:
        return {"name": _collection_name(act_code), "count": 0}


# ---------------------------------------------------------------------------
# Databricks Vector Search backend
# ---------------------------------------------------------------------------

_vs_client = None


def _get_vs_client():
    """Lazy-init Databricks Vector Search client."""
    global _vs_client
    if _vs_client is not None:
        return _vs_client

    try:
        from databricks.vector_search.client import VectorSearchClient
    except ImportError:
        raise ImportError(
            "databricks-vectorsearch is required. "
            "pip install databricks-vectorsearch"
        )

    if DATABRICKS_HOST and DATABRICKS_TOKEN:
        _vs_client = VectorSearchClient(
            workspace_url=DATABRICKS_HOST,
            personal_access_token=DATABRICKS_TOKEN,
        )
    else:
        # Inside Databricks runtime — auto-auth
        _vs_client = VectorSearchClient()

    logger.info("Databricks Vector Search client initialised (endpoint=%s)", VS_ENDPOINT_NAME)
    return _vs_client


def vs_search(act_code: str, query_embedding, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Search Databricks Vector Search index, optionally filtering by act_code."""
    client = _get_vs_client()
    try:
        index = client.get_index(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=VS_INDEX_NAME,
        )
    except Exception as e:
        logger.error(
            "Vector Search index '%s' not found or not ready. "
            "Please run 'notebooks/db_04_vector_search_setup' first. Error: %s",
            VS_INDEX_NAME, e
        )
        return []

    emb = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
    if isinstance(emb[0], list):
        emb = emb[0]  # flatten if 2-D

    filters = {"act_code": act_code} if act_code else None
    results = index.similarity_search(
        query_vector=emb,
        columns=["chunk_id", "act_code", "section_number", "title", "chunk_text"],
        num_results=top_k,
        filters=filters,
    )

    out = []
    if results and "result" in results:
        data_array = results["result"].get("data_array", [])
        for row in data_array:
            out.append({
                "chunk_id": row[0],
                "act": row[1],
                "section_number": row[2],
                "title": row[3],
                "text": row[4],
                "score": float(row[5]) if len(row) > 5 else 0.0,
            })
    return out


def vs_get_by_section(act_code: str, section_number: str) -> list[dict]:
    """Get chunks by section from Databricks VS using a filter-only query."""
    from core.config import OPENAI_EMBEDDING_DIM
    client = _get_vs_client()
    try:
        index = client.get_index(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=VS_INDEX_NAME,
        )
    except Exception as e:
        logger.error(
            "Vector Search index '%s' not found. "
            "Please run 'notebooks/db_04_vector_search_setup' first. Error: %s",
            VS_INDEX_NAME, e
        )
        return []

    filters = {"act_code": act_code, "section_number": str(section_number)}
    results = index.similarity_search(
        query_vector=[0.0] * OPENAI_EMBEDDING_DIM,  # dummy vector — only filters matter
        columns=["chunk_id", "act_code", "section_number", "title", "chunk_text"],
        num_results=20,
        filters=filters,
    )

    out = []
    if results and "result" in results:
        for row in results["result"].get("data_array", []):
            out.append({
                "chunk_id": row[0],
                "act": row[1],
                "section_number": row[2],
                "title": row[3],
                "text": row[4],
                "score": 1.0,
            })
    return out


def vs_exists() -> bool:
    """Check if the Vector Search index exists and is ready."""
    try:
        client = _get_vs_client()
        index = client.get_index(
            endpoint_name=VS_ENDPOINT_NAME,
            index_name=VS_INDEX_NAME,
        )
        return True
    except Exception:
        # Index doesn't exist or isn't ready
        return False
        return index is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Unified interface — dispatches to the configured backend
# ---------------------------------------------------------------------------

def upsert_act(act_code: str, chunks: list[dict], embeddings: np.ndarray):
    """Insert/update vectors for one act into the configured vector DB."""
    if VECTOR_DB in ("chroma_local", "chroma_remote"):
        chroma_upsert(act_code, chunks, embeddings)
    elif VECTOR_DB == "databricks_vs":
        # Vector Search uses Delta Sync — no manual upsert needed.
        # Data flows: Gold Delta table → auto-sync → VS index.
        logger.info("databricks_vs: skipping upsert (Delta Sync auto-syncs from %s)", VS_SOURCE_TABLE)
    else:
        raise ValueError(
            f"Unknown VECTOR_DB: {VECTOR_DB}. Use chroma_local, chroma_remote, or databricks_vs"
        )


def search_act(act_code: str, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Search one act's vector store."""
    # Flatten to 1-D if needed
    if query_embedding.ndim == 2:
        query_embedding = query_embedding[0]

    if VECTOR_DB in ("chroma_local", "chroma_remote"):
        return chroma_search(act_code, query_embedding, top_k)
    elif VECTOR_DB == "databricks_vs":
        return vs_search(act_code, query_embedding, top_k)
    else:
        raise ValueError(f"Unknown VECTOR_DB: {VECTOR_DB}")


def get_section(act_code: str, section_number: str) -> list[dict]:
    """Get chunks for a specific section number (exact metadata match)."""
    if VECTOR_DB in ("chroma_local", "chroma_remote"):
        return chroma_get_by_section(act_code, section_number)
    elif VECTOR_DB == "databricks_vs":
        return vs_get_by_section(act_code, section_number)
    return []


def search_acts(act_codes: list[str], query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """Search multiple acts and return merged top-k."""
    all_results = []
    for act_code in act_codes:
        try:
            results = search_act(act_code, query_embedding, top_k)
            all_results.extend(results)
        except Exception as e:
            logger.warning("Search failed for %s: %s", act_code, e)
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results[:top_k]


def act_exists(act_code: str) -> bool:
    """Check if an act has vectors in the configured DB."""
    if VECTOR_DB in ("chroma_local", "chroma_remote"):
        return chroma_collection_exists(act_code)
    elif VECTOR_DB == "databricks_vs":
        return vs_exists()
    return False


def get_available_acts() -> list[str]:
    """Return list of acts that have vectors in the configured DB."""
    return [code for code in ACTS if act_exists(code)]


def get_db_info() -> dict:
    """Return info about the current vector DB configuration."""
    info = {"provider": VECTOR_DB, "acts": {}}
    for code in ACTS:
        if VECTOR_DB in ("chroma_local", "chroma_remote"):
            info["acts"][code] = chroma_get_stats(code)
        elif VECTOR_DB == "databricks_vs":
            info["acts"][code] = {"vs_index": VS_INDEX_NAME, "exists": vs_exists()}

    if VECTOR_DB == "chroma_local":
        info["path"] = CHROMA_LOCAL_DIR
    elif VECTOR_DB == "chroma_remote":
        info["host"] = f"{CHROMA_HOST}:{CHROMA_PORT}"
        info["ssl"] = CHROMA_SSL
    elif VECTOR_DB == "databricks_vs":
        info["endpoint"] = VS_ENDPOINT_NAME
        info["index"] = VS_INDEX_NAME
        info["source_table"] = VS_SOURCE_TABLE
    return info
