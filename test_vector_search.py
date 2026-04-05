#!/usr/bin/env python3
"""Quick test to verify vector search is working with ChromaDB"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_db import search_acts, get_available_acts
from models.embeddings.embedder import embed_query

def test_vector_search():
    print("Testing Vector Search with ChromaDB...")
    
    # Check available acts
    acts = get_available_acts()
    print(f"Available acts: {acts}")
    
    if not acts:
        print("❌ No acts found in vector DB")
        return
    
    # Test query
    query = "What is punishment for murder?"
    print(f"\nQuery: {query}")
    
    # Get embedding
    embedding = embed_query(query)
    print(f"Embedding dimension: {len(embedding)}")
    
    # Search
    results = search_acts(acts, embedding, top_k=3)
    print(f"\nFound {len(results)} results:")
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Act: {r.get('act', 'Unknown')}")
        print(f"   Section: {r.get('section_number', 'Unknown')}")
        print(f"   Title: {r.get('title', '')[:80]}...")
        print(f"   Score: {r.get('score', 0):.3f}")
        print(f"   Text: {r.get('text', '')[:100]}...")

if __name__ == "__main__":
    test_vector_search()
