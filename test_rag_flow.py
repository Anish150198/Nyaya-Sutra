import sys
import os
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.pipeline import run_legal_rag
from models.translation.indictrans2_runner import to_canonical

def test_rag():
    text = "मुझे बीएनएस धारा 300 के बारे में बताएं"
    print(f"User Query: {text}")
    
    canon, _ = to_canonical(text, "hi")
    print(f"Translated Query: {canon}")
    
    ans = run_legal_rag(canon, "citizen")
    print(f"Answer: {ans.answer_text[:200]}...")
    print(f"Confidence: {ans.confidence}")
    print("Citations:")
    for c in ans.citations:
        print(f" - {c.code} section {c.section_no}")

if __name__ == "__main__":
    test_rag()
