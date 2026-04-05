import sys
import os
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.translation.indictrans2_runner import to_canonical
from rag.pipeline import _extract_section_refs

def debug():
    text = "मुझे बीएनएस धारा 300 के बारे में बताएं"
    print(f"Original: {text}")
    
    canon, lang = to_canonical(text, "hi")
    print(f"Canonical: '{canon}'")
    print(f"Detected Lang: {lang}")
    
    refs = _extract_section_refs(canon)
    print(f"Extracted Refs: {refs}")

if __name__ == "__main__":
    debug()
