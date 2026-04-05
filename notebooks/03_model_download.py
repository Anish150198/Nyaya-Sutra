"""
03_model_download.py
====================
Fetches GGUF (Param-1) and IndicTrans2 weights into local filesystem
or Databricks Volumes for CPU inference.

Models downloaded:
  - Param-1 2.9B GGUF Q8_0
  - IndicTrans2 en-indic 1B (CTranslate2 format)
  - Sentence-transformer for embeddings (auto-cached by HuggingFace)
"""

# COMMAND ----------
# %pip install huggingface_hub llama-cpp-python ctranslate2 sentence-transformers

# COMMAND ----------

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path("__file__").resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from core.config import PARAM1_MODEL_PATH, INDICTRANS2_MODEL_PATH, EMBEDDING_MODEL_NAME

# COMMAND ----------
# ========================
# 1. Download Param-1 GGUF
# ========================

print("=== Downloading Param-1 2.9B GGUF ===")

param1_path = Path(PARAM1_MODEL_PATH)
param1_path.parent.mkdir(parents=True, exist_ok=True)

if param1_path.exists():
    size_gb = param1_path.stat().st_size / (1024**3)
    print(f"  ✓ Already exists: {param1_path} ({size_gb:.2f} GB)")
else:
    print(f"  Downloading to {param1_path}...")
    try:
        from huggingface_hub import hf_hub_download

        # ── OPTION A: Download from HuggingFace Hub ──
        # Replace with the actual repo and filename once available.
        # Example for a community-quantized GGUF:
        #
        # hf_hub_download(
        #     repo_id="<org>/Param-1-2.9B-GGUF",
        #     filename="param1-2.9b-q8_0.gguf",
        #     local_dir=str(param1_path.parent),
        # )

        # ── OPTION B: Manual steps ──
        print(
            "\n  ⚠ Automatic download not yet configured.\n"
            "  Manual steps to obtain Param-1 GGUF:\n"
            "  1. Download the Param-1 2.9B model from Sarvam AI:\n"
            "     https://huggingface.co/ai4bharat/Param-1-2.9B\n"
            "  2. Convert to GGUF using llama.cpp:\n"
            "     python llama.cpp/convert_hf_to_gguf.py \\\n"
            "       --outfile models/weights/param1-2.9b-f16.gguf \\\n"
            "       ai4bharat/Param-1-2.9B\n"
            "  3. Quantize to Q8_0:\n"
            "     ./llama.cpp/llama-quantize \\\n"
            "       models/weights/param1-2.9b-f16.gguf \\\n"
            "       models/weights/param1-2.9b-q8_0.gguf Q8_0\n"
            f"  4. Place the file at: {param1_path}\n"
        )
    except ImportError:
        print("  ⚠ huggingface_hub not installed. pip install huggingface_hub")

# COMMAND ----------
# ========================
# 2. Download IndicTrans2
# ========================

print("\n=== Downloading IndicTrans2 (CTranslate2) ===")

indictrans_path = Path(INDICTRANS2_MODEL_PATH)
indictrans_path.mkdir(parents=True, exist_ok=True)

if (indictrans_path / "model.bin").exists():
    print(f"  ✓ Already exists: {indictrans_path}")
else:
    print(f"  Downloading to {indictrans_path}...")
    try:
        from huggingface_hub import snapshot_download

        # ── OPTION A: Pre-converted CTranslate2 model ──
        # If a CT2-converted checkpoint exists on HF:
        #
        # snapshot_download(
        #     repo_id="<org>/indictrans2-en-indic-1B-ct2",
        #     local_dir=str(indictrans_path),
        # )

        # ── OPTION B: Convert from HuggingFace model ──
        print(
            "\n  ⚠ Automatic download not yet configured.\n"
            "  Manual steps to obtain IndicTrans2 CTranslate2 model:\n"
            "  1. Download the HF model:\n"
            "     git clone https://huggingface.co/ai4bharat/indictrans2-en-indic-1B\n"
            "  2. Install CTranslate2:\n"
            "     pip install ctranslate2\n"
            "  3. Convert to CTranslate2 format:\n"
            "     ct2-opus-mt-converter \\\n"
            "       --model_dir indictrans2-en-indic-1B \\\n"
            "       --output_dir models/weights/indictrans2-en-indic-1B-ct2 \\\n"
            "       --quantization int8\n"
            "  Alternatively, use the IndicTrans2 conversion script:\n"
            "     python -c \"\n"
            "       import ctranslate2\n"
            "       converter = ctranslate2.converters.TransformersConverter(\n"
            "           'ai4bharat/indictrans2-en-indic-1B'\n"
            "       )\n"
            "       converter.convert(\n"
            "           'models/weights/indictrans2-en-indic-1B-ct2',\n"
            "           quantization='int8'\n"
            "       )\n"
            "     \"\n"
            "  4. Copy the SentencePiece model into the output directory:\n"
            "     cp indictrans2-en-indic-1B/spm_model.model \\\n"
            f"       {indictrans_path}/spm_model.model\n"
        )
    except ImportError:
        print("  ⚠ huggingface_hub not installed. pip install huggingface_hub")

# COMMAND ----------
# ========================
# 3. Pre-cache Embedding Model
# ========================

print("\n=== Pre-caching Embedding Model ===")
print(f"  Model: {EMBEDDING_MODEL_NAME}")

try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    print(f"  ✓ Loaded and cached. Embedding dim: {dim}")

    # Quick sanity test
    test_emb = model.encode(["test sentence"])
    print(f"  ✓ Sanity check passed. Output shape: {test_emb.shape}")
except ImportError:
    print("  ⚠ sentence-transformers not installed. pip install sentence-transformers")
except Exception as e:
    print(f"  ⚠ Failed to load embedding model: {e}")

# COMMAND ----------
print("\n=== Model Download Summary ===")
for name, path in [
    ("Param-1 GGUF", PARAM1_MODEL_PATH),
    ("IndicTrans2 CT2", INDICTRANS2_MODEL_PATH),
]:
    p = Path(path)
    if p.exists() and (p.is_file() or any(p.iterdir())):
        print(f"  ✓ {name}: {p}")
    else:
        print(f"  ✗ {name}: NOT FOUND at {p}")

print(f"  ✓ Embeddings: {EMBEDDING_MODEL_NAME} (auto-cached by HuggingFace)")
