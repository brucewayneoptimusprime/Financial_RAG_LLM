# app/build_index.py
from pathlib import Path
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "processed" / "chunks.jsonl"
INDEX_PATH = ROOT / "processed" / "vectors.faiss"
META_PATH = ROOT / "processed" / "chunk_meta.jsonl"   # id-aligned metadata

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    if not CHUNKS_PATH.exists():
        print(f"❌ Missing: {CHUNKS_PATH}")
        return

    # Load chunks
    chunks = list(read_jsonl(CHUNKS_PATH))
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks.")

    # Load embedding model
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Embed in batches
    embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i+BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embs.append(vecs)
    embs = np.vstack(embs).astype("float32")  # shape: (N, D)
    n, d = embs.shape
    print(f"Embeddings shape: {embs.shape}")

    # Build FAISS index (cosine via inner product on normalized vectors)
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"✅ Wrote FAISS index → {INDEX_PATH}")

    # Write aligned metadata for later lookup
    with META_PATH.open("w", encoding="utf-8") as f:
        for rec in chunks:
            out = {
                "doc": rec.get("doc"),
                "page_start": rec.get("page_start"),
                "page_end": rec.get("page_end"),
                "n_words": rec.get("n_words"),
                # keep a short preview to display in answers
                "preview": (rec.get("text") or "")[:240].replace("\n", " ")
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"✅ Wrote metadata (aligned to index ids) → {META_PATH}")

if __name__ == "__main__":
    main()
