# app/search_query.py
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "processed" / "vectors.faiss"
META_PATH = ROOT / "processed" / "chunk_meta.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"   # same model as build_index.py
TOP_K = 3

def load_meta(meta_path: Path):
    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    return meta

def main():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        print("❌ Missing index or metadata. Run app/build_index.py first.")
        return

    # 1) load index + metadata
    index = faiss.read_index(str(INDEX_PATH))
    meta = load_meta(META_PATH)
    dim = index.d  # vector dimension

    # 2) load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # 3) ask user for a query
    query = input("Your question: ").strip()
    if not query:
        print("No query provided.")
        return

    # 4) embed query (normalized to match IndexFlatIP)
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")  # shape (1, dim)

    # 5) search
    scores, ids = index.search(q_vec, TOP_K)  # inner product on normalized vectors ~ cosine similarity
    scores = scores[0]
    ids = ids[0]

    # 6) print results
    print("\nTop results:")
    for rank, (idx, sc) in enumerate(zip(ids, scores), start=1):
        if idx < 0:
            continue
        m = meta[idx]
        print(f"\n#{rank}  score={sc:.3f}  pages {m['page_start']}–{m['page_end']}  ({m['doc']})")
        print(f"snippet: {m['preview']}…")

if __name__ == "__main__":
    main()
