# app/answer_with_citations.py
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "processed" / "vectors.faiss"
META_PATH = ROOT / "processed" / "chunk_meta.jsonl"
CHUNKS_PATH = ROOT / "processed" / "chunks.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
GOOD_SCORE = 0.35  # if top score below this, we say "insufficient context"

def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def main():
    if not INDEX_PATH.exists():
        print("❌ Missing FAISS index. Run app/build_index.py first.")
        return
    if not META_PATH.exists() or not CHUNKS_PATH.exists():
        print("❌ Missing metadata or chunks. Build index first.")
        return

    # Load index + data
    index = faiss.read_index(str(INDEX_PATH))
    meta = load_jsonl(META_PATH)         # aligned with index IDs
    chunks = load_jsonl(CHUNKS_PATH)     # same order as meta

    # Embed query
    model = SentenceTransformer(MODEL_NAME)
    query = input("Your question: ").strip()
    if not query:
        print("No query provided.")
        return
    q = model.encode([query], normalize_embeddings=True).astype("float32")

    # Search
    scores, ids = index.search(q, TOP_K)
    scores = scores[0]
    ids = ids[0]

    # Guardrail: insufficient context if top score is low or invalid id
    if len(ids) == 0 or ids[0] < 0 or scores[0] < GOOD_SCORE:
        print("🤔 Insufficient context in the indexed document. Try rephrasing or adding more filings.")
        return

    # Build a simple answer from the top chunks (no LLM)
    print("\n=== Answer (constructed from retrieved text) ===")
    used = 0
    bullets = []
    for idx, sc in zip(ids, scores):
        if idx < 0:
            continue
        c = chunks[idx]
        # take a concise slice from the chunk
        snippet = (c["text"][:400].replace("\n", " ").strip())
        bullets.append(f"- p.{c['page_start']}–{c['page_end']}: “{snippet}…”")
        used += 1
        if used == 3:  # keep it short
            break

    # naive stitched summary (first sentence from each snippet)
    def first_sentence(s: str) -> str:
        for end in [". ", "; ", " — ", " - "]:
            if end in s:
                return s.split(end)[0]
        return s

    stitched = " ".join(first_sentence(b) for b in [chunks[i]["text"] for i in ids[:2] if i >= 0])[:600].replace("\n"," ")
    print(stitched if stitched else "(See citations below.)")

    print("\n=== Citations ===")
    for b in bullets:
        print(b)

    print("\n(score top-1 =", f"{scores[0]:.3f}", ")")

if __name__ == "__main__":
    main()
