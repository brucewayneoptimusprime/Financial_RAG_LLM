# app/qa_cli.py
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Tuple, Optional, Dict

import faiss
from sentence_transformers import SentenceTransformer

from app.formatting import best_sentences
from app.macro_utils import latest_value, latest_yoy
from app.text_utils import build_vocab_from_chunks, autocorrect_query
from app.rag_prompt import build_rag_prompt
from app.ollama_client import generate_with_ollama

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "processed" / "vectors.faiss"
META_PATH = ROOT / "processed" / "chunk_meta.jsonl"  # (not used but kept for parity)
CHUNKS_PATH = ROOT / "processed" / "chunks.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
CAND_K = TOP_K * 8          # pull wider, then filter by issuer/year
GOOD_SCORE = 0.30

# ---- deterministic pre-replacements (word-boundary) ----
PRE_REPLACEMENTS = {
    r"\brish\b": "risk",
    r"\bfacors\b": "factors",
    r"\bfacotr\b": "factor",
    r"\bfinacial\b": "financial",
    r"\boperatons\b": "operations",
    r"\bconditon\b": "condition",
    r"\benviroment\b": "environment",
}

ISSUER_ALIASES = {
    "AAPL": {"aapl", "apple", "apple inc", "apple’s", "apple's"},
    "MSFT": {"msft", "microsoft", "microsoft corp", "microsoft corporation"},
    # add more as you ingest more issuers
}

def apply_pre_replacements(q: str) -> tuple[str, dict]:
    changes = {}
    new_q = q
    for pat, repl in PRE_REPLACEMENTS.items():
        matches = list(re.finditer(pat, new_q, flags=re.IGNORECASE))
        if matches:
            new_q = re.sub(pat, repl, new_q, flags=re.IGNORECASE)
            for m in matches:
                orig = m.group(0)
                if orig not in changes and orig.lower() != repl.lower():
                    changes[orig] = repl
    return new_q, changes

def load_jsonl(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

# ---- Macro intent detection (simple) ----
def try_answer_macro(q: str):
    t = q.lower()
    if "cpi" in t:
        if "yoy" in t or "year over year" in t or "year-on-year" in t:
            d, v = latest_yoy("cpi")
            if v is not None:
                return (f"Latest CPI YoY is {float(v):.2f}% as of {d}. Source: FRED series CPIAUCSL.", [])
            return ("Sorry, I couldn't compute CPI YoY from the local CSV.", [])
        if any(k in t for k in ("level", "index", "latest", "value")):
            d, v = latest_value("cpi")
            return (f"Latest CPI level is {float(v):.3f} (index, 1982–84=100) as of {d}. Source: FRED series CPIAUCSL.", [])
    if "unemployment" in t or "jobless" in t:
        d, v = latest_value("unemployment")
        return (f"Latest unemployment rate is {float(v):.2f}% as of {d}. Source: FRED series UNRATE.", [])
    return (None, None)

# ---- Issuer & year parsing ----
def parse_requested_issuers_and_year(q: str) -> Tuple[set, Optional[int]]:
    ql = q.lower()
    wanted = set()
    for ticker, aliases in ISSUER_ALIASES.items():
        if any(a in ql for a in aliases):
            wanted.add(ticker)
    # detect a 4-digit year between 1990 and 2100
    year = None
    m = re.search(r"\b(19|20)\d{2}\b", q)
    if m:
        try:
            y = int(m.group(0))
            if 1990 <= y <= 2100:
                year = y
        except Exception:
            pass
    return wanted, year

def get_field(c: Dict, key: str, default=None):
    return c.get(key, default)

def prefer_by_filters(ids: List[int], scores: List[float], chunks: List[Dict], issuers: set, year: Optional[int]) -> List[int]:
    """
    Re-rank/filter a wider candidate list to honor issuer/year hints.
    Strategy:
      - if issuers specified, keep only those whose chunk['issuer'] ∈ issuers
        (but if that empties, fall back to originals)
      - if year specified, prefer matching year (stable sort boost)
      - if multiple issuers, interleave so each gets representation
    """
    pairs = [(i, s) for i, s in zip(ids, scores) if i >= 0]
    # issuer filter
    if issuers:
        filt = [(i, s) for (i, s) in pairs if get_field(chunks[i], "issuer") in issuers]
        if filt:  # only apply if we still have candidates
            pairs = filt

    if not pairs:
        return []

    # year preference: stable boost
    if year is not None:
        prefers = []
        others = []
        for i, s in pairs:
            if get_field(chunks[i], "year") == year:
                prefers.append((i, s + 1e-3))  # tiny boost
            else:
                others.append((i, s))
        pairs = prefers + others

    # if multiple issuers requested, interleave to balance
    if len(issuers) >= 2:
        buckets: Dict[str, List[Tuple[int, float]]] = {}
        for i, s in pairs:
            key = get_field(chunks[i], "issuer")
            buckets.setdefault(key, []).append((i, s))
        # round-robin draw
        out: List[int] = []
        while len(out) < TOP_K and any(buckets.values()):
            for tick in sorted(buckets.keys()):
                if buckets[tick]:
                    out.append(buckets[tick].pop(0)[0])
                    if len(out) == TOP_K:
                        break
        return out

    # otherwise, take top_k by score order
    out = [i for i, _ in pairs[:TOP_K]]
    return out

# ---- RAG answer stitching (for non-LLM fallback and citations preview) ----
def format_answer(ids, scores, chunks, query: str):
    used = 0
    cites = []
    stitched_bits = []
    seen_pages = set()
    for idx, sc in zip(ids, scores):
        if idx < 0:
            continue
        c = chunks[idx]
        page_sig = (c.get("page_start"), c.get("page_end"), c.get("doc"))
        if page_sig in seen_pages:
            continue
        seen_pages.add(page_sig)
        sentences = best_sentences(query, c.get("text",""), max_sentences=2 if used == 0 else 1)
        if not sentences:
            continue
        stitched_bits.extend(sentences)
        preview = sentences[0][:240].replace("\n", " ")
        cites.append(f"- p.{c.get('page_start')}–{c.get('page_end')}: “{preview}…”")
        used += 1
        if used == 3:
            break
    stitched = " ".join(stitched_bits)[:600]
    return stitched, cites

def main():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        print("❌ Missing index/chunks. Build index first.")
        return

    index = faiss.read_index(str(INDEX_PATH))
    chunks = load_jsonl(CHUNKS_PATH)

    vocab = build_vocab_from_chunks(chunks)
    model = SentenceTransformer(MODEL_NAME)

    print("Type a question. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        # deterministic fixes + fuzzy autocorrect
        pre_q, pre_changes = apply_pre_replacements(q)
        fixed_q, changed, corr = autocorrect_query(pre_q, vocab)
        if pre_changes or changed:
            merged = {**pre_changes, **corr}
            print(f"(did you mean: {fixed_q}  — corrected {list(merged.items())})")
            q_use = fixed_q
        else:
            q_use = pre_q

        # macro answers first
        macro_answer, macro_cites = try_answer_macro(q_use)
        if macro_answer is not None:
            print("\nAssistant:", macro_answer)
            print()
            continue

        # issuer/year hints
        wanted_issuers, wanted_year = parse_requested_issuers_and_year(q_use)

        # embed & retrieve wider candidate pool
        q_vec = model.encode([q_use], normalize_embeddings=True).astype("float32")
        scores_all, ids_all = index.search(q_vec, CAND_K)
        scores_all, ids_all = scores_all[0], ids_all[0]

        if len(ids_all) == 0 or ids_all[0] < 0 or scores_all[0] < GOOD_SCORE:
            print("Assistant: 🤔 Insufficient context in the indexed filings. Try rephrasing or add more docs.\n")
            continue

        # apply issuer/year preferences
        chosen_ids = prefer_by_filters(list(ids_all), list(scores_all), chunks, wanted_issuers, wanted_year)
        if not chosen_ids:
            chosen_ids = list(ids_all[:TOP_K])

        # build a small context pack for the LLM
        contexts = []
        for i in chosen_ids:
            c = chunks[i]
            contexts.append({
                "doc": c.get("doc", "?"),
                "page_start": c.get("page_start", 1),
                "page_end": c.get("page_end", 1),
                "text": c.get("text", ""),
            })

        # compose RAG prompt with formatting awareness
        prompt = build_rag_prompt(q_use, contexts)
        llm_answer = generate_with_ollama(prompt).strip()

        # if the model gave nothing, fall back to stitched sentences
        if not llm_answer:
            stitched, cites = format_answer(chosen_ids, scores_all, chunks, q_use)
            if not stitched:
                print("Assistant: 🤔 Insufficient context.\n")
                continue
            print("\nAssistant:", stitched)
            print("\nCitations:")
            for c in cites:
                print(c)
            print(f"\n[confidence ~ {scores_all[0]:.3f}]\n")
            continue

        # Print LLM answer and synthetic “Sources” block from our chosen chunks
        print("\nAssistant:", llm_answer)
        print("\nSources:")
        # try to label each distinct doc once; keep order
        seen = set()
        for i in chosen_ids:
            c = chunks[i]
            doc = c.get("doc", "?")
            key = (doc, c.get("page_start"), c.get("page_end"))
            if key in seen:
                continue
            seen.add(key)
            print(f"[•] {doc} p.{c.get('page_start')}–{c.get('page_end')}")
        print(f"\n[confidence ~ {scores_all[0]:.3f}]\n")

if __name__ == "__main__":
    main()
