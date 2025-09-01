# app/chunk_pdf.py
from pathlib import Path
import json
import re
import pdfplumber

# === config ===
DOC_NAME = "Apple_10K_2023.pdf"
TARGET_WORDS = 900
OVERLAP_WORDS = 120

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "filings" / DOC_NAME
OUT_PATH = ROOT / "processed" / "chunks.jsonl"

def normalize_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    # keep single newlines; collapse multiple blank lines to one blank line
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_paragraphs(text: str):
    # split on blank lines only; keep paragraphs intact
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras

def word_count(s: str) -> int:
    return len(s.split())

def make_paragraph_corpus(pdf_path: Path):
    """Return (all_paras, para_page_map). Each para has a known page number."""
    all_paras = []
    para_page_map = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_no = i + 1  # 1-indexed
            text = (page.extract_text() or "")
            text = normalize_spaces(text)
            paras = split_into_paragraphs(text)
            all_paras.extend(paras)
            para_page_map.extend([page_no] * len(paras))
    return all_paras, para_page_map

def chunk_by_paragraph_index(all_paras, target_words=TARGET_WORDS, overlap_words=OVERLAP_WORDS):
    """
    Greedy pack paragraphs into chunks and return a list of (start_idx, end_idx) pairs,
    where indices refer to all_paras. Overlap is applied in paragraph units so indices
    stay consistent.
    """
    spans = []
    n = len(all_paras)
    i = 0
    while i < n:
        # start a new chunk at i
        total = 0
        j = i
        while j < n and (total + word_count(all_paras[j]) <= target_words or j == i):
            total += word_count(all_paras[j])
            j += 1
        # now we have a chunk [i, j) (end exclusive)
        start_idx = i
        end_idx_excl = j
        spans.append((start_idx, end_idx_excl))

        # compute next i with overlap (in words) translated to paragraphs
        # walk backward from j-1 adding paragraphs until overlap_words reached
        overlap_paras = 0
        acc_words = 0
        k = j - 1
        while k >= i and acc_words < overlap_words:
            acc_words += word_count(all_paras[k])
            overlap_paras += 1
            k -= 1
        # next start = end - overlap_paras (but not less than previous start + 1 to ensure progress)
        i = max(start_idx + 1, end_idx_excl - overlap_paras)
    return spans

def main():
    if not PDF_PATH.exists():
        print(f"❌ PDF not found: {PDF_PATH}")
        return

    all_paras, para_page_map = make_paragraph_corpus(PDF_PATH)

    if not all_paras:
        print("❌ No paragraphs extracted. The PDF may be mostly images.")
        return

    spans = chunk_by_paragraph_index(all_paras, TARGET_WORDS, OVERLAP_WORDS)

    # build output records using indices (no re-splitting)
    out_recs = []
    for (start_idx, end_idx_excl) in spans:
        paras = all_paras[start_idx:end_idx_excl]
        text = "\n\n".join(paras)
        page_start = para_page_map[start_idx]
        page_end = para_page_map[end_idx_excl - 1]
        rec = {
            "doc": DOC_NAME,
            "page_start": int(page_start),
            "page_end": int(page_end),
            "n_words": word_count(text),
            "text": text
        }
        out_recs.append(rec)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in out_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(out_recs)} chunks → {OUT_PATH}")

if __name__ == "__main__":
    main()
