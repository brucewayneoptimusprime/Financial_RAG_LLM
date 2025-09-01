# app/formatting.py
import re

TOC_HINTS = {"table of contents", "exhibit", "index of", "item 1."}
MIN_SENT_LEN = 40    # skip super-short heading lines
MAX_SENT_LEN = 350   # avoid overly long rambles

def normalize(text: str) -> str:
    # collapse spaces, keep newlines meaningful
    t = text.replace("\xa0", " ")
    # join hyphenated line-breaks like "informa-\ntion" -> "information"
    t = re.sub(r"-\s*\n\s*", "", t)
    # turn single newlines inside paragraphs into spaces; keep blank lines as paragraph breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def split_sentences(text: str):
    # very simple sentence splitter
    text = normalize(text)
    # protect abbreviations a bit: "U.S." etc
    text = re.sub(r"(\b[A-Z]\.){2,}", lambda m: m.group(0).replace(".", "∯"), text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.replace("∯", ".").strip() for s in parts if s.strip()]
    # length filter
    sents = [s for s in sents if MIN_SENT_LEN <= len(s) <= MAX_SENT_LEN]
    return sents

def looks_like_toc(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in TOC_HINTS)

def keyword_score(query: str, sentence: str) -> int:
    q_words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", query)}
    s_words = {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", sentence)}
    return len(q_words & s_words)

def best_sentences(query: str, chunk_text: str, max_sentences: int = 2):
    """Return up to N sentences from the chunk that best match query words, skipping TOC-like lines."""
    if looks_like_toc(chunk_text):
        return []

    sents = split_sentences(chunk_text)
    if not sents:
        return []

    # score each sentence by keyword overlap
    scored = [(keyword_score(query, s), i, s) for i, s in enumerate(sents)]
    # prefer higher score; stable by original order
    scored.sort(key=lambda x: (-x[0], x[1]))

    # if no keyword overlap at all, fallback to first 1–2 clean sentences
    if scored[0][0] == 0:
        return sents[:max_sentences]

    picked = [s for _, _, s in scored[:max_sentences]]
    return picked
