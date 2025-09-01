# app/text_utils.py
import re
import difflib
import string
from collections import Counter

WORD_RE = re.compile(r"[a-zA-Z]{3,}")

# Words we will never try to correct (common words & domain terms)
# Words we will never try to correct (common words & domain terms)
# Words we will never try to correct (common words & domain terms)
NEVER_CORRECT = {
    "main", "risk", "risks", "factors", "factor",
    "describe", "apple", "apple's", "what", "are", "the",
    "latest", "yoy", "rate", "level", "index", "cpi", "unemployment",
    # formatting keywords (do not autocorrect these)
    "bullet", "bullets", "point", "points", "list", "listed",
    "numbered", "numbers", "items", "sentences", "lines", "newline", "newlines"
}



# Known typo → fix (deterministic)
DIRECT_REPLACEMENTS = {
    "rish": "risk",
    "facors": "factors",
    "facotr": "factor",
    "finacial": "financial",
    "operatons": "operations",
    "conditon": "condition",
    "enviroment": "environment",
}

def tokenize(text: str):
    return WORD_RE.findall(text.lower())

def build_vocab_from_chunks(chunks, max_words: int = 5000):
    cnt = Counter()
    for c in chunks:
        cnt.update(tokenize(c.get("text", "")))
    vocab = [w for w, _ in cnt.most_common(max_words)]
    return set(vocab)

def _is_titlecase(word: str) -> bool:
    # e.g., "Apple"
    return len(word) >= 2 and word[0].isupper() and word[1:].islower()

def _is_possessive(word: str) -> bool:
    # e.g., "Apple's"
    return "'" in word

def _strip_punct(tok: str) -> str:
    # Remove surrounding punctuation like "factors?" -> "factors"
    return tok.strip(string.punctuation)

def autocorrect_query(query: str, vocab: set[str], max_corrections: int = 2):
    """
    Conservative autocorrect:
      - strip trailing punctuation when comparing
      - never touch common/domain words (NEVER_CORRECT)
      - prefer deterministic DIRECT_REPLACEMENTS
      - fuzzy match only with stricter cutoff
      - at most `max_corrections` changes
    Returns (fixed_query, did_change, corrections_dict).
    """
    words = query.split()
    changed = False
    corrections = {}
    vocab_list = list(vocab)

    for i, w in enumerate(words):
        tok_raw = w.lower()
        tok = _strip_punct(tok_raw)

        # skip empties after stripping
        if not tok:
            continue

        # never correct certain common/domain words
        if tok in NEVER_CORRECT:
            continue

        # skip short, possessive, proper nouns, or already-known vocab
        if len(tok) < 4 or _is_possessive(w) or _is_titlecase(w) or tok in vocab:
            continue

        new = None

        # 1) deterministic map
        if tok in DIRECT_REPLACEMENTS:
            new = DIRECT_REPLACEMENTS[tok]
        else:
            # 2) fuzzy match (stricter)
            cand = difflib.get_close_matches(tok, vocab_list, n=1, cutoff=0.75)
            if cand:
                new = cand[0]

        if new and new != tok:
            # preserve case only if original was all lower; otherwise just use new
            words[i] = new if w.islower() else new
            corrections[w] = words[i]
            changed = True
            if len(corrections) >= max_corrections:
                break

    fixed_q = " ".join(words)

    # If nothing effectively changed, suppress correction notice
    if fixed_q.strip() == query.strip():
        changed = False
        corrections = {}

    return fixed_q, changed, corrections
