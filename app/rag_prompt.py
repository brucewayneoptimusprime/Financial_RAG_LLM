# app/rag_prompt.py
from typing import List, Dict, Tuple, Optional
import re

def _infer_style_and_count(question: str) -> Tuple[str, Optional[int]]:
    """
    Infer desired output style from the user's question.
    Returns (style, n) where style ∈ {"bullets","numbered","lines","prose"}.
    n is the requested count if we can detect it (else None).
    """
    q = question.lower().strip()

    # try to detect an explicit count
    n = None
    # common phrasings: "top 5", "give me 3", "list 4", "in 3 sentences"
    m = re.search(r"(top|give me|list|into|in|show|summarize|provide)\s+(\d{1,2})", q)
    if m:
        try:
            n = int(m.group(2))
        except Exception:
            n = None
    else:
        # "3 bullets", "3 points", "3 numbered", "3 sentences"
        m2 = re.search(r"\b(\d{1,2})\s+(bullets?|points?|items?|numbered|sentences?)\b", q)
        if m2:
            try:
                n = int(m2.group(1))
            except Exception:
                n = None

    # style hints
    if any(k in q for k in ["bullet", "bulleted", "bullet points", "points", "as a list"]) or q.startswith("list "):
        return "bullets", n
    if any(k in q for k in ["numbered", "1.", "2.", "3."]) or re.search(r"\b(top\s+\d+)\b", q):
        # treat "top N" as numbered if not explicitly bullets
        return "numbered", n
    if any(k in q for k in ["each on a new line", "separate lines", "new line", "new lines", "line by line"]):
        return "lines", n
    # fallback
    return "prose", n


def _format_instruction(style: str, n: Optional[int]) -> str:
    """
    Build a strict formatting instruction for the LLM.
    """
    # sensible defaults
    count = n if (n and 1 <= n <= 10) else None

    if style == "bullets":
        base = [
            "Format the answer strictly as bullet points using '- ' at the start of each line.",
            "Each bullet should be one concise sentence.",
        ]
        if count:
            base.append(f"Write exactly {count} bullet points.")
        else:
            base.append("Write 3–5 bullet points.")
        base.append("Do not include any preamble or conclusion—bullets only.")
        base.append("End each bullet with source markers like [1], [2] based on the context items used.")
        return " ".join(base)

    if style == "numbered":
        base = [
            "Format the answer strictly as a numbered list using '1.', '2.', '3.' at the start of lines.",
            "Each item should be one concise sentence.",
        ]
        if count:
            base.append(f"Write exactly {count} numbered items.")
        else:
            base.append("Write 3–5 numbered items.")
        base.append("Do not include any preamble or conclusion—list only.")
        base.append("End each item with source markers like [1], [2] based on the context items used.")
        return " ".join(base)

    if style == "lines":
        base = [
            "Write clear sentences, each on its own new line.",
        ]
        if count:
            base.append(f"Write exactly {count} sentences.")
        else:
            base.append("Write 3 sentences.")
        base.append("Append source markers like [1], [2] at the end of sentences where relevant.")
        return " ".join(base)

    # prose default
    return (
        "Write a concise answer in 3–5 sentences. "
        "Respect any formatting hints in the question. "
        "Include source markers like [1], [2] inline where appropriate."
    )


def build_rag_prompt(question: str, contexts: List[Dict], max_chars_per_chunk: int = 1200) -> str:
    """
    Build a retrieval-augmented prompt with strict formatting control.

    contexts item shape:
      {
        "doc": str,
        "page_start": int,
        "page_end": int,
        "text": str
      }
    """
    # 1) Prepare context block
    items = []
    for i, c in enumerate(contexts, start=1):
        snippet = (c.get("text") or "")[:max_chars_per_chunk]
        snippet = snippet.replace("\n", " ").strip()
        items.append(f"[{i}] (Doc: {c.get('doc','?')}, pages {c.get('page_start','?')}-{c.get('page_end','?')})\n{snippet}")
    ctx_block = "\n\n".join(items)

    # 2) Infer style from question and craft strict instruction
    style, n = _infer_style_and_count(question)
    style_instr = _format_instruction(style, n)

    # 3) System guidance
    system_rules = (
        "You are a financial document assistant. "
        "Use ONLY the provided context to answer. "
        "If the context is insufficient to answer, reply exactly: \"Insufficient context.\" "
        "Never invent facts. "
        "Always include inline source markers like [1], [2] that correspond to the context items used. "
        "Follow the requested output formatting strictly."
    )

    # 4) Final prompt
    prompt = (
        f"{system_rules}\n\n"
        f"Formatting requirements: {style_instr}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{ctx_block}\n\n"
        "Answer:"
    )
    return prompt
