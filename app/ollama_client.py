# app/ollama_client.py
"""
Tiny Ollama client used by qa_cli.py

- Exposes: generate_with_ollama(prompt, model=None, options=None, timeout=60) -> str
- Uses only urllib (no extra deps)
- Reads defaults from environment:
    OLLAMA_URL   (default: http://localhost:11434)
    OLLAMA_MODEL (default: qwen2.5:3b-instruct)
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")


def generate_with_ollama(
    prompt: str,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> str:
    """
    Send a non-streaming /api/generate request to Ollama and return the response text.
    Returns "" on error (qa_cli.py will fall back to stitched text).
    """
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            return (obj.get("response") or "").strip()
    except urllib.error.URLError:
        # server not running / network issue
        return ""
    except Exception:
        return ""


if __name__ == "__main__":
    # quick self-test
    out = generate_with_ollama("Say 'hello' and stop.", options={"temperature": 0.2, "num_ctx": 2048})
    print("Ollama replied:", out or "(no response)")
