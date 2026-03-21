# ── utils.py ─────────────────────────────────────────────────────────────────
import re


def preprocess_input(text: str) -> str:
    """Clean and normalise user input before matching or sending to the model."""
    text = text.strip().lower()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # remove trailing punctuation that can break matching
    text = text.rstrip("?!.,;:")
    return text


def postprocess_output(text: str) -> str:
    """Clean model output before sending to the frontend."""
    text = text.strip()
    # remove any <think> or reasoning blocks some models emit
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # strip markdown bold
    text = re.sub(r"\*(.*?)\*",     r"\1", text)   # strip markdown italic
    text = re.sub(r"`(.*?)`",       r"\1", text)   # strip inline code
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse excessive newlines
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 400) -> list[str]:
    """
    Split text into overlapping chunks.
    Each chunk starts with the last sentence of the previous one so facts
    at paragraph boundaries are never lost.
    """
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks     = []
    prev_tail  = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        block = (prev_tail + " " + p).strip() if prev_tail else p

        if len(block) <= chunk_size:
            chunks.append(block)
            sentences = re.split(r"(?<=[.!?])\s+", block)
            prev_tail = sentences[-1] if sentences else ""
        else:
            sentences = re.split(r"(?<=[.!?])\s+", block)
            cur, length = [], 0
            for s in sentences:
                if length + len(s) < chunk_size:
                    cur.append(s)
                    length += len(s)
                else:
                    if cur:
                        chunks.append(" ".join(cur))
                    cur, length = [s], len(s)
            if cur:
                chunks.append(" ".join(cur))
            prev_tail = sentences[-1] if sentences else ""

    return [c for c in chunks if len(c) > 30]