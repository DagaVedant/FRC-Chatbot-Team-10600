import re

def preprocess_input(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip("?!.,;:")
    return text

def postprocess_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags = re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 400) -> list:
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None
    except ImportError:
        nlp = None

    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    prev_tail = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        block = (prev_tail + " " + p).strip() if prev_tail else p

        if nlp:
            doc = nlp(block)
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        else:
            sentences = re.split(r"(?<=[.!?])\s+", block)
            sentences = [s.strip() for s in sentences if s.strip()]

        if len(block) <= chunk_size:
            chunks.append(block)
            prev_tail = sentences[-1] if sentences else ""
        else:
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
