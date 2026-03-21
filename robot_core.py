# ── robot_core.py ─────────────────────────────────────────────────────────────
import json
import re
import time
import hashlib
import numpy as np
from collections import deque, OrderedDict
from typing import Optional
import requests

from config import (
    OLLAMA_URL, MODEL_NAME, TOP_K, MAX_CONTEXT_CHARS,
    MAX_TOKENS, TEMPERATURE, MAX_HISTORY, SEMANTIC_THRESHOLD,
    RESPONSE_CACHE_SIZE, CONFIDENCE_THRESHOLD,
    SESSION_EXPIRY, SUMMARY_THRESHOLD, CHROMA_PATH
)
from utils import split_into_chunks, preprocess_input, postprocess_output


# ── RAG ENGINE — ChromaDB with TF-IDF fallback ────────────────────────────────

class RAGEngine:
    """
    Primary: ChromaDB with sentence-transformers embeddings (semantic search).
    Fallback: TF-IDF + keyword hybrid (if ChromaDB not installed).
    Also returns a confidence score for each retrieval.
    """

    def __init__(self):
        self.chunks     = []
        self.vectorizer = None
        self.vectors    = None
        self._chroma    = None
        self._collection = None
        self._use_chroma = False
        self._embed_model = None

    def _init_chroma(self):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            import os
            os.makedirs(CHROMA_PATH, exist_ok=True)
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            self._collection  = client.get_or_create_collection(
                name="robot_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_chroma  = True
            print("  [ok] ChromaDB initialised")
        except ImportError:
            print("  [info] ChromaDB not installed — using TF-IDF fallback")
            self._use_chroma = False

    def _build_tfidf(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            self.vectors    = self.vectorizer.fit_transform(self.chunks)
        except ImportError:
            self.vectorizer = None
            self.vectors    = None

    def ingest(self, text: str, chunk_size: int = 400) -> int:
        if self._collection is None:
            self._init_chroma()

        new_chunks = split_into_chunks(text, chunk_size)
        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)

        if self._use_chroma and self._embed_model:
            embeddings = self._embed_model.encode(
                new_chunks, convert_to_numpy=True, normalize_embeddings=True
            ).tolist()
            start_id = self._collection.count()
            self._collection.add(
                documents  = new_chunks,
                embeddings = embeddings,
                ids        = [f"chunk_{start_id + i}" for i in range(len(new_chunks))]
            )
        else:
            self._build_tfidf()

        return len(new_chunks)

    def _keyword_score(self, query_tokens: list, chunk: str) -> float:
        chunk_lower = chunk.lower()
        hits = sum(1 for t in query_tokens if t in chunk_lower)
        return hits / (len(query_tokens) or 1)

    def retrieve(self, query: str) -> tuple:
        """Returns (list of chunks, confidence score 0-1)."""
        if not self.chunks:
            return [], 0.0

        q_tokens = re.findall(r"[a-z0-9]+", query.lower())

        # ChromaDB semantic search
        if self._use_chroma and self._collection and self._embed_model:
            try:
                q_emb = self._embed_model.encode(
                    [query], convert_to_numpy=True, normalize_embeddings=True
                ).tolist()
                results = self._collection.query(
                    query_embeddings=q_emb,
                    n_results=min(TOP_K, self._collection.count())
                )
                docs      = results["documents"][0]
                distances = results["distances"][0]
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score 0-1
                confidence = max(0.0, 1.0 - (distances[0] / 2)) if distances else 0.0

                # Boost with keyword overlap
                scored = []
                for doc, dist in zip(docs, distances):
                    sim = 1.0 - (dist / 2)
                    kw  = self._keyword_score(q_tokens, doc)
                    scored.append((sim + 0.2 * kw, doc))
                scored.sort(reverse=True)
                return [c for _, c in scored], confidence
            except Exception as e:
                print(f"  [warn] ChromaDB query failed: {e}, falling back")

        # TF-IDF fallback
        if self.vectorizer and self.vectors is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec  = self.vectorizer.transform([query])
            tfidf  = cosine_similarity(q_vec, self.vectors)[0]
        else:
            tfidf = [0.0] * len(self.chunks)

        scored = [
            (float(tfidf[i]) + 0.3 * self._keyword_score(q_tokens, c), c)
            for i, c in enumerate(self.chunks)
        ]
        scored.sort(reverse=True)
        top_score  = scored[0][0] if scored else 0.0
        confidence = min(top_score, 1.0)
        return [c for s, c in scored[:TOP_K] if s > 0], confidence

    def clear(self):
        self.chunks     = []
        self.vectorizer = None
        self.vectors    = None
        if self._collection:
            try:
                import chromadb
                import os
                client = chromadb.PersistentClient(path=CHROMA_PATH)
                client.delete_collection("robot_knowledge")
                self._collection = client.get_or_create_collection(
                    name="robot_knowledge",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass


# ── SEMANTIC DICTIONARY MATCHER ───────────────────────────────────────────────

class SemanticMatcher:
    def __init__(self, dictionary: dict):
        self.dictionary = dictionary
        self.questions  = list(dictionary.keys())
        self.answers    = list(dictionary.values())
        self.embeddings = None
        self.model      = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            print("  Loading sentence-transformers...")
            self.model      = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.model.encode(
                self.questions, convert_to_numpy=True, normalize_embeddings=True
            )
            print(f"  [ok] Semantic matcher ready — {len(self.questions)} questions")
        except ImportError:
            print("  [warn] sentence-transformers not installed — keyword fallback")

    def match(self, user_input: str) -> Optional[str]:
        cleaned = preprocess_input(user_input)

        if self.model and self.embeddings is not None:
            q_vec    = self.model.encode(
                [cleaned], convert_to_numpy=True, normalize_embeddings=True
            )
            sims     = np.dot(q_vec, self.embeddings.T)[0]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= SEMANTIC_THRESHOLD:
                return self.answers[best_idx]
            return None

        # Keyword fallback
        tokens = set(re.findall(r"[a-z0-9]+", cleaned))
        best_score, best_answer = 0, None
        for q, a in self.dictionary.items():
            q_tokens = set(re.findall(r"[a-z0-9]+", q.lower()))
            if not q_tokens:
                continue
            overlap = len(tokens & q_tokens) / len(q_tokens)
            if overlap > best_score:
                best_score, best_answer = overlap, a
        return best_answer if best_score >= 0.55 else None


# ── CONVERSATION MEMORY WITH SUMMARISATION ────────────────────────────────────

class MemoryStore:
    """
    Per-session rolling history with automatic summarisation.
    When history hits SUMMARY_THRESHOLD messages, older exchanges
    are compressed into a summary so context stays useful without
    growing indefinitely.
    """

    def __init__(self):
        self._store:    dict = {}   # session_id -> deque of messages
        self._summary:  dict = {}   # session_id -> summary string
        self._last_used: dict = {}  # session_id -> timestamp

    def _expired(self, session_id: str) -> bool:
        last = self._last_used.get(session_id, 0)
        return (time.time() - last) > SESSION_EXPIRY

    def _cleanup(self):
        expired = [sid for sid in self._store if self._expired(sid)]
        for sid in expired:
            self._store.pop(sid, None)
            self._summary.pop(sid, None)
            self._last_used.pop(sid, None)

    def get(self, session_id: str) -> list:
        self._cleanup()
        if session_id not in self._store:
            return []
        msgs = list(self._store[session_id])
        summary = self._summary.get(session_id)
        if summary:
            return [{"role": "system", "content": f"Earlier conversation summary: {summary}"}] + msgs
        return msgs

    def add(self, session_id: str, role: str, content: str):
        self._cleanup()
        if session_id not in self._store:
            self._store[session_id] = deque(maxlen=MAX_HISTORY)
        self._store[session_id].append({"role": role, "content": content})
        self._last_used[session_id] = time.time()

        # Summarise if history is getting long
        if len(self._store[session_id]) >= SUMMARY_THRESHOLD:
            self._summarise(session_id)

    def _summarise(self, session_id: str):
        """Compress the oldest half of messages into a summary."""
        msgs = list(self._store[session_id])
        half = len(msgs) // 2
        to_summarise = msgs[:half]
        keep         = msgs[half:]

        # Build a simple extractive summary
        pairs = []
        for i in range(0, len(to_summarise) - 1, 2):
            if to_summarise[i]["role"] == "user" and i + 1 < len(to_summarise):
                q = to_summarise[i]["content"][:80]
                a = to_summarise[i+1]["content"][:120]
                pairs.append(f"Q: {q} → A: {a}")

        if pairs:
            prev = self._summary.get(session_id, "")
            self._summary[session_id] = (prev + " | " if prev else "") + " | ".join(pairs)

        self._store[session_id] = deque(keep, maxlen=MAX_HISTORY)

    def clear(self, session_id: str):
        self._store.pop(session_id, None)
        self._summary.pop(session_id, None)
        self._last_used.pop(session_id, None)

    def count(self) -> int:
        self._cleanup()
        return len(self._store)


# ── RESPONSE CACHE ────────────────────────────────────────────────────────────

class ResponseCache:
    """
    LRU cache for AI responses.
    Key = hash of (question + top chunk).
    Instant return for repeated questions.
    """

    def __init__(self, maxsize: int = RESPONSE_CACHE_SIZE):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def _key(self, question: str, chunks: list) -> str:
        top = chunks[0] if chunks else ""
        raw = (question.lower().strip() + top[:100])
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, question: str, chunks: list) -> Optional[str]:
        key = self._key(question, chunks)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, question: str, chunks: list, response: str):
        key = self._key(question, chunks)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = response
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


# ── RATE LIMITER ──────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Sliding window rate limiter per IP address.
    """

    def __init__(self):
        self._windows: dict = {}   # ip -> deque of timestamps

    def is_allowed(self, ip: str) -> bool:
        from config import RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
        now = time.time()
        if ip not in self._windows:
            self._windows[ip] = deque()
        window = self._windows[ip]
        # Remove old timestamps
        while window and now - window[0] > RATE_LIMIT_WINDOW:
            window.popleft()
        if len(window) >= RATE_LIMIT_REQUESTS:
            return False
        window.append(now)
        return True


# ── REQUEST QUEUE ─────────────────────────────────────────────────────────────

import asyncio

class RequestQueue:
    """
    Async semaphore-based queue so concurrent requests are
    processed in order without one blocking another.
    Max 3 concurrent Ollama calls.
    """
    def __init__(self, max_concurrent: int = 3):
        self._sem = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, *args):
        self._sem.release()


# ── SYSTEM PROMPT BUILDER ─────────────────────────────────────────────────────

def build_system_prompt(prompts: dict) -> str:
    info  = prompts.get("team_info",  {})
    robot = prompts.get("robot_info", {})
    game  = prompts.get("game_info",  {})
    base  = (
        "You are Avocado, the pit assistant for FRC Team 10600 Two Steps Ahead. "
        "Answer in exactly 1-2 sentences. Neutral tone. No enthusiasm, no filler, no extra context. "
        "Use only the facts in the provided context and team info. "
        "Do not mention yourself or the team name unless directly asked. "
        "Do not repeat the question back. "
        "If the answer is not available say exactly: "
        "\"Great question. I don't have that information. Ask one of our team members at the pit.\""
    )
    knowledge = (
        f"\n\n=== TEAM INFORMATION ===\n"
        f"Team Name    : {info.get('team_name','N/A')}\n"
        f"Team Number  : {info.get('team_number','N/A')}\n"
        f"Location     : {info.get('location','N/A')}\n"
        f"Members      : {info.get('members','N/A')}\n"
        f"Subteams     : {info.get('subteams','N/A')}\n"
        f"Awards       : {info.get('awards','N/A')}\n"
        f"Outreach     : {info.get('outreach','N/A')}\n"
        f"Email        : {info.get('email','N/A')}\n"
        f"Instagram    : {info.get('instagram','N/A')}\n"
        f"Website      : {info.get('website','N/A')}\n\n"
        f"=== ROBOT INFORMATION ===\n"
        f"Robot Name   : {robot.get('robot_name','N/A')} ({robot.get('season','N/A')})\n"
        f"Drivetrain   : {robot.get('drivetrain','N/A')}\n"
        f"Top Speed    : {robot.get('top_speed','N/A')}\n"
        f"Battery      : {robot.get('battery','N/A')}\n"
        f"Scoring      : {robot.get('scoring','N/A')}\n"
        f"Intake       : {robot.get('intake','N/A')}\n"
        f"Shooter      : {robot.get('shooter','N/A')}\n"
        f"Language     : {robot.get('language','N/A')}\n"
        f"Autonomous   : {robot.get('autonomous','N/A')}\n"
        f"Data Logging : {robot.get('data_logging','N/A')}\n\n"
        f"=== GAME INFORMATION ===\n"
        f"Game         : {game.get('game_name','N/A')}\n"
        f"Objective    : {game.get('objective','N/A')}\n"
        f"Game Pieces  : {game.get('game_pieces','N/A')}\n"
        f"Autonomous   : {game.get('autonomous','N/A')}\n"
        f"Teleop       : {game.get('teleop','N/A')}\n"
        f"Endgame      : {game.get('endgame','N/A')}\n"
    )
    return base + knowledge


# ── OLLAMA ASYNC STREAM ───────────────────────────────────────────────────────

async def stream_ollama(question: str, context_chunks: list,
                        session_id: str, memory: MemoryStore,
                        system_prompt: str):
    """Async generator — yields text tokens as they arrive via httpx."""
    import httpx
    context_text = "\n\n".join(context_chunks)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]

    user_content = (
        f"Context:\n{context_text}\n\nQuestion: {question}"
        if context_text else f"Question: {question}"
    )

    history  = memory.get(session_id)
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages += [{"role": "user", "content": user_content}]

    full_reply = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=180)) as client:
            async with client.stream(
                "POST", OLLAMA_URL,
                json={
                    "model":    MODEL_NAME,
                    "messages": messages,
                    "options":  {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
                    "stream":   True
                }
            ) as resp:
                import json as _json
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = _json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        full_reply += token
                        yield token
                        if chunk.get("done"):
                            break
                    except _json.JSONDecodeError:
                        continue
    except httpx.ConnectTimeout:
        yield "\n[Error: Cannot connect to Ollama. Run: ollama serve]"
        return
    except httpx.ReadTimeout:
        yield "\n[Error: Ollama timed out.]"
        return
    except Exception as e:
        yield f"\n[Error: {e}]"
        return

    if full_reply.strip() and "don't have that information" not in full_reply.lower():
        memory.add(session_id, "user",      question)
        memory.add(session_id, "assistant", postprocess_output(full_reply))