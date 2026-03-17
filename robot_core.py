# ── robot_core.py ─────────────────────────────────────────────────────────────
import json
import re
import uuid
import numpy as np
from collections import deque
from typing import Optional
import requests

from config import (
    OLLAMA_URL, MODEL_NAME, TOP_K, MAX_CONTEXT_CHARS,
    MAX_TOKENS, TEMPERATURE, MAX_HISTORY, SEMANTIC_THRESHOLD
)
from utils import split_into_chunks, preprocess_input, postprocess_output


# ── RAG ENGINE ────────────────────────────────────────────────────────────────

class RAGEngine:
    """Hybrid TF-IDF (bigrams) + keyword overlap retrieval."""

    def __init__(self):
        self.chunks     = []
        self.vectorizer = None
        self.vectors    = None

    def _build_index(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            self.vectors    = self.vectorizer.fit_transform(self.chunks)
        except ImportError:
            print("sklearn not found — keyword search only")
            self.vectorizer = None
            self.vectors    = None

    def ingest(self, text: str, chunk_size: int = 400) -> int:
        new = split_into_chunks(text, chunk_size)
        self.chunks.extend(new)
        if self.chunks:
            self._build_index()
        return len(new)

    def _keyword_score(self, query_tokens: list, chunk: str) -> float:
        chunk_lower = chunk.lower()
        hits = sum(1 for t in query_tokens if t in chunk_lower)
        return hits / (len(query_tokens) or 1)

    def retrieve(self, query: str) -> list:
        if not self.chunks:
            return []
        q_tokens = re.findall(r"[a-z0-9]+", query.lower())
        if self.vectorizer and self.vectors is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self.vectorizer.transform([query])
            tfidf = cosine_similarity(q_vec, self.vectors)[0]
        else:
            tfidf = [0.0] * len(self.chunks)
        scored = [
            (float(tfidf[i]) + 0.3 * self._keyword_score(q_tokens, c), c)
            for i, c in enumerate(self.chunks)
        ]
        scored.sort(reverse=True)
        return [c for score, c in scored[:TOP_K] if score > 0]

    def clear(self):
        self.chunks     = []
        self.vectorizer = None
        self.vectors    = None


# ── SEMANTIC DICTIONARY MATCHER ───────────────────────────────────────────────

class SemanticMatcher:
    """
    Embeds dictionary questions once on startup.
    Matches user queries by cosine similarity — handles synonyms and
    paraphrasing. Falls back to keyword overlap if sentence-transformers
    is not installed.
    """

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
            print("  Loading sentence-transformers (all-MiniLM-L6-v2)...")
            self.model      = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.model.encode(
                self.questions,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            print(f"  Semantic matcher ready — {len(self.questions)} questions embedded")
        except ImportError:
            print("  sentence-transformers not installed — using keyword fallback")
            print("  Install: pip install sentence-transformers")

    def match(self, user_input: str) -> Optional[str]:
        cleaned = preprocess_input(user_input)

        # ── Semantic path ──
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

        # ── Keyword fallback ──
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


# ── CONVERSATION MEMORY ───────────────────────────────────────────────────────

class MemoryStore:
    def __init__(self):
        self._store: dict = {}

    def get(self, session_id: str) -> list:
        return list(self._store.get(session_id, deque(maxlen=MAX_HISTORY)))

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self._store:
            self._store[session_id] = deque(maxlen=MAX_HISTORY)
        self._store[session_id].append({"role": role, "content": content})

    def clear(self, session_id: str):
        self._store.pop(session_id, None)

    def count(self) -> int:
        return len(self._store)


# ── SYSTEM PROMPT BUILDER ─────────────────────────────────────────────────────

def build_system_prompt(prompts: dict) -> str:
    """
    Injects all team info, robot specs, AND game info into the system prompt
    so the AI always has full context even without RAG chunks.
    """
    info  = prompts.get("team_info",  {})
    robot = prompts.get("robot_info", {})
    game  = prompts.get("game_info",  {})
    base  = prompts.get("system_prompt", "You are a helpful FRC pit assistant.")
    # Override with a strict, no-nonsense prompt
    base = (
        "You are Avacado, the pit assistant for FRC Team 10600 Two Steps Ahead. "
        "Answer in exactly 1-2 sentences. Neutral tone. No enthusiasm, no filler phrases, no extra context. "
        "Use only the facts provided. Do not mention yourself or the team name unless directly asked. "
        "If the answer is not in the provided information say exactly: "
        "\"Great question. I don't have that information. Ask one of our team members at the pit.\""
    )

    knowledge = (
        f"\n\n=== TEAM INFORMATION ===\n"
        f"Team Name    : {info.get('team_name', 'N/A')}\n"
        f"Team Number  : {info.get('team_number', 'N/A')}\n"
        f"Location     : {info.get('location', 'N/A')}\n"
        f"Members      : {info.get('members', 'N/A')}\n"
        f"Subteams     : {info.get('subteams', 'N/A')}\n"
        f"Awards       : {info.get('awards', 'N/A')}\n"
        f"Outreach     : {info.get('outreach', 'N/A')}\n"
        f"Email        : {info.get('email', 'N/A')}\n"
        f"Instagram    : {info.get('instagram', 'N/A')}\n"
        f"Website      : {info.get('website', 'N/A')}\n\n"
        f"=== ROBOT INFORMATION ===\n"
        f"Robot Name   : {robot.get('robot_name', 'N/A')} ({robot.get('season', 'N/A')} season)\n"
        f"Drivetrain   : {robot.get('drivetrain', 'N/A')}\n"
        f"Top Speed    : {robot.get('top_speed', 'N/A')}\n"
        f"Battery      : {robot.get('battery', 'N/A')}\n"
        f"Scoring      : {robot.get('scoring', 'N/A')}\n"
        f"Intake       : {robot.get('intake', 'N/A')}\n"
        f"Shooter      : {robot.get('shooter', 'N/A')}\n"
        f"Language     : {robot.get('language', 'N/A')}\n"
        f"Autonomous   : {robot.get('autonomous', 'N/A')}\n"
        f"Data Logging : {robot.get('data_logging', 'N/A')}\n\n"
        f"=== GAME INFORMATION ===\n"
        f"Game         : {game.get('game_name', 'N/A')}\n"
        f"Objective    : {game.get('objective', 'N/A')}\n"
        f"Game Pieces  : {game.get('game_pieces', 'N/A')}\n"
        f"HUB          : {game.get('hub', 'N/A')}\n"
        f"Autonomous   : {game.get('autonomous', 'N/A')}\n"
        f"Teleop       : {game.get('teleop', 'N/A')}\n"
        f"Endgame      : {game.get('endgame', 'N/A')}\n"
        f"Ranking Pts  : {game.get('ranking_points', 'N/A')}\n"
    )
    return base + knowledge


# ── OLLAMA CALL ───────────────────────────────────────────────────────────────

def ask_ollama(
    question:       str,
    context_chunks: list,
    session_id:     str,
    memory:         MemoryStore,
    system_prompt:  str
):
    context_text = "\n\n".join(context_chunks)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]

    user_content = (
        f"Context from documents:\n{context_text}\n\nQuestion: {question}"
        if context_text else
        f"Question: {question}"
    )

    history  = memory.get(session_id)
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages += [{"role": "user", "content": user_content}]

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":    MODEL_NAME,
                "messages": messages,
                "options":  {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS
                },
                "stream": True
            },
            stream=True,
            timeout=(10, 180)
        )

        full_reply = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                full_reply += token
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue

        if not full_reply.strip():
            return None, "Empty response from model"

        reply = postprocess_output(full_reply)

        # Save to memory only on real answers
        if "don't have that information" not in reply.lower():
            memory.add(session_id, "user",      question)
            memory.add(session_id, "assistant", reply)

        return reply, None

    except requests.exceptions.ConnectTimeout:
        return None, "Cannot connect to Ollama. Run: ollama serve"
    except requests.exceptions.ReadTimeout:
        return None, "Ollama timed out. Try a shorter question."
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach Ollama. Run: ollama serve"
    except Exception as e:
        return None, str(e)