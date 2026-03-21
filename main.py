# ── main.py ───────────────────────────────────────────────────────────────────
import os
import uuid
import json
import time
import random
import asyncio
import sqlite3
import requests
import httpx
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pypdf import PdfReader

# Always run relative to this file so paths work from anywhere
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    PORT, DATA_FILE, GAME_MANUAL_FILE, DICTIONARY_FILE,
    PROMPTS_FILE, CHUNK_SIZE_ROBOT, CHUNK_SIZE_MANUAL, MODEL_NAME,
    OLLAMA_URL
)
from robot_core import RAGEngine, SemanticMatcher, MemoryStore, build_system_prompt


# ── FILE LOADERS ──────────────────────────────────────────────────────────────

def load_text(path: str) -> str:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path: str) -> str:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return ""
    try:
        reader = PdfReader(path)
        pages  = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        result = "\n\n".join(pages)
        print(f"  [ok] {path} — {len(reader.pages)} pages, {len(result)} chars")
        return result
    except Exception as e:
        print(f"  [error] reading PDF: {e}")
        return ""

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── ANALYTICS DB ──────────────────────────────────────────────────────────────

DB_PATH = "data/analytics.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT    NOT NULL,
            message   TEXT    NOT NULL,
            source    TEXT    NOT NULL,
            answered  INTEGER NOT NULL DEFAULT 1,
            think_sec REAL
        )
    """)
    conn.commit()
    conn.close()

def log_question(message: str, source: str, answered: bool, think_sec: float):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO questions (ts, message, source, answered, think_sec) VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(), message, source, int(answered), think_sec)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  [warn] analytics log failed: {e}")

def get_analytics():
    try:
        conn = sqlite3.connect(DB_PATH)
        total     = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        answered  = conn.execute("SELECT COUNT(*) FROM questions WHERE answered=1").fetchone()[0]
        dict_hits = conn.execute("SELECT COUNT(*) FROM questions WHERE source='dictionary'").fetchone()[0]
        ai_hits   = conn.execute("SELECT COUNT(*) FROM questions WHERE source='ai'").fetchone()[0]
        unanswered = conn.execute("SELECT COUNT(*) FROM questions WHERE answered=0").fetchone()[0]
        avg_time  = conn.execute("SELECT AVG(think_sec) FROM questions WHERE source='ai'").fetchone()[0]
        top_q     = conn.execute(
            "SELECT message, COUNT(*) as c FROM questions GROUP BY lower(message) ORDER BY c DESC LIMIT 10"
        ).fetchall()
        unanswered_q = conn.execute(
            "SELECT message, ts FROM questions WHERE answered=0 ORDER BY ts DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return {
            "total": total, "answered": answered, "unanswered": unanswered,
            "dict_hits": dict_hits, "ai_hits": ai_hits,
            "avg_ai_time": round(avg_time or 0, 1),
            "top_questions": [{"q": r[0], "count": r[1]} for r in top_q],
            "unanswered_questions": [{"q": r[0], "ts": r[1]} for r in unanswered_q]
        }
    except Exception as e:
        return {"error": str(e)}


# ── REWORD VIA MODEL (sync, runs in thread) ───────────────────────────────────

REWORD_PROMPT = """Rephrase this answer in one sentence. Keep every fact identical. Neutral tone. No excitement, no filler. Output only the rephrased sentence, nothing else.

Answer: {answer}
One sentence rephrasing:"""

def _reword_sync(answer: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":    MODEL_NAME,
                "messages": [{"role": "user", "content": REWORD_PROMPT.format(answer=answer)}],
                "options":  {"temperature": 0.7, "num_predict": 120},
                "stream":   False
            },
            timeout=(10, 60)
        )
        raw = resp.json().get("message", {}).get("content", "").strip()
        for prefix in ["One sentence rephrasing:", "Rephrased:", "Answer:", "Here is", "Here's", "Here:"]:
            if raw.lower().startswith(prefix.lower()):
                raw = raw[len(prefix):].strip().lstrip(":").strip()
        import re
        first = re.split(r"(?<=[.!?])\s+", raw)
        raw = first[0].strip() if first else raw
        return raw if raw else answer
    except Exception:
        return answer


# ── OLLAMA — STREAMING ────────────────────────────────────────────────────────

async def ask_ollama_stream(question: str, context_chunks: list,
                             session_id: str, memory: MemoryStore,
                             system_prompt: str):
    """
    Async generator that streams tokens from Ollama using httpx.
    Yields text tokens as they arrive.
    """
    from config import MAX_CONTEXT_CHARS, MAX_TOKENS, TEMPERATURE
    context_text = "\n\n".join(context_chunks)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]

    user_content = (
        f"Context from documents:\n{context_text}\n\nQuestion: {question}"
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
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        full_reply += token
                        yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectTimeout:
        yield "\n[Error: Cannot connect to Ollama. Run: ollama serve]"
        return
    except httpx.ReadTimeout:
        yield "\n[Error: Ollama timed out. Try a shorter question.]"
        return
    except Exception as e:
        yield f"\n[Error: {e}]"
        return

    # Save to memory
    if full_reply.strip() and "don't have that information" not in full_reply.lower():
        memory.add(session_id, "user",      question)
        memory.add(session_id, "assistant", full_reply.strip())


# ── APP SETUP ─────────────────────────────────────────────────────────────────

rag    = RAGEngine()
memory = MemoryStore()

matcher       = None
system_prompt = ""
WEB_UI        = ""
ADMIN_UI      = ""


def startup():
    global matcher, system_prompt, WEB_UI, ADMIN_UI

    print("\n── Two Steps Ahead Chatbot ─────────────────────────")

    # Init analytics DB
    init_db()
    print("  [ok] Analytics DB ready")

    # Load HTML files
    WEB_UI   = load_text(os.path.join("frontend", "web_ui.html")) or "<h1>web_ui.html not found</h1>"
    ADMIN_UI = load_text(os.path.join("frontend", "admin.html"))  or "<h1>admin.html not found</h1>"

    # Ingest robot data
    robot_text = load_text(DATA_FILE)
    if robot_text:
        n = rag.ingest(robot_text, CHUNK_SIZE_ROBOT)
        print(f"  [ok] {DATA_FILE} — {n} chunks")
    else:
        print(f"  [warn] {DATA_FILE} not found")

    # Ingest game manual
    print(f"  [info] Looking for game manual at: {os.path.abspath(GAME_MANUAL_FILE)}")
    manual_text = load_pdf(GAME_MANUAL_FILE)
    if manual_text:
        n = rag.ingest(manual_text, CHUNK_SIZE_MANUAL)
        print(f"  [ok] {GAME_MANUAL_FILE} — {n} chunks")
    else:
        print(f"  [warn] Game manual not loaded")

    print(f"  [ok] Total RAG chunks: {len(rag.chunks)}")

    # Dictionary + semantic matcher
    dictionary = load_json(DICTIONARY_FILE)
    matcher    = SemanticMatcher(dictionary)

    # Prompts + system prompt
    prompts       = load_json(PROMPTS_FILE)
    system_prompt = build_system_prompt(prompts)
    print(f"  [ok] System prompt built ({len(system_prompt)} chars)")

    # Cache warmup — send dummy request so first real question is fast
    print("  [..] Warming up model...")
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model":    MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "options":  {"num_predict": 1},
                "stream":   False
            },
            timeout=30
        )
        print("  [ok] Model warmed up")
    except Exception:
        print("  [warn] Warmup failed — Ollama may not be running yet")

    print(f"  [ok] Model : {MODEL_NAME}")
    print(f"  [ok] Server: http://localhost:{PORT}")
    print("  Run Ollama : ollama serve")
    print("────────────────────────────────────────────────────\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield

app = FastAPI(lifespan=lifespan)


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(WEB_UI)


@app.get("/health")
def health():
    try:
        requests.get("http://localhost:11434", timeout=3)
        ollama_ok = True
    except Exception:
        ollama_ok = False
    return {
        "status":          "ok",
        "ollama":          "reachable" if ollama_ok else "unreachable",
        "model":           MODEL_NAME,
        "rag_chunks":      len(rag.chunks),
        "active_sessions": memory.count()
    }


@app.post("/chat")
async def chat(req: Request):
    body       = await req.json()
    msg        = body.get("message", "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())

    if not msg:
        return JSONResponse({"error": "message required"}, status_code=400)

    # Check if Ollama is alive
    try:
        requests.get("http://localhost:11434", timeout=2)
        ollama_alive = True
    except Exception:
        ollama_alive = False

    # Offline fallback — dictionary only
    if not ollama_alive:
        instant = matcher.match(msg) if matcher else None
        if instant:
            start      = time.time()
            fake_delay = random.uniform(3, 8)
            await asyncio.sleep(fake_delay)
            elapsed = time.time() - start
            memory.add(session_id, "user",      msg)
            memory.add(session_id, "assistant", instant)
            log_question(msg, "dictionary", True, elapsed)
            return {
                "reply":         instant,
                "source":        "dictionary",
                "chunks_used":   0,
                "session_id":    session_id,
                "think_seconds": round(elapsed, 1)
            }
        log_question(msg, "offline", False, 0)
        return JSONResponse({
            "error": "AI model offline and no dictionary match found. Ask a team member!"
        }, status_code=503)

    # Step 1 — semantic dictionary match
    instant = matcher.match(msg) if matcher else None
    if instant:
        start      = time.time()
        fake_delay = random.uniform(5, 15)
        loop       = asyncio.get_event_loop()

        reworded, _ = await asyncio.gather(
            loop.run_in_executor(None, _reword_sync, instant),
            asyncio.sleep(fake_delay)
        )

        elapsed = time.time() - start
        memory.add(session_id, "user",      msg)
        memory.add(session_id, "assistant", reworded)
        log_question(msg, "dictionary", True, elapsed)
        return {
            "reply":         reworded,
            "source":        "dictionary",
            "chunks_used":   0,
            "session_id":    session_id,
            "think_seconds": round(elapsed, 1)
        }

    # Step 2 — RAG + streaming Ollama
    chunks = rag.retrieve(msg)
    start  = time.time()

    # Collect full streamed reply then return as JSON
    # (frontend doesn't yet support streaming — see /chat/stream for that)
    full_reply = ""
    async for token in ask_ollama_stream(msg, chunks, session_id, memory, system_prompt):
        full_reply += token

    elapsed  = time.time() - start
    answered = "don't have that information" not in full_reply.lower()
    log_question(msg, "ai", answered, round(elapsed, 1))

    if full_reply.startswith("\n[Error:"):
        return JSONResponse({"error": full_reply.strip()}, status_code=500)

    return {
        "reply":         full_reply.strip(),
        "source":        "ai",
        "chunks_used":   len(chunks),
        "session_id":    session_id,
        "think_seconds": round(elapsed, 1)
    }


@app.get("/chat/stream")
async def chat_stream(message: str, session_id: str = None):
    """
    Server-Sent Events endpoint for true streaming responses.
    Frontend connects and receives tokens as they arrive.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    chunks = rag.retrieve(message)

    async def event_generator():
        # Send session_id first
        yield f"data: {json.dumps({'session_id': session_id, 'type': 'meta', 'chunks_used': len(chunks)})}\n\n"

        full = ""
        async for token in ask_ollama_stream(message, chunks, session_id, memory, system_prompt):
            full += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        answered = "don't have that information" not in full.lower()
        log_question(message, "ai_stream", answered, 0)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest")
async def ingest(req: Request):
    body = await req.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    n = rag.ingest(text)
    return {"ingested_chunks": n, "total_chunks": len(rag.chunks)}


@app.delete("/ingest")
async def clear_rag():
    rag.clear()
    return {"message": "RAG knowledge base cleared"}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    memory.clear(session_id)
    return {"message": f"Session {session_id} cleared"}


# ── ADMIN PANEL ───────────────────────────────────────────────────────────────

ADMIN_PASSWORD = "avocado2026"
_admin_tokens: set = set()

import secrets

@app.get("/admin", response_class=HTMLResponse)
def admin_ui():
    return HTMLResponse(ADMIN_UI)

@app.post("/admin/login")
async def admin_login(req: Request):
    body = await req.json()
    if body.get("password") == ADMIN_PASSWORD:
        token = secrets.token_hex(16)
        _admin_tokens.add(token)
        return {"token": token}
    return JSONResponse({"error": "wrong password"}, status_code=401)

def check_admin(req: Request):
    return req.headers.get("X-Admin-Token") in _admin_tokens

@app.get("/admin/stats")
def admin_stats(req: Request):
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        requests.get("http://localhost:11434", timeout=2)
        ollama_ok = "online"
    except:
        ollama_ok = "offline"
    dict_count = len(matcher.dictionary) if matcher else 0
    return {
        "rag_chunks":      len(rag.chunks),
        "active_sessions": memory.count(),
        "dict_entries":    dict_count,
        "ollama":          ollama_ok,
        "model":           MODEL_NAME
    }

@app.get("/admin/analytics")
def admin_analytics(req: Request):
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return get_analytics()

@app.get("/admin/robot-data")
def admin_get_robot(req: Request):
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"content": load_text(DATA_FILE)}

@app.post("/admin/robot-data")
async def admin_save_robot(req: Request):
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await req.json()
    content = body.get("content", "")
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        rag.clear()
        n = rag.ingest(content, CHUNK_SIZE_ROBOT)
        manual = load_pdf(GAME_MANUAL_FILE)
        if manual:
            rag.ingest(manual, CHUNK_SIZE_MANUAL)
        return {"message": f"Saved and reloaded — {n} chunks indexed"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/admin/dictionary")
def admin_get_dict(req: Request):
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"content": load_json(DICTIONARY_FILE)}

@app.post("/admin/dictionary")
async def admin_save_dict(req: Request):
    global matcher
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    body = await req.json()
    content = body.get("content", {})
    try:
        os.makedirs(os.path.dirname(DICTIONARY_FILE), exist_ok=True)
        with open(DICTIONARY_FILE, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        matcher = SemanticMatcher(content)
        return {"message": f"Saved and reloaded — {len(content)} entries"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/admin/clear-sessions")
def admin_clear_sessions(req: Request):
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    memory._store.clear()
    return {"message": "All sessions cleared"}


# ── START ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)