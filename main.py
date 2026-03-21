# ── main.py ───────────────────────────────────────────────────────────────────
import os
import uuid
import json
import time
import random
import asyncio
import sqlite3
import secrets
import requests
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pypdf import PdfReader

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    PORT, DATA_FILE, GAME_MANUAL_FILE, DICTIONARY_FILE,
    PROMPTS_FILE, CHUNK_SIZE_ROBOT, CHUNK_SIZE_MANUAL,
    MODEL_NAME, OLLAMA_URL, CONFIDENCE_THRESHOLD
)
from robot_core import (
    RAGEngine, SemanticMatcher, MemoryStore, ResponseCache,
    RateLimiter, RequestQueue, build_system_prompt, stream_ollama
)


# ── FILE LOADERS ──────────────────────────────────────────────────────────────

def load_text(path):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return ""
    try:
        reader = PdfReader(path)
        pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
        result = "\n\n".join(pages)
        print(f"  [ok] {path} — {len(reader.pages)} pages")
        return result
    except Exception as e:
        print(f"  [error] PDF: {e}")
        return ""

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── ANALYTICS ─────────────────────────────────────────────────────────────────

DB_PATH = "data/analytics.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL, message TEXT NOT NULL,
        source TEXT NOT NULL, answered INTEGER NOT NULL DEFAULT 1,
        think_sec REAL, cached INTEGER NOT NULL DEFAULT 0
    )""")
    # Migrate existing DB — add cached column if it doesn't exist yet
    try:
        conn.execute("ALTER TABLE questions ADD COLUMN cached INTEGER NOT NULL DEFAULT 0")
        conn.commit()
        print("  [ok] Migrated analytics DB — added cached column")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()

def log_question(message, source, answered, think_sec, cached=False):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO questions (ts,message,source,answered,think_sec,cached) VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(), message, source, int(answered), think_sec, int(cached))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  [warn] analytics: {e}")

def get_analytics():
    try:
        conn = sqlite3.connect(DB_PATH)
        total      = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        answered   = conn.execute("SELECT COUNT(*) FROM questions WHERE answered=1").fetchone()[0]
        dict_hits  = conn.execute("SELECT COUNT(*) FROM questions WHERE source='dictionary'").fetchone()[0]
        ai_hits    = conn.execute("SELECT COUNT(*) FROM questions WHERE source='ai'").fetchone()[0]
        cached_hits= conn.execute("SELECT COUNT(*) FROM questions WHERE cached=1").fetchone()[0]
        unanswered = conn.execute("SELECT COUNT(*) FROM questions WHERE answered=0").fetchone()[0]
        avg_time   = conn.execute("SELECT AVG(think_sec) FROM questions WHERE source='ai' AND cached=0").fetchone()[0]
        top_q      = conn.execute(
            "SELECT message, COUNT(*) c FROM questions GROUP BY lower(message) ORDER BY c DESC LIMIT 10"
        ).fetchall()
        unanswered_q = conn.execute(
            "SELECT message, ts FROM questions WHERE answered=0 ORDER BY ts DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return {
            "total": total, "answered": answered, "unanswered": unanswered,
            "dict_hits": dict_hits, "ai_hits": ai_hits, "cached_hits": cached_hits,
            "avg_ai_time": round(avg_time or 0, 1),
            "top_questions":       [{"q": r[0], "count": r[1]} for r in top_q],
            "unanswered_questions":[{"q": r[0], "ts":    r[1]} for r in unanswered_q]
        }
    except Exception as e:
        return {"error": str(e)}


# ── REWORD ────────────────────────────────────────────────────────────────────

REWORD_PROMPT = """Rephrase this answer in one sentence. Keep every fact identical. Neutral tone. No excitement, no filler. Output only the rephrased sentence, nothing else.

Answer: {answer}
One sentence rephrasing:"""

def _reword_sync(answer):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":    MODEL_NAME,
                "messages": [{"role":"user","content":REWORD_PROMPT.format(answer=answer)}],
                "options":  {"temperature":0.7,"num_predict":120},
                "stream":   False
            },
            timeout=(10,60)
        )
        import re
        raw = resp.json().get("message",{}).get("content","").strip()
        for prefix in ["One sentence rephrasing:","Rephrased:","Answer:","Here is","Here's","Here:"]:
            if raw.lower().startswith(prefix.lower()):
                raw = raw[len(prefix):].strip().lstrip(":").strip()
        first = re.split(r"(?<=[.!?])\s+", raw)
        raw = first[0].strip() if first else raw
        return raw if raw else answer
    except Exception:
        return answer


# ── APP GLOBALS ───────────────────────────────────────────────────────────────

rag          = RAGEngine()
memory       = MemoryStore()
cache        = ResponseCache()
rate_limiter = RateLimiter()
req_queue    = None   # initialised in lifespan (needs running event loop)

matcher       = None
system_prompt = ""
WEB_UI        = ""
ADMIN_UI      = ""


def startup():
    global matcher, system_prompt, WEB_UI, ADMIN_UI, req_queue

    print("\n── Two Steps Ahead Chatbot ─────────────────────────")

    init_db()
    print("  [ok] Analytics DB ready")

    WEB_UI   = load_text(os.path.join("frontend","web_ui.html")) or "<h1>web_ui.html not found</h1>"
    ADMIN_UI = load_text(os.path.join("frontend","admin.html"))  or "<h1>admin.html not found</h1>"

    robot_text = load_text(DATA_FILE)
    if robot_text:
        n = rag.ingest(robot_text, CHUNK_SIZE_ROBOT)
        print(f"  [ok] {DATA_FILE} — {n} chunks")

    manual_text = load_pdf(GAME_MANUAL_FILE)
    if manual_text:
        n = rag.ingest(manual_text, CHUNK_SIZE_MANUAL)
        print(f"  [ok] {GAME_MANUAL_FILE} — {n} chunks")

    print(f"  [ok] Total RAG chunks : {len(rag.chunks)}")

    dictionary    = load_json(DICTIONARY_FILE)
    matcher       = SemanticMatcher(dictionary)

    prompts       = load_json(PROMPTS_FILE)
    system_prompt = build_system_prompt(prompts)
    print(f"  [ok] System prompt    : {len(system_prompt)} chars")

    # Cache warmup
    print("  [..] Warming up model...")
    try:
        requests.post(
            OLLAMA_URL,
            json={"model":MODEL_NAME,"messages":[{"role":"user","content":"hi"}],
                  "options":{"num_predict":1},"stream":False},
            timeout=30
        )
        print("  [ok] Model warmed up")
    except Exception:
        print("  [warn] Warmup skipped — Ollama may not be running")

    print(f"  [ok] Model  : {MODEL_NAME}")
    print(f"  [ok] Server : http://localhost:{PORT}")
    print("  Ollama      : ollama serve")
    print("────────────────────────────────────────────────────\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global req_queue
    req_queue = RequestQueue(max_concurrent=3)
    startup()
    yield

app = FastAPI(lifespan=lifespan)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_ip(req: Request) -> str:
    forwarded = req.headers.get("X-Forwarded-For")
    return forwarded.split(",")[0] if forwarded else (req.client.host if req.client else "unknown")

def ollama_alive() -> bool:
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(WEB_UI)


@app.get("/health")
def health():
    return {
        "ollama":          "reachable" if ollama_alive() else "unreachable",
        "model":           MODEL_NAME,
        "rag_chunks":      len(rag.chunks),
        "cache_size":      cache.size(),
        "active_sessions": memory.count()
    }


@app.post("/chat")
async def chat(req: Request):
    # Rate limit
    ip = get_ip(req)
    if not rate_limiter.is_allowed(ip):
        return JSONResponse({"error":"Too many requests. Please wait a moment."}, status_code=429)

    body       = await req.json()
    msg        = body.get("message","").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())

    if not msg:
        return JSONResponse({"error":"message required"}, status_code=400)

    # Offline fallback
    if not ollama_alive():
        instant = matcher.match(msg) if matcher else None
        if instant:
            start = time.time()
            await asyncio.sleep(random.uniform(3,8))
            elapsed = time.time() - start
            memory.add(session_id,"user",msg)
            memory.add(session_id,"assistant",instant)
            log_question(msg,"dictionary",True,elapsed)
            return {"reply":instant,"source":"dictionary","chunks_used":0,
                    "session_id":session_id,"think_seconds":round(elapsed,1)}
        log_question(msg,"offline",False,0)
        return JSONResponse({"error":"AI offline and no dictionary match found."}, status_code=503)

    # Step 1 — semantic dictionary match
    instant = matcher.match(msg) if matcher else None
    if instant:
        start      = time.time()
        fake_delay = random.uniform(5,15)
        loop       = asyncio.get_event_loop()
        reworded, _ = await asyncio.gather(
            loop.run_in_executor(None, _reword_sync, instant),
            asyncio.sleep(fake_delay)
        )
        elapsed = time.time() - start
        memory.add(session_id,"user",msg)
        memory.add(session_id,"assistant",reworded)
        log_question(msg,"dictionary",True,elapsed)
        return {"reply":reworded,"source":"dictionary","chunks_used":0,
                "session_id":session_id,"think_seconds":round(elapsed,1)}

    # Step 2 — RAG retrieval with confidence check
    chunks, confidence = rag.retrieve(msg)

    # Low confidence + no chunks — skip model call entirely
    if confidence < CONFIDENCE_THRESHOLD and not chunks:
        fallback = "Great question. I don't have that information. Ask one of our team members at the pit."
        log_question(msg,"low_confidence",False,0)
        return {"reply":fallback,"source":"fallback","chunks_used":0,
                "session_id":session_id,"think_seconds":0.0}

    # Step 3 — Check response cache
    cached = cache.get(msg, chunks)
    if cached:
        log_question(msg,"ai",True,0,cached=True)
        return {"reply":cached,"source":"ai","chunks_used":len(chunks),
                "session_id":session_id,"think_seconds":0.1,"cached":True}

    # Step 4 — Stream from Ollama (queued so concurrent requests don't block)
    start      = time.time()
    full_reply = ""

    async with req_queue:
        async for token in stream_ollama(msg, chunks, session_id, memory, system_prompt):
            full_reply += token

    elapsed  = time.time() - start
    answered = "don't have that information" not in full_reply.lower()

    if full_reply.startswith("\n[Error:"):
        return JSONResponse({"error":full_reply.strip()}, status_code=500)

    reply = full_reply.strip()

    # Cache successful answers
    if answered:
        cache.set(msg, chunks, reply)

    log_question(msg,"ai",answered,round(elapsed,1))
    return {"reply":reply,"source":"ai","chunks_used":len(chunks),
            "session_id":session_id,"think_seconds":round(elapsed,1)}


@app.get("/chat/stream")
async def chat_stream(message: str, session_id: str = None):
    """SSE endpoint — tokens arrive in real time."""
    if not session_id:
        session_id = str(uuid.uuid4())
    chunks, confidence = rag.retrieve(message)

    async def event_gen():
        yield f"data: {json.dumps({'type':'meta','session_id':session_id,'chunks_used':len(chunks)})}\n\n"
        full = ""
        async with req_queue:
            async for token in stream_ollama(message,chunks,session_id,memory,system_prompt):
                full += token
                yield f"data: {json.dumps({'type':'token','content':token})}\n\n"
        answered = "don't have that information" not in full.lower()
        log_question(message,"ai_stream",answered,0)
        if answered:
            cache.set(message,chunks,full.strip())
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/ingest")
async def ingest(req: Request):
    body = await req.json()
    text = body.get("text","").strip()
    if not text:
        return JSONResponse({"error":"text required"}, status_code=400)
    n = rag.ingest(text)
    cache.clear()
    return {"ingested_chunks":n,"total_chunks":len(rag.chunks)}


@app.delete("/ingest")
async def clear_rag():
    rag.clear()
    cache.clear()
    return {"message":"RAG and cache cleared"}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    memory.clear(session_id)
    return {"message":f"Session {session_id} cleared"}


# ── ADMIN ─────────────────────────────────────────────────────────────────────

ADMIN_PASSWORD = "avocado2026"
_admin_tokens: set = set()

@app.get("/admin", response_class=HTMLResponse)
def admin_ui_route():
    return HTMLResponse(ADMIN_UI)

@app.post("/admin/login")
async def admin_login(req: Request):
    body = await req.json()
    if body.get("password") == ADMIN_PASSWORD:
        token = secrets.token_hex(16)
        _admin_tokens.add(token)
        return {"token": token}
    return JSONResponse({"error":"wrong password"}, status_code=401)

def check_admin(req: Request):
    return req.headers.get("X-Admin-Token") in _admin_tokens

@app.get("/admin/stats")
def admin_stats(req: Request):
    if not check_admin(req):
        return JSONResponse({"error":"unauthorized"}, status_code=401)
    return {
        "rag_chunks":      len(rag.chunks),
        "active_sessions": memory.count(),
        "cache_size":      cache.size(),
        "dict_entries":    len(matcher.dictionary) if matcher else 0,
        "ollama":          "online" if ollama_alive() else "offline",
        "model":           MODEL_NAME
    }

@app.get("/admin/analytics")
def admin_analytics(req: Request):
    if not check_admin(req):
        return JSONResponse({"error":"unauthorized"}, status_code=401)
    return get_analytics()

@app.get("/admin/robot-data")
def admin_get_robot(req: Request):
    if not check_admin(req):
        return JSONResponse({"error":"unauthorized"}, status_code=401)
    return {"content": load_text(DATA_FILE)}

@app.post("/admin/robot-data")
async def admin_save_robot(req: Request):
    if not check_admin(req):
        return JSONResponse({"error":"unauthorized"}, status_code=401)
    body = await req.json()
    content = body.get("content","")
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE,"w",encoding="utf-8") as f:
            f.write(content)
        rag.clear(); cache.clear()
        n = rag.ingest(content, CHUNK_SIZE_ROBOT)
        manual = load_pdf(GAME_MANUAL_FILE)
        if manual:
            rag.ingest(manual, CHUNK_SIZE_MANUAL)
        return {"message":f"Saved and reloaded — {n} chunks"}
    except Exception as e:
        return JSONResponse({"error":str(e)}, status_code=500)

@app.get("/admin/dictionary")
def admin_get_dict(req: Request):
    if not check_admin(req):
        return JSONResponse({"error":"unauthorized"}, status_code=401)
    return {"content": load_json(DICTIONARY_FILE)}

@app.post("/admin/dictionary")
async def admin_save_dict(req: Request):
    global matcher
    if not check_admin(req):
        return JSONResponse({"error":"unauthorized"}, status_code=401)
    body = await req.json()
    content = body.get("content",{})
    try:
        os.makedirs(os.path.dirname(DICTIONARY_FILE), exist_ok=True)
        with open(DICTIONARY_FILE,"w",encoding="utf-8") as f:
            json.dump(content,f,indent=2)
        matcher = SemanticMatcher(content)
        return {"message":f"Saved — {len(content)} entries"}
    except Exception as e:
        return JSONResponse({"error":str(e)}, status_code=500)

@app.post("/admin/clear-sessions")
def admin_clear_sessions(req: Request):
    if not check_admin(req):
        return JSONResponse({"error":"unauthorized"}, status_code=401)
    memory._store.clear()
    memory._summary.clear()
    return {"message":"All sessions cleared"}

@app.post("/admin/clear-cache")
def admin_clear_cache(req: Request):
    if not check_admin(req):
        return JSONResponse({"error":"unauthorized"}, status_code=401)
    cache.clear()
    return {"message":"Response cache cleared"}


# ── START ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)