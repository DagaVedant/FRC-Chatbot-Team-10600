# ── main.py ───────────────────────────────────────────────────────────────────
import os
import uuid
import json
import time
import random
import asyncio
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pypdf import PdfReader

# Always run relative to this file's location so paths work from anywhere
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    PORT, DATA_FILE, GAME_MANUAL_FILE, DICTIONARY_FILE,
    PROMPTS_FILE, CHUNK_SIZE_ROBOT, CHUNK_SIZE_MANUAL, MODEL_NAME,
    OLLAMA_URL
)
from robot_core import RAGEngine, SemanticMatcher, MemoryStore, ask_ollama, build_system_prompt


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
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        result = "\n\n".join(pages)
        print(f"  [ok] {path} — {len(reader.pages)} pages, {len(result)} chars extracted")
        return result
    except Exception as e:
        print(f"  [error] reading PDF: {e}")
        return ""

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found — using empty dict")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── APP ───────────────────────────────────────────────────────────────────────

rag    = RAGEngine()
memory = MemoryStore()

matcher       = None
system_prompt = ""
WEB_UI        = ""


def startup():
    global matcher, system_prompt, WEB_UI

    print("\n── Two Steps Ahead Chatbot ─────────────────────────")

    # Load HTML
    ui_path = os.path.join("frontend", "web_ui.html")
    WEB_UI  = load_text(ui_path) or "<h1>UI not found — place index.html in frontend/</h1>"

    # Ingest robot data
    robot_text = load_text(DATA_FILE)
    if robot_text:
        n = rag.ingest(robot_text, CHUNK_SIZE_ROBOT)
        print(f"  [ok] {DATA_FILE} — {n} chunks")
    else:
        print(f"  [warn] No robot_data.txt found — AI will rely on prompts.json only")

    # Ingest game manual PDF — verbose logging to help debug
    print(f"  [info] Looking for game manual at: {os.path.abspath(GAME_MANUAL_FILE)}")
    manual_text = load_pdf(GAME_MANUAL_FILE)
    if manual_text:
        n = rag.ingest(manual_text, CHUNK_SIZE_MANUAL)
        print(f"  [ok] {GAME_MANUAL_FILE} — {n} chunks ingested into RAG")
    else:
        print(f"  [warn] Game manual not loaded — game questions will use prompts.json fallback")

    print(f"  [ok] Total RAG chunks: {len(rag.chunks)}")

    # Load dictionary + semantic matcher
    dictionary = load_json(DICTIONARY_FILE)
    matcher    = SemanticMatcher(dictionary)

    # Load prompts + build enriched system prompt
    prompts       = load_json(PROMPTS_FILE)
    system_prompt = build_system_prompt(prompts)

    print(f"  [ok] System prompt built ({len(system_prompt)} chars)")
    print(f"  [ok] Model : {MODEL_NAME}")
    print(f"  [ok] Server: http://localhost:{PORT}")
    print("  Run Ollama : ollama serve")
    print("────────────────────────────────────────────────────\n")


# ── ROUTES ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield

app = FastAPI(lifespan=lifespan)


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


REWORD_PROMPT = """Rephrase this answer in one sentence. Keep every fact identical. Neutral tone. No excitement, no filler, no extra context. Output only the rephrased sentence, nothing else.

Answer: {answer}
One sentence rephrasing:"""

def _reword_sync(answer: str) -> str:
    """Blocking call to reword a dictionary answer via Ollama."""
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
        # Take only the first sentence if model still rambles
        import re
        first = re.split(r"(?<=[.!?])\s+", raw)
        raw = first[0].strip() if first else raw
        return raw if raw else answer
    except Exception:
        return answer


@app.post("/chat")
async def chat(req: Request):
    body       = await req.json()
    msg        = body.get("message", "").strip()
    session_id = body.get("session_id") or str(uuid.uuid4())

    if not msg:
        return JSONResponse({"error": "message required"}, status_code=400)

    # Step 1 — check if Ollama is reachable (offline fallback)
    try:
        requests.get("http://localhost:11434", timeout=2)
        ollama_alive = True
    except Exception:
        ollama_alive = False

    # If Ollama is down, force dictionary-only mode
    if not ollama_alive:
        instant = matcher.match(msg) if matcher else None
        if instant:
            start = time.time()
            fake_delay = random.uniform(3, 8)
            await asyncio.sleep(fake_delay)
            elapsed = time.time() - start
            memory.add(session_id, "user", msg)
            memory.add(session_id, "assistant", instant)
            return {
                "reply": instant,
                "source": "dictionary",
                "chunks_used": 0,
                "session_id": session_id,
                "think_seconds": round(elapsed, 1),
                "offline_mode": True
            }
        return JSONResponse({
            "error": "AI model is offline and no dictionary match found. Ask a team member directly!"
        }, status_code=503)

    # Step 1 — semantic dictionary match
    instant = matcher.match(msg) if matcher else None
    if instant:
        start      = time.time()
        fake_delay = random.uniform(5, 15)
        loop       = asyncio.get_event_loop()

        # Run model reword (blocking) + fake delay concurrently
        reworded, _ = await asyncio.gather(
            loop.run_in_executor(None, _reword_sync, instant),
            asyncio.sleep(fake_delay)
        )

        elapsed = time.time() - start
        memory.add(session_id, "user",      msg)
        memory.add(session_id, "assistant", reworded)
        return {
            "reply":         reworded,
            "source":        "dictionary",
            "chunks_used":   0,
            "session_id":    session_id,
            "think_seconds": round(elapsed, 1)
        }

    # Step 2 — RAG retrieval + Ollama
    start  = time.time()
    chunks = rag.retrieve(msg)
    reply, err = ask_ollama(msg, chunks, session_id, memory, system_prompt)
    elapsed = time.time() - start

    if err:
        return JSONResponse({"error": err}, status_code=500)

    return {
        "reply":         reply,
        "source":        "ai",
        "chunks_used":   len(chunks),
        "session_id":    session_id,
        "think_seconds": round(elapsed, 1)
    }


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

ADMIN_PASSWORD = "avocado2026"   # change this before comp

ADMIN_UI = open(os.path.join("frontend", "admin.html"), "r", encoding="utf-8").read()

import secrets
_admin_tokens: set = set()

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
        "rag_chunks": len(rag.chunks),
        "active_sessions": memory.count(),
        "dict_entries": dict_count,
        "ollama": ollama_ok,
        "model": MODEL_NAME
    }

@app.get("/admin/robot-data")
def admin_get_robot(req: Request):
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    content = load_text(DATA_FILE)
    return {"content": content}

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
        return {"message": f"Saved and reloaded — {n} new chunks indexed"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/admin/dictionary")
def admin_get_dict(req: Request):
    if not check_admin(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    content = load_json(DICTIONARY_FILE)
    return {"content": content}

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