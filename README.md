# Avocado — Two Steps Ahead AI Pit Assistant
**Built by Vedant for FRC Team 10600**

Avocado is an AI pit assistant I built for our team to use at competitions. Visitors can walk up, ask anything about our robot, the game, or the team, and get an answer in 5-30 seconds without any internet. It runs completely on any laptop, and I built it from scratch using FastAPI, Ollama, and a custom RAG pipeline.

---

## Why I Built This

At competitions, visitors and other teams walk up to the pit and ask questions about our bot: how fast is your robot, what drivetrain do you use, how does the intake work? The goal of this chatbot was to allow people to understand how our robot works, but it would also serve as a medium of communication when a team member is busy working on the actual bot.

I also wanted it to be smart enough to handle questions from the game manual so that people can understand what the goals of the game are, and be able to ask follow-up questions, unlike a FAQ page.

---

## What It Does

- Answers questions about our robot **Double Dip**, our team, and the 2026 FRC game
- Uses a **semantic dictionary** so it understands different ways of asking the same question
- Falls back to a **local LLM (Ollama)** for anything not in the dictionary, using our robot data and game manual as context
- **ChromaDB** vector database with sentence-transformer embeddings for accurate retrieval
- **Response caching** — repeated questions return instantly without calling the model
- **Confidence scoring** — skips the model entirely if retrieval confidence is too low
- **Async request queue** — handles multiple users simultaneously without blocking
- **Rate limiting** — prevents spam from slowing the bot down
- **Conversation memory with summarisation** — follow-up questions work naturally across long sessions
- **Session expiry** — stale sessions cleaned up automatically after 30 minutes
- Shows a **"Thought for X seconds"** label on every reply
- Runs **100% offline** — no API keys, no internet required at competition
- Custom UI styled to match our team website [stepsahead10600.web.app](https://stepsahead10600.web.app)
- **Admin panel** at `/admin` for live editing of robot data and dictionary, with an analytics dashboard

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | Ollama (`llama3.2:1b`) |
| Vector DB | ChromaDB (persistent, on-disk) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Fallback RAG | scikit-learn TF-IDF + bigrams + keyword hybrid |
| Chunking | spaCy `en_core_web_sm` with regex fallback |
| Async HTTP | httpx |
| Analytics | SQLite |
| PDF parsing | pypdf |
| Frontend | Vanilla HTML / CSS / JS |

---

## Project Structure

```
chatbot_project/
├── main.py              # FastAPI server — start here
├── robot_core.py        # RAG, ChromaDB, matcher, memory, cache, queue
├── utils.py             # spaCy chunking and text preprocessing
├── config.py            # All settings
├── start.py             # One-click launcher
├── requirements.txt
├── data/                # Not included in repo — create locally
│   ├── robot_data.txt
│   ├── game_manual.pdf
│   ├── dictionary.json
│   ├── prompts.json
│   ├── analytics.db     # Auto-created on first run
│   └── chroma_db/       # Auto-created on first run
└── frontend/
    ├── web_ui.html
    └── admin.html
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/DagaVedant/FRC-Chatbot-Team-10600.git
cd FRC-Chatbot-Team-10600
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Install Ollama and pull the model

Download from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3.2:1b
ollama serve
```

### 4. Pre-cache models before competition

Run these once at home while you have WiFi — after this everything works offline:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ollama pull llama3.2:1b
```

### 5. Run

```bash
python start.py
```

Or directly:

```bash
python main.py
```

Go to `http://localhost:8000`

---

## How It Works

I built a multi-stage pipeline:

```
Question
   │
   ├─ Rate limit check (10 req / 60s per IP)
   │
   ├─ Ollama alive? → No → Dictionary-only offline fallback
   │
   ├─ Semantic dictionary match?
   │     Yes → LLM rewords it + 5-15s fake delay → Return
   │
   ├─ ChromaDB retrieval + confidence score
   │     Low confidence + no chunks → Instant fallback, no model call
   │
   ├─ Response cache hit?
   │     Yes → Instant return
   │
   └─ Request queue → Async Ollama stream → Cache result → Return
```

**Dictionary layer** — I wrote ~60 Q&A pairs for common pit questions. They get embedded with `sentence-transformers` on startup so the bot can match questions even when the wording is different.

**RAG layer** — `robot_data.txt` and the game manual PDF are chunked using spaCy's sentence detector with overlap, then embedded and stored in ChromaDB. Retrieval uses cosine similarity boosted by keyword overlap scoring. Falls back to TF-IDF + bigrams if ChromaDB is unavailable.

**Memory** — each session gets a unique ID. The last 6 messages are kept per session. When history gets long, older exchanges are compressed into an extractive summary so the bot stays coherent across long conversations without blowing up the context window.

**Cache** — an LRU cache stores the last 128 AI responses. Repeated questions return instantly without touching Ollama.

---

## Configuration

Everything lives in `config.py` so I can change settings without touching the logic:

| Setting | Default | What it does |
|---|---|---|
| `MODEL_NAME` | `llama3.2:1b` | Ollama model |
| `TOP_K` | `5` | RAG chunks retrieved per query |
| `MAX_CONTEXT_CHARS` | `1800` | Characters passed to the LLM |
| `MAX_HISTORY` | `6` | Messages kept per session |
| `SUMMARY_THRESHOLD` | `6` | When to compress history into summary |
| `SEMANTIC_THRESHOLD` | `0.55` | Similarity cutoff for dictionary match |
| `CONFIDENCE_THRESHOLD` | `0.05` | RAG confidence below which model is skipped |
| `RESPONSE_CACHE_SIZE` | `128` | Max cached AI responses |
| `SESSION_EXPIRY` | `1800` | Session timeout in seconds (30 min) |
| `RATE_LIMIT_REQUESTS` | `10` | Max requests per window per IP |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |
| `MAX_TOKENS` | `256` | Max response length |
| `TEMPERATURE` | `0.0` | 0 = deterministic, no hallucinations |
| `PORT` | `8000` | Server port |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/chat` | Send message, get reply |
| `GET` | `/chat/stream` | SSE streaming endpoint |
| `GET` | `/health` | System status |
| `POST` | `/ingest` | Add text to RAG at runtime |
| `DELETE` | `/ingest` | Clear RAG + cache |
| `DELETE` | `/session/{id}` | Clear a session |
| `GET` | `/admin` | Admin panel |

**POST /chat example:**

```json
// Request
{ "message": "How fast is your robot?", "session_id": "abc123" }

// Response
{
  "reply": "The robot reaches a top speed of 13.5 feet per second.",
  "source": "dictionary",
  "chunks_used": 0,
  "session_id": "abc123",
  "think_seconds": 8.3,
  "cached": false
}
```

`source` is `"dictionary"`, `"ai"`, `"fallback"`, or `"offline"`.

---

## Admin Panel

Go to `http://localhost:8000/admin`. Password set in `ADMIN_PASSWORD` in `main.py`.

| Tab | What it shows |
|---|---|
| Status | RAG chunks, cache size, sessions, Ollama status |
| Analytics | Question counts, hit rates, avg response time, top questions, unanswered questions |
| Robot Data | Edit `robot_data.txt` live and reload RAG instantly |
| Dictionary | Edit `dictionary.json` live and rebuild embeddings instantly |
| Danger Zone | Clear RAG, cache, or sessions |

---

## The Team

**Two Steps Ahead · FRC Team 10600 · Edison, New Jersey**

| | |
|---|---|
| Email | twostepsaheadrobotics@gmail.com |
| Instagram | [@twostepsahead.robotics](https://instagram.com/twostepsahead.robotics) |
| Website | [stepsahead10600.web.app](https://stepsahead10600.web.app) |

---

## License

MIT — feel free to adapt this for your own FRC team.

> **Note:** `data/` files are not included in this repo. Create your own `robot_data.txt`, `dictionary.json`, `prompts.json`, and add your `game_manual.pdf` before running.
