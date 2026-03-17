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
- **Remembers the conversation** so follow-up questions work naturally
- Shows a **"Thought for X seconds"** label on every reply
- Runs **100% offline** — no API keys, no internet required
- Custom UI styled to match our team website [stepsahead10600.web.app](https://stepsahead10600.web.app)

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | Ollama (`llama3.2:1b`) |
| RAG | TF-IDF + bigrams + keyword hybrid (scikit-learn) |
| Semantic matching | sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF parsing | pypdf |
| Frontend | Vanilla HTML/CSS/JS |

---

## Project Structure

```
chatbot_project/
├── main.py              # FastAPI server — start here
├── robot_core.py        # RAG engine, semantic matcher, memory, Ollama
├── utils.py             # Text cleaning and chunking
├── config.py            # All settings
├── requirements.txt
├── data/
│   ├── robot_data.txt   # Our robot and team info
│   ├── game_manual.pdf  # FRC 2026 game manual
│   ├── dictionary.json  # Pre-written Q&A pairs
│   └── prompts.json     # Team/robot/game facts + system prompt
└── frontend/
    └── web_ui.html      # Chat UI
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/DagaVedant/FRC-Chatbot-10600.git
cd FRC-Chatbot-10600
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama and pull the model

Download from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3.2:1b
ollama serve
```

### 4. Run

```bash
python main.py
```

Go to `http://localhost:8000`

---

## How It Works

I built a three-layer pipeline:

```
Question
   │
   ▼
Semantic dictionary match?
   │ yes ──▶  LLM rephrases it  ──▶  Returns answer in UI
   │ no
   ▼
RAG retrieval from robot_data.txt + game_manual.pdf
   │
   ▼
Ollama LLM with context + team knowledge + conversation history
   │
   ▼
Answer + think time shown in UI
```

**Dictionary layer** — I wrote ~60 Q&A pairs for common pit questions. They get embedded with `sentence-transformers` on startup so the bot can match questions even when the wording is different. When it hits a dictionary match, it sends the answer to the LLM to rephrase it slightly so it sounds different each time, then waits a random 5–15 seconds before responding so it looks like it's thinking.

**RAG layer** — `robot_data.txt` and the game manual PDF get chunked with sentence-level overlap and indexed with TF-IDF + bigrams. Retrieval uses a hybrid score combining TF-IDF cosine similarity and keyword overlap so technical terms like "MK4i" or "AprilTag" don't get missed.

**Memory** — each session gets a unique ID and I store the last 6 messages (3 exchanges) so the bot can handle follow-ups like "how fast is it?" after asking about the drivetrain.

---

## Configuration

Everything lives in `config.py` so I can change settings without touching the logic:

| Setting | Default | What it does |
|---|---|---|
| `MODEL_NAME` | `llama3.2:1b` | Ollama model |
| `TOP_K` | `5` | RAG chunks retrieved per query |
| `MAX_CONTEXT_CHARS` | `1800` | Characters passed to the LLM |
| `MAX_HISTORY` | `6` | Messages kept per session |
| `SEMANTIC_THRESHOLD` | `0.55` | Similarity cutoff for dictionary match |
| `MAX_TOKENS` | `256` | Max response length |
| `TEMPERATURE` | `0.0` | 0 = deterministic, no hallucinations |
| `PORT` | `8000` | Server port |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/chat` | Send message, get reply |
| `GET` | `/health` | Ollama status + chunk count |
| `POST` | `/ingest` | Add text to RAG at runtime |
| `DELETE` | `/ingest` | Clear RAG knowledge base |
| `DELETE` | `/session/{id}` | Clear a session |

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
  "think_seconds": 8.3
}
```

`source` is `"dictionary"` or `"ai"`.

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

Note: data/ files are not included in this repo. Create your own robot_data.txt, dictionary.json, and prompts.json before running
