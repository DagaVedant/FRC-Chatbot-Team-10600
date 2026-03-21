# ── config.py ─────────────────────────────────────────────────────────────────

# Ollama
OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL_NAME  = "llama3.2:1b"

# RAG
TOP_K             = 5
MAX_CONTEXT_CHARS = 1800
CHUNK_SIZE_ROBOT  = 400
CHUNK_SIZE_MANUAL = 700

# Model generation
MAX_TOKENS   = 256
TEMPERATURE  = 0.0

# Conversation memory
MAX_HISTORY  = 6        # raw messages kept before summarisation kicks in
SUMMARY_THRESHOLD = 6   # summarise when history hits this length

# Semantic matching
SEMANTIC_THRESHOLD = 0.55

# Response cache
RESPONSE_CACHE_SIZE = 128   # number of AI responses to cache

# Confidence — if best RAG score is below this AND no dict match, skip model
CONFIDENCE_THRESHOLD = 0.05

# Session expiry (seconds)
SESSION_EXPIRY = 1800   # 30 minutes

# Rate limiting
RATE_LIMIT_REQUESTS = 10   # max requests
RATE_LIMIT_WINDOW   = 60   # per N seconds per IP

# Server
PORT = 8000

# File paths
DATA_FILE        = "data/robot_data.txt"
GAME_MANUAL_FILE = "data/game_manual.pdf"
DICTIONARY_FILE  = "data/dictionary.json"
PROMPTS_FILE     = "data/prompts.json"
CHROMA_PATH      = "data/chroma_db"