# ── config.py ────────────────────────────────────────────────────────────────
# Change settings here without touching any other file.

# Ollama
OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL_NAME  = "llama3.2:1b"   # fast & smart; swap to "phi3" for more accuracy

# RAG
TOP_K             = 5      # how many chunks to retrieve per query
MAX_CONTEXT_CHARS = 1800   # characters sent to the model
CHUNK_SIZE_ROBOT  = 400    # chunk size for robot_data.txt
CHUNK_SIZE_MANUAL = 700    # chunk size for game_manual.pdf

# Model generation
MAX_TOKENS   = 256
TEMPERATURE  = 0.0         # 0 = deterministic, no hallucinations

# Conversation memory
MAX_HISTORY  = 6           # messages kept per session (3 exchanges)

# Semantic matching
SEMANTIC_THRESHOLD = 0.55  # cosine similarity needed to use dictionary answer
                           # lower = more lenient, higher = stricter

# Server
PORT = 8000

# File paths
DATA_FILE        = "data/robot_data.txt"
GAME_MANUAL_FILE = "data/game_manual.pdf"
DICTIONARY_FILE  = "data/dictionary.json"
PROMPTS_FILE     = "data/prompts.json"