import os
from dotenv import load_dotenv

load_dotenv()

# Endee
ENDEE_BASE_URL   = os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
ENDEE_INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "rag_docs")

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# Chunking
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80

# Retrieval
TOP_K = 5

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
