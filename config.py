"""Central configuration loaded from .env"""
import os
from dotenv import load_dotenv

load_dotenv()

# Grok / xAI
# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# GitHub
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "langchain-ai/langchain")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "langchain_codebase")

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIM = 384  # bge-large output dimension

# Chunking
CODE_CHUNK_SIZE = 800
CODE_CHUNK_OVERLAP = 100
DOC_CHUNK_SIZE = 500
DOC_CHUNK_OVERLAP = 50

# Retrieval
TOP_K_DENSE = 6
TOP_K_SPARSE = 6
TOP_K_FINAL = 3
RRF_K = 60  # reciprocal rank fusion constant

# Local repo clone path
REPO_CLONE_DIR = "./data/repo"
