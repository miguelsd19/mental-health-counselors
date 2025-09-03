import os
from dotenv import load_dotenv


load_dotenv()


PORT = int(os.getenv("PORT", 8000))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag.db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_DIR = os.getenv("FAISS_DIR", "./data/faiss")
TOP_K = int(os.getenv("TOP_K", 5))
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.3))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT_LANG = os.getenv("SYSTEM_PROMPT_LANG", "es")