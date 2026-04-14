"""Paths and environment for rag-api."""

import os
from pathlib import Path

from dotenv import load_dotenv

# rag-api/app/config.py -> parents[1] = rag-api, parents[2] = monorepo root
RAG_API_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

# Single monorepo env file at repository root
load_dotenv(REPO_ROOT / ".env")


def _path_from_env(
    key: str,
    default: Path,
    *,
    relative_to: Path,
) -> Path:
    raw = os.getenv(key)
    if not raw:
        return default
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (relative_to / p).resolve()


FIXTURES_DIR = REPO_ROOT / "fixtures"
DEFAULT_HANDBOOK_PATH = _path_from_env(
    "DEFAULT_HANDBOOK_PATH",
    FIXTURES_DIR / "handbook.md",
    relative_to=REPO_ROOT,
)
EMPLOYEES_JSON_PATH = _path_from_env(
    "EMPLOYEES_JSON_PATH",
    FIXTURES_DIR / "employees.json",
    relative_to=REPO_ROOT,
)

CHROMA_DIR = _path_from_env(
    "CHROMA_PERSIST_DIR",
    RAG_API_ROOT / "data" / "chroma",
    relative_to=RAG_API_ROOT,
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "handbook_chunks")

# Only paths under this directory may be ingested via handbookPath (relative to repo root)
ALLOWED_INGEST_ROOT = _path_from_env(
    "ALLOWED_INGEST_ROOT",
    REPO_ROOT,
    relative_to=REPO_ROOT,
)

# LangChain / OpenAI (OPENAI_API_KEY is read by langchain-openai from the environment)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def cors_origins() -> list[str]:
    raw = os.getenv("RAG_CORS_ORIGINS")
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
