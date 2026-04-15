"""Chroma vector store + OpenAI embeddings + KB file discovery."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import CHROMA_COLLECTION, CHROMA_DIR, FIXTURES_DIR, OPENAI_EMBEDDING_MODEL

KB_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
KB_CHUNK_SIZE = 400
KB_CHUNK_OVERLAP = 50
_EMBEDDINGS_CACHE: OpenAIEmbeddings | None = None
_VECTORSTORE_CACHE: dict[str, Chroma] = {}


def discover_knowledge_files(fixtures_dir: Path | None = None) -> list[Path]:
    """
    Scan the entire fixtures folder and return supported knowledge files.

    Files are sorted for deterministic ingest ordering.
    """
    root = fixtures_dir or FIXTURES_DIR
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in KB_SUFFIXES:
            out.append(p.resolve())
    out.sort(key=lambda x: str(x).lower())
    return out


def get_embeddings() -> OpenAIEmbeddings:
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        _EMBEDDINGS_CACHE = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    return _EMBEDDINGS_CACHE


def get_vectorstore(persist_dir: Path | None = None) -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    d = persist_dir or CHROMA_DIR
    cache_key = str(d.resolve())
    cached = _VECTORSTORE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    instance = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=cache_key,
    )
    _VECTORSTORE_CACHE[cache_key] = instance
    return instance


def clear_collection(persist_dir: Path | None = None) -> None:
    """Remove persisted DB directory (for tests)."""
    d = persist_dir or CHROMA_DIR
    cache_key = str(d.resolve())
    _VECTORSTORE_CACHE.pop(cache_key, None)
    if d.exists():
        import shutil

        shutil.rmtree(d, ignore_errors=True)
