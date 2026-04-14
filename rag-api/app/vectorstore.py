"""Chroma vector store + OpenAI embeddings + KB file discovery."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import CHROMA_COLLECTION, CHROMA_DIR, FIXTURES_DIR, OPENAI_EMBEDDING_MODEL

KB_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
KB_CHUNK_SIZE = 400
KB_CHUNK_OVERLAP = 50


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
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def get_vectorstore(persist_dir: Path | None = None) -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    d = persist_dir or CHROMA_DIR
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(d),
    )


def clear_collection(persist_dir: Path | None = None) -> None:
    """Remove persisted DB directory (for tests)."""
    d = persist_dir or CHROMA_DIR
    if d.exists():
        import shutil

        shutil.rmtree(d, ignore_errors=True)
