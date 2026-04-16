"""Chunk fixtures knowledge-base files and upsert into Chroma."""

import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import ALLOWED_INGEST_ROOT, CHROMA_DIR, FIXTURES_DIR
from app.vectorstore import (
    KB_CHUNK_OVERLAP,
    KB_CHUNK_SIZE,
    discover_knowledge_files,
    get_vectorstore,
)
from app.scope_index import invalidate_scope_index


def _infer_section_title(chunk: str) -> str:
    for raw_line in chunk.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            return line[4:].strip()
        if line.startswith("## "):
            return line[3:].strip()
        if line.startswith("# "):
            return line[2:].strip()
    preview = " ".join(chunk.strip().split())
    if not preview:
        return "General"
    return (preview[:80] + "…") if len(preview) > 80 else preview


def resolve_handbook_path(relative_or_absolute: str) -> Path:
    raw = Path(relative_or_absolute)
    resolved = (raw if raw.is_absolute() else (ALLOWED_INGEST_ROOT / raw)).resolve()
    if not str(resolved).startswith(str(ALLOWED_INGEST_ROOT.resolve())):
        raise ValueError("Handbook path escapes allowed root")
    if not resolved.is_file():
        raise FileNotFoundError(f"Handbook not found: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in (".md", ".txt", ".markdown"):
        raise ValueError("Only .md / .txt handbooks are supported")
    return resolved


def ingest_handbook_file(_file_path: Path, *, replace: bool = True) -> int:
    if replace and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # Knowledge-base mode: always scan entire fixtures/ folder.
    source_files = discover_knowledge_files(FIXTURES_DIR)
    if not source_files:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=KB_CHUNK_SIZE,
        chunk_overlap=KB_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    docs: list[Document] = []
    for f in source_files:
        if f.suffix.lower() == ".pdf":
            reader = PdfReader(str(f))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        else:
            text = f.read_text(encoding="utf-8")
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        docs.extend(
            Document(
                page_content=c,
                metadata={
                    "source": str(f),
                    "source_name": f.name,
                    "chunk_index": i,
                    "section_title": _infer_section_title(c),
                },
            )
            for i, c in enumerate(chunks)
        )

    if not docs:
        return 0
    vs = get_vectorstore()
    vs.add_documents(docs)
    invalidate_scope_index()
    return len(docs)
