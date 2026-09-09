from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import settings
from .document_loader import load_and_split_documents


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def rebuild_vector_store() -> tuple[Chroma, int]:
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_and_split_documents()
    if not chunks:
        raise RuntimeError("No supported documents were found in data/policies or data/uploads.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
    )
    return vector_store, len(chunks)


def retrieve(query: str, k: int | None = None) -> list[Document]:
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k or settings.top_k)
