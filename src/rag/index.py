"""Построение индекса в ChromaDB."""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "tire_shop_demo"


def _load_markdown_documents(data_dir: Path) -> list[Document]:
    """Читает только *.md из data/ (без рекурсии — HTML-заглушки не попадут)."""
    files = sorted(data_dir.glob("*.md"))
    return [
        Document(
            text=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        )
        for path in files
    ]


def build_index(data_dir: str | Path = "data", reset: bool = True) -> VectorStoreIndex:
    """Строит VectorStoreIndex по Markdown-файлам из data/.

    При `reset=True` коллекция ChromaDB предварительно очищается,
    чтобы при пересборке не оставалось «стейл»-записей от старых данных.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass  # коллекции ещё нет — это нормально для первого запуска

    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = _load_markdown_documents(Path(data_dir))
    if not documents:
        raise FileNotFoundError(
            f"В каталоге {data_dir!r} нет Markdown-файлов. "
            "Сначала вызовите src.pipeline.build_knowledge_base()."
        )

    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
