"""Настройка LLM и query engine."""
from __future__ import annotations

import os
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.llama_cpp import LlamaCPP

from src.rag.prompts import build_qa_prompt, completion_to_prompt, messages_to_prompt

# Модель можно переопределить через переменные окружения
# (например, для быстрой проверки на маленькой GGUF в Colab/локально):
#   RAG_MODEL_REPO=Qwen/Qwen2.5-0.5B-Instruct-GGUF
MODEL_REPO = os.environ.get("RAG_MODEL_REPO", "TheBloke/saiga_mistral_7b-GGUF")
MODEL_FILE = os.environ.get("RAG_MODEL_FILE", "saiga_mistral_7b.Q4_K_M.gguf")
MODEL_DIR = Path(os.environ.get("RAG_MODEL_DIR", "./models"))

# Cross-encoder для rerank'а — это отдельная архитектура от биэнкодер-эмбеддингов.
RERANK_MODEL = os.environ.get(
    "RAG_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

CONTEXT_WINDOW = 4096


def ensure_model() -> Path:
    """Скачивает GGUF-модель с Hugging Face Hub, если её ещё нет."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILE
    if not model_path.exists():
        from huggingface_hub import hf_hub_download

        print(f"Скачивание модели {MODEL_REPO}/{MODEL_FILE} …")
        hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=str(MODEL_DIR))
    return model_path


def build_llm(model_path: Path, n_gpu_layers: int = -1) -> LlamaCPP:
    """Llama.cpp с ChatML-форматированием из src/rag/prompts.py."""
    return LlamaCPP(
        model_path=str(model_path),
        temperature=0.05,
        max_new_tokens=256,
        context_window=CONTEXT_WINDOW,
        model_kwargs={
            "n_gpu_layers": n_gpu_layers,  # -1 = все слои на GPU, 0 = CPU
            "n_threads": os.cpu_count() or 4,
            "n_ctx": CONTEXT_WINDOW,  # единый размер контекста
            "verbose": False,
            "stop": ["\n\n", "###", "<|im_end|>", "Пользователь:", "User:"],
            "repeat_penalty": 1.2,
            "top_p": 0.9,
            "top_k": 30,
        },
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False,
    )


def build_reranker() -> SentenceTransformerRerank:
    return SentenceTransformerRerank(model=RERANK_MODEL, top_n=3)


def build_query_engine(index: VectorStoreIndex, llm: LlamaCPP, reranker: SentenceTransformerRerank):
    """Retrieval top-5 → rerank top-3 → стриктивный anti-hallucination промпт."""
    Settings.llm = llm
    return index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[reranker],
        text_qa_template=build_qa_prompt(),
        response_mode="compact",
        streaming=False,
        # ChromaVectorStore по умолчанию шлёт where={}, что не проходит
        # валидацию в chromadb>=0.5; None означает «без метаданных-фильтров».
        vector_store_kwargs={"where": None},
    )
