"""Оркестрация пайплайна: база знаний, индексация, evaluation, GPU-детект.

Единый источник истины для двух интерфейсов:
- `main.py` (локальный запуск в VS Code);
- `notebooks/01_pipeline.ipynb` (Google Colab / локальный Jupyter).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from src.parsers.catalog import parse_and_save_catalog
from src.parsers.delivery import parse_and_save_delivery
from src.parsers.payment import parse_and_save_payment
from src.rag.embeddings import configure_embeddings
from src.rag.engine import build_llm, build_query_engine, build_reranker, ensure_model
from src.rag.index import build_index
from src.utils import ensure_dir

SAMPLES_DIR = Path("data/samples")
DATA_DIR = Path("data")
EVAL_PATH = Path("evaluation/qa_pairs.json")


def build_knowledge_base(
    samples_dir: str | Path = SAMPLES_DIR,
    data_dir: str | Path = DATA_DIR,
) -> None:
    """Парсит локальные HTML-заглушки и сохраняет базу знаний в data/*.md."""
    samples, out = Path(samples_dir), Path(data_dir)
    ensure_dir(out)
    parse_and_save_delivery(samples / "delivery.html", out / "delivery_policy.md")
    parse_and_save_payment(samples / "payment.html", out / "payment_policy.md")
    parse_and_save_catalog(samples / "catalog.html", out / "product_catalog.md")
    print(f"База знаний собрана: {out}/*.md")


def detect_gpu_layers(force_cpu: bool = False) -> int:
    """Определяет число слоёв LLM для оффлоада в GPU.

    Возвращает -1 (все слои), если найдена CUDA-совместимая GPU,
    иначе 0 (чистый CPU). В Colab GPU-рантайм даёт `nvidia-smi`.
    """
    if force_cpu:
        return 0

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"], capture_output=True, text=True, timeout=15
            )
            if "GPU" in result.stdout:
                return -1
        except Exception:
            pass

    try:
        import torch  # в Colab уже установлен

        if torch.cuda.is_available():
            return -1
    except Exception:
        pass
    return 0


def build_rag_stack(gpu_layers: int, reset_collection: bool = True):
    """Собирает весь RAG-стек: эмбеддинги → индекс → LLM → query engine."""
    configure_embeddings()
    index = build_index(DATA_DIR, reset=reset_collection)
    model_path = ensure_model()
    llm = build_llm(model_path, n_gpu_layers=gpu_layers)
    return build_query_engine(index, llm, build_reranker())


def run_evaluation(engine, eval_path: str | Path = EVAL_PATH, preview_chars: int = 300) -> dict:
    """Прогоняет QA-пары из evaluation/qa_pairs.json и печатает статистику."""
    pairs = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    total = len(pairs)
    passed = 0

    for item in pairs:
        question = item["question"]
        expected = item.get("expected_keywords", [])
        answer = str(engine.query(question))
        hit = all(kw.lower() in answer.lower() for kw in expected)
        passed += int(hit)

        preview = answer if len(answer) <= preview_chars else answer[:preview_chars] + "…"
        print(f"[{'OK' if hit else 'MISS'}] {question}\n  → {preview}\n")

    stats = {"total": total, "passed": passed, "accuracy": passed / total}
    print(f"✅ Passed {passed}/{total} ({stats['accuracy']:.0%})")
    return stats
