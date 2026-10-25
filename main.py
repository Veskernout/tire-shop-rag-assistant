"""Точка входа для локального запуска (VS Code).

Парсит базу знаний из data/samples, строит индекс, скачивает LLM и
прогоняет evaluation. Тот же пайплайн доступен из ноутбука:
`notebooks/01_pipeline.ipynb` (Google Colab / локальный Jupyter).
"""
from __future__ import annotations

import argparse

from src.pipeline import build_knowledge_base, build_rag_stack, detect_gpu_layers, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tire Shop RAG Assistant — сборка базы знаний и evaluation"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="форсировать CPU (не оффлоадить слои LLM в GPU)",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="пропустить скачивание LLM и evaluation (только парсинг + индекс)",
    )
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="не очищать коллекцию ChromaDB перед пересборкой",
    )
    args = parser.parse_args()

    build_knowledge_base()

    if args.skip_model:
        from src.rag.embeddings import configure_embeddings
        from src.rag.index import build_index

        configure_embeddings()
        build_index("data", reset=not args.keep_collection)
        print("Готово: база знаний собрана, индекс создан (LLM пропущен, --skip-model).")
        return

    gpu_layers = detect_gpu_layers(force_cpu=args.cpu)
    print(f"n_gpu_layers={gpu_layers} ({'GPU' if gpu_layers == -1 else 'CPU'})")

    engine = build_rag_stack(gpu_layers=gpu_layers, reset_collection=not args.keep_collection)
    run_evaluation(engine)


if __name__ == "__main__":
    main()
