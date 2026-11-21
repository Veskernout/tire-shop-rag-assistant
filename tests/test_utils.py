"""Тесты для утилит (src/utils.py)."""
from pathlib import Path

from src.utils import clean_markdown, save_markdown


def test_clean_markdown_removes_double_empty_lines():
    text = "Строка 1\n\n\nСтрока 2\n\n\n\nСтрока 3\n"
    result = clean_markdown(text)
    assert "\n\n\n" not in result
    assert result.endswith("\n")


def test_clean_markdown_preserves_single_empty_lines():
    text = "Строка 1\n\nСтрока 2\n"
    result = clean_markdown(text)
    assert "\n\nСтрока 2\n" in result


def test_save_markdown_writes_file(tmp_path: Path):
    out_path = tmp_path / "test.md"
    save_markdown("# Заголовок\n\nТекст", out_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "# Заголовок" in content
    assert "Текст" in content
