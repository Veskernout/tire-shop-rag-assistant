"""Парсер страницы доставки (локальная HTML-заглушка)."""
from __future__ import annotations

from pathlib import Path
from bs4 import BeautifulSoup

from src.utils import save_markdown


def parse_delivery_html(html_path: str | Path) -> str:
    """Читает локальный HTML и возвращает Markdown."""
    html = Path(html_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    container = soup.find("div", class_="delivery-content") or soup
    for tag in container(["script", "style"]):
        tag.decompose()

    lines: list[str] = ["# Политика доставки", ""]
    for el in container.find_all(["h1", "h2", "h3", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in {"h1", "h2", "h3"}:
            lines.append(f"### {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines)


def parse_and_save_delivery(html_path: str | Path, out_path: str | Path) -> Path:
    md = parse_delivery_html(html_path)
    return save_markdown(md, out_path)