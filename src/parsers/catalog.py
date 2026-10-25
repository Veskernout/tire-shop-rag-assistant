"""Парсер каталога шин (локальная HTML-заглушка)."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup

from src.utils import save_markdown


@dataclass
class Product:
    name: str
    code: str
    season: str
    spikes: str
    availability: str
    pickup: str
    delivery: str
    price: str


def parse_catalog_html(html_path: str | Path) -> list[Product]:
    html = Path(html_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    products: list[Product] = []
    for card in soup.find_all("div", class_="catalog-product"):
        name_tag = card.find("a", class_="product-name")
        name = name_tag.get_text(" ", strip=True) if name_tag else "Не указано"

        code_tag = card.find("div", class_="product-code")
        code = code_tag.get_text(strip=True) if code_tag else "Не указан"

        season = spikes = "Не указано"
        for li in card.select("ul.properties li"):
            text = li.get_text(" ", strip=True)
            if text.startswith("Сезон"):
                season = text.replace("Сезон", "").strip(" :")
            elif text.startswith("Шипы"):
                spikes = text.replace("Шипы", "").strip(" :")

        availability = pickup = delivery = "Не указано"
        for li in card.select("ul.delivery li"):
            text = li.get_text(" ", strip=True)
            if text.startswith("Наличие"):
                availability = text.replace("Наличие", "").strip(" :")
            elif text.startswith("Самовывоз"):
                pickup = text.replace("Самовывоз", "").strip(" :")
            elif text.startswith("Доставка"):
                delivery = text.replace("Доставка", "").strip(" :")

        price = "Не указана"
        price_tag = card.find("span", class_="price")
        if price_tag:
            m = re.search(r"[\d\s]+(?:[.,]\d{2})?", price_tag.get_text())
            if m:
                price = m.group(0).replace(" ", "").replace(",", ".")

        products.append(Product(name, code, season, spikes, availability, pickup, delivery, price))

    return products


def format_catalog_markdown(products: list[Product]) -> str:
    if not products:
        return "# Каталог шин\n\nТовары не найдены.\n"

    lines = ["# Каталог шин", "", f"*Всего товаров: {len(products)}*", ""]
    for i, p in enumerate(products, 1):
        lines += [
            f"## Товар #{i}",
            "",
            f"**{p.name}**",
            "",
            f"- **Код товара:** {p.code}",
            f"- **Сезон:** {p.season}",
            f"- **Шипы:** {p.spikes}",
            f"- **Цена:** {p.price} ₽",
            "",
            "### Доставка и наличие",
            f"- **Наличие:** {p.availability}",
            f"- **Самовывоз:** {p.pickup}",
            f"- **Доставка:** {p.delivery}",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def parse_and_save_catalog(html_path: str | Path, out_path: str | Path) -> Path:
    products = parse_catalog_html(html_path)
    return save_markdown(format_catalog_markdown(products), out_path)