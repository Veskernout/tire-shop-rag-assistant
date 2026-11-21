"""Тесты для парсеров (src/parsers/)."""
from pathlib import Path

from src.parsers.catalog import Product, parse_catalog_html
from src.parsers.delivery import parse_delivery_html
from src.parsers.payment import parse_payment_html

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_parse_catalog_returns_two_products():
    products = parse_catalog_html(SAMPLES_DIR / "catalog.html")
    assert len(products) == 2
    assert all(isinstance(p, Product) for p in products)


def test_parse_catalog_product_fields():
    products = parse_catalog_html(SAMPLES_DIR / "catalog.html")
    p = products[0]
    assert p.name == "Demo Winter A 205/55 R16"
    assert p.price == "7990"
    assert p.season == "Зима"
    assert p.spikes == "Да"


def test_parse_catalog_second_product_no_spikes():
    products = parse_catalog_html(SAMPLES_DIR / "catalog.html")
    p = products[1]
    assert p.name == "Demo Nordic B 215/60 R17"
    assert p.spikes == "Нет"
    assert p.price == "9450"


def test_parse_delivery_returns_markdown():
    md = parse_delivery_html(SAMPLES_DIR / "delivery.html")
    assert "# Политика доставки" in md
    assert "1000" in md
    assert "500" in md


def test_parse_payment_returns_markdown():
    md = parse_payment_html(SAMPLES_DIR / "payment.html")
    assert "# Способы оплаты" in md
    assert "наличн" in md.lower()
    assert "карт" in md.lower()
