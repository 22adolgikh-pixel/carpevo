# kerimov_book.py — номера таблиц и рисунков Керимова т. I «по книге».
#
# Студия нумеровала таблицы подряд по страницам скана, а в книге есть страницы-разделители глав
# (скан 249, 284, 339) и дубль страницы (322 = 323, табл. 201), поэтому с 250-й страницы номер таблицы
# уезжал на 1–4. Кроме того, на 22 листах порядок рамок студии не совпадает с напечатанными номерами
# рисунков. Карта сверена со сканами 2026-10-31: tools/kerimov_fix/book_map.json.
import json, os, re

_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "kerimov_fix", "book_map.json")
_cache = None


def _pages():
    global _cache
    if _cache is None:
        try: _cache = json.load(open(_MAP_PATH, encoding="utf-8")).get("pages", {})
        except Exception: _cache = {}
    return _cache


def page_of(sheet_rel):
    m = re.search(r"kerimov_vol1[^/]*/page_(\d+)", sheet_rel or "")
    return int(m.group(1)) if m else None


def book_ref(sheet_rel, table, fig):
    """(таблица, рисунок) по книге для рамки листа Керимова т. I; для других источников — как есть.
    Возвращает (None, None) для кадров на страницах-разделителях и дубле."""
    p = page_of(sheet_rel)
    if p is None: return table, fig
    info = _pages().get(str(p))
    if not info: return table, fig
    if info.get("drop"): return None, None
    bt = info.get("book_table", table)
    try: bf = int((info.get("figs") or {}).get(str(int(fig)), fig))
    except (TypeError, ValueError): bf = fig
    return bt, bf
