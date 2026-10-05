# Разбор книги E. Gans-Ruedin, «Caucasian Carpets» (1986, англ. пер. с франц.; фото M. Hilber)

Данные: `/tmp/claude-0/ruedin/pages.json` — список из 378 страниц `{page, method, chars, text, image}`.
`page` — номер страницы PDF (с 1). Печатный номер книги = page − 4.
`method`: text_layer (текстовый слой скана, обычно хороший, но ДВЕ КОЛОНКИ часто перемешаны построчно — читайте внимательно,
восстанавливая порядок фраз), ocr (распознано — хуже), empty (нет текста: обычно это цветная фотография ковра на всю страницу).

Устройство раздела «Plates and commentaries»: для каждого ковра — страница-комментарий с шапкой:
НАЗВАНИЕ (напр. «'DRAGON' RUG, KARABAKH OR SHIRVAN», «CHELABERD KARABAKH», «KUBA ZEYKHUR»), датировка, собрание и инв. номер,
Dimensions, тип узла и плотность (knots per square metre / per 10 cm), Warp/Weft/Pile, затем текст о рисунке и каймах.
Фото ковра — на соседней странице (обычно empty или ocr-мусор рядом с комментарием, чаще всего перед ним или после него).
Бывают страницы «(DETAIL)» — деталь того же или другого ковра.

## Задача
Для каждого ковра (пластины) в вашем диапазоне страниц — одна запись. Ничего не выдумывать: только то, что есть в тексте;
непрочитанное — null. Числа точно как в книге.

```json
{"plates": [{
  "pdf_page": 81, "printed_page": 77,
  "section": "Kazak",                       // раздел книги: Early|Kazak|Karabakh|Genje|Shirvan|Kuba|Dagestan|Flat-woven
  "title_raw": "LORI-PAMBAK KAZAK",          // шапка как в книге
  "type_name": "Lori-Pambak",                // название типа/композиции/села без слова группы (null, если его нет)
  "group": "Kazak",                          // группа по автору (Kazak, Karabakh, Genje, Shirvan, Kuba, Baku, Dagestan, Talish, Moghan…)
  "attribution_note": null,                  // «Karabakh or Shirvan», «(?)» и т.п.
  "date": "Mid-19th century",
  "inscribed_date": null,                    // дата, вытканная на ковре, если упомянута (напр. «1269 H = 1852»)
  "collection": "Private collection", "inv_no": null,
  "dimensions_cm": "245 x 160", "knots_per_m2": "96,100", "knot": "Symmetrical",
  "warp": "wool", "weft": "wool", "pile": "wool",
  "is_detail": false,
  "photo_pages": [80],                       // PDF-страницы с фото этого ковра (по соседству; если не уверены — пусто)
  "design_ru": "2–4 предложения по-русски: что в поле (медальоны, их число и форма, раппорт), цвета фона, особенности; без оценок.",
  "borders_ru": "1–2 предложения о каймах (или null)",
  "motifs_en": ["crab border", "S motif", "eight-pointed star"],   // названия мотивов и кайм ТОЧНО как в тексте (англ.)
  "names_local": ["Lesghi star"],            // местные/торговые названия узоров и ковров, упомянутые в тексте
  "comparisons": ["Plate on pages 120-1"],   // ссылки автора на другие ковры/книги, коротко
  "text_quality": "good|mixed|poor"
}]}
```
Перевод на русский — простой, без терминов, которых нет в тексте. Цитировать дословно не больше 1 короткой фразы.
Файл — валидный JSON (`python3 -m json.tool ФАЙЛ > /dev/null`). Не трогать другие файлы.
