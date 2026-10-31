#!/usr/bin/env python3
# apply_kerimov_fix.py — исправить подписи схем Керимова т. I в данных студии по сверке со сканами (2026-10-31).
#
#   python3 apply_kerimov_fix.py /путь/к/data            # сухой прогон: только отчёт
#   python3 apply_kerimov_fix.py /путь/к/data --apply    # записать
#
# Что делает:
#  1. Бэкап: data/backup_kerimov_fix_<время>/ (legend.csv и meta всех затронутых work/*.json).
#  2. legend.csv ← legend_book.csv (легенда, переписанная со сканов стр. 85–117, с проверенной латиницей).
#  3. Для каждой схемы t*_f* с листа kerimov_vol1: book_table/book_fig по книге (book_map.json),
#     meta.name/name_az/translation/carpet/type/note — из книжной легенды, meta.source = «Табл. X, рис. Y».
#     Привязка к сайту (site_id) ставилась по старому названию, т. е. по строке легенды (табл., рис.) студии —
#     поэтому она переезжает на схему, у которой теперь этот книжный номер (если название совпадает);
#     иначе снимается, старое значение — в meta.site_id_prev. Пишется remap.json {старая схема → новая}
#     для решений по схемам (site/model/apply_decisions.py --remap).
#  4. Кадры со страниц-разделителей и дубля 323: done=False, kerimov_fix="drop" (уходят из выгрузки).
import csv, json, os, re, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from kerimov_book import book_ref, page_of  # noqa: E402

FIELDS = ("name_az", "name", "translation", "carpet", "type", "note")


def norm(s): return re.sub(r"[\s«»\"„“”\-—–.,|()]", "", s or "").lower().replace("ё", "е").replace("ә", "э").replace("ь", "")


def main(data, apply=False):
    legend = {}
    for r in csv.DictReader(open(os.path.join(HERE, "legend_book.csv"), encoding="utf-8")):
        legend[(int(r["table"]), int(r["fig"]))] = r
    wdir = os.path.join(data, "work")
    stat, changed, report = {}, {}, []
    # проход 1: книжные номера всех схем и старые привязки
    allw, by_book = {}, {}
    for fn in sorted(os.listdir(wdir)):
        if not re.match(r"t\d+_f\d+\.json$", fn): continue
        w = json.load(open(os.path.join(wdir, fn), encoding="utf-8"))
        if page_of(w.get("sheet")) is None: continue
        bt, bf = book_ref(w.get("sheet"), w.get("table"), w.get("fig"))
        allw[fn] = (w, bt, bf)
        if bt is not None: by_book[(bt, bf)] = fn
    remap, moved_site = {}, {}
    for fn, (w, bt, bf) in allw.items():
        m = w.get("meta") or {}
        tgt = by_book.get((w.get("table"), w.get("fig")))   # старая подпись схемы = строка легенды с её студийным номером
        remap[fn[:-5]] = tgt[:-5] if tgt else None
        sid = (m.get("site_id") or "").strip()
        if sid and tgt:
            lg = legend.get((w.get("table"), w.get("fig"))) or {}
            on, nn = norm(m.get("name")), norm(lg.get("name"))
            if on and nn and (on == nn or nn.startswith(on) or on.startswith(nn)): moved_site[tgt] = sid
    for fn in sorted(allw):
        if not re.match(r"t\d+_f\d+\.json$", fn): continue
        w, bt, bf = allw[fn]
        m = dict(w.get("meta") or {})
        if bt is None:
            if w.get("kerimov_fix") != "drop":
                w["kerimov_fix"] = "drop"; w["done"] = False; changed[fn] = w
            stat["drop"] = stat.get("drop", 0) + 1; report.append((fn, "drop", "", "")); continue
        lg = legend.get((bt, bf))
        w["book_table"], w["book_fig"] = bt, bf
        new = dict(m)
        new["source"] = f"Табл. {bt}, рис. {bf}"
        if lg:
            for k in FIELDS:
                v = (lg.get(k) or "").strip()
                new[k] = "" if v == "?" else v
            new.pop("name_az_auto", None)
            old_sid = (m.get("site_id") or "").strip(); sid = moved_site.get(fn, "")
            if sid != old_sid:
                new["site_id"] = sid
                if old_sid: new["site_id_prev"] = old_sid
        kind = "same" if all(norm(m.get(k)) == norm(new.get(k)) for k in ("name", "translation", "carpet")) else \
               ("filled" if not (m.get("name") or m.get("carpet")) else
                ("renamed" if norm(m.get("name")) != norm(new.get("name")) else "fixed"))
        if not lg: kind = "no_legend"
        stat[kind] = stat.get(kind, 0) + 1
        if new != m or w.get("book_table") != bt:
            w["meta"] = new; w["kerimov_fix"] = "2026-10-31"; changed[fn] = w
        if kind != "same": report.append((fn, kind, m.get("name", ""), new.get("name", "")))
    print("схем Керимова:", sum(stat.values()), stat)
    print("файлов к записи:", len(changed), "| привязок к сайту на новом месте:", len(moved_site))
    for r in report[:25]: print("  ", *r)
    if not apply:
        print("\nСухой прогон. Для записи: --apply"); return
    bk = os.path.join(data, "backup_kerimov_fix_" + time.strftime("%Y%m%d_%H%M%S")); os.makedirs(os.path.join(bk, "work"))
    if os.path.exists(os.path.join(data, "legend.csv")): shutil.copy2(os.path.join(data, "legend.csv"), bk)
    for fn in changed: shutil.copy2(os.path.join(wdir, fn), os.path.join(bk, "work", fn))
    shutil.copy2(os.path.join(HERE, "legend_book.csv"), os.path.join(data, "legend.csv"))
    for fn, w in changed.items():
        tmp = os.path.join(wdir, fn + ".tmp")
        json.dump(w, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(tmp, os.path.join(wdir, fn))
    json.dump([list(r) for r in report], open(os.path.join(bk, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(remap, open(os.path.join(bk, "remap.json"), "w", encoding="utf-8"), indent=0)
    print("записано; бэкап:", bk)


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__ or "usage: apply_kerimov_fix.py DATA [--apply]")
    main(sys.argv[1], "--apply" in sys.argv)
