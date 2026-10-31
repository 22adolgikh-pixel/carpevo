#!/usr/bin/env python3
# patch_schemes_db.py — перенести исправленные подписи схем Керимова (после apply_kerimov_fix.py) во вход сайта.
#   python3 patch_schemes_db.py site/proto/inputs/schemes_db_full.json.gz DATA/work
# Обновляет записи t*_f*: название (az / книга), перевод, ковёр, тип, примечание, номер табл./рис. по книге,
# привязки к понятиям (site_ids); убирает кадры разделителей; добавляет принятые схемы, которых во входе не было.
import gzip, json, os, re, sys


def main(db_path, wdir):
    db = json.load(gzip.open(db_path))
    by = {x.get("id"): i for i, x in enumerate(db) if x.get("id")}
    upd = add = drop = 0
    for fn in sorted(os.listdir(wdir)):
        if not re.match(r"t\d+_f\d+\.json$", fn): continue
        w = json.load(open(os.path.join(wdir, fn), encoding="utf-8"))
        if not w.get("kerimov_fix"): continue
        sid = w["id"]
        if w["kerimov_fix"] == "drop":
            if sid in by: db[by[sid]] = None; drop += 1
            continue
        if not w.get("matrix") or not w.get("done"): continue
        m = w.get("meta") or {}
        bt, bf = w.get("book_table") or w.get("table"), w.get("book_fig") or w.get("fig")
        carpet = (m.get("carpet") or "").strip()
        f = {"name": m.get("name_az") or m.get("name", ""), "name_book": m.get("name", ""), "translation": m.get("translation", ""),
             "carpet": carpet, "carpet_i18n": {"ru": carpet} if carpet else None, "type": m.get("type", ""), "note": m.get("note", ""),
             "table": bt, "fig": bf, "cps_table": w.get("table"), "cps_fig": w.get("fig"),
             "source": f"Табл. {bt}, рис. {bf}",
             "source_i18n": {"ru": f"Табл. {bt}, рис. {bf}", "az": f"Cədvəl {bt}, şəkil {bf}", "en": f"Table {bt}, fig. {bf}"},
             "site_ids": [s.strip() for s in (m.get("site_id") or "").split(",") if s.strip()],
             "kerimov_fix": "2026-10-31"}
        if sid in by:
            x = db[by[sid]]
            if (x.get("name_book") or "") != f["name_book"]:
                x["names"] = {"az": f["name"], "ru": f["translation"], "en": ""}
            x.update(f); upd += 1
        else:
            pg = re.search(r"page_(\d+)", w.get("sheet", ""))
            x = {"id": sid, "w": w["matrix"]["w"], "h": w["matrix"]["h"], "rows": w["matrix"]["rows"],
                 "palette": w.get("palette"), "palette_names": w.get("palette_names", []), "grid_origin": "hand_drawn",
                 "names": {"az": f["name"], "ru": f["translation"], "en": ""}, "section": "", "link_origin": "none",
                 "source_ref": {"scan": w.get("sheet"), "scan_page": int(pg.group(1)) if pg else None, "source_id": "kerimov_vol1"},
                 "updated": w.get("saved_at", "")}
            x.update(f); db.append(x); add += 1
    db = [x for x in db if x is not None]
    with gzip.open(db_path, "wt", encoding="utf-8") as fh: json.dump(db, fh, ensure_ascii=False)
    print(f"обновлено {upd}, добавлено {add}, убрано {drop}; всего {len(db)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
