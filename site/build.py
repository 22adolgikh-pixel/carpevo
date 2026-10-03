"""Сборка сайта «Ковровое ДНК»: template.html + data.json → index.html (публикуется как артефакт).
data.json содержит цитаты из книг — в публичный репозиторий НЕ коммитится (лежит в Project «CarpetsCat» и на Drive).
Использование: python build.py [data.json] [out.html]"""
import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data.json", encoding="utf8"))
t = open("template.html", encoding="utf8").read()
dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
vis, sl = d.pop("visual_data"), d.pop("scan_links")
t = t.replace("/*@@VISUAL_DATA@@*/{}", dump(vis)).replace("/*@@SCAN_LINKS@@*/{}", dump(sl)).replace("/*@@DATA@@*/{}", dump(d))
open(sys.argv[2] if len(sys.argv) > 2 else "index.html", "w", encoding="utf8").write(t)
print("ok", len(t))
