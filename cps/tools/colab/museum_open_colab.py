# museum_open_colab.py — фото кавказских/азербайджанских ковров из открытых API музеев (V&A + Метрополитен, дополнение к museum_photos_colab.py).
# Берёт только предметы с картинкой; подписи музея (место, дата, название, номер) — как есть, это «хорошая» разметка.
# Лицензии: Met — только isPublicDomain (CC0); V&A — изображения разрешены для некоммерческого использования (на сайт без проверки не публиковать).
# Результат: Drive carpet-dna/photos/museum_open/{img/<id>.jpg, catalog.csv, catalog.json}. Повторный запуск докачивает недостающее.
# Видеокарта не нужна. Время: 10–30 минут.
import os, re, json, csv, time
import requests
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/carpet-dna/photos/museum_open'; os.makedirs(f'{OUT}/img', exist_ok=True)
H = {'User-Agent': 'CarpetDNA/1.0 (research catalogue of Azerbaijani carpets; github.com/22adolgikh-pixel)'}
S = requests.Session(); S.headers.update(H)
RUG = re.compile(r'carpet|rug|kilim|sumakh|soumak|palas|jajim|zili|verneh|khorjin|mafrash|bag face', re.I)
CAU = re.compile(r'caucas|azerbaijan|shirvan|kuba|quba|baku|karabag|karabakh|kazak|gazakh|ganja|gendje|genje|talish|lenkoran|shusha|derbent|daghestan|borchal|moghan|mughan|tabriz|nakhichevan|nakhchivan|zakatal', re.I)
items = {}


def get(url, **kw):
    for a in range(4):
        try:
            r = S.get(url, timeout=40, **kw)
            if r.status_code == 200: return r
            if r.status_code in (429, 503): time.sleep(5 * (a + 1)); continue
            return None
        except Exception: time.sleep(3)
    return None


# ---- V&A ----
seen = set()
for q in ('carpet', 'rug', 'kilim', 'sumak', 'bag'):
    for place in ('Caucasus', 'Azerbaijan', 'Daghestan', 'Iran'):
        page = 1
        while True:
            r = get('https://api.vam.ac.uk/v2/objects/search', params={'q': q, 'q_place_name': place, 'images_exist': 1, 'page_size': 100, 'page': page})
            if not r: break
            j = r.json(); rec = j.get('records', [])
            for o in rec:
                sid = o.get('systemNumber')
                if not sid or sid in seen: continue
                seen.add(sid)
                txt = ' '.join(str(o.get(k, '')) for k in ('objectType', '_primaryTitle', '_primaryPlace'))
                if not RUG.search(txt) or not CAU.search(txt + ' ' + place): continue
                base = (o.get('_images') or {}).get('_iiif_image_base_url')
                if not base: continue
                items['vam_' + sid] = dict(id='vam_' + sid, museum='V&A', inv=o.get('accessionNumber', ''), title=o.get('_primaryTitle') or o.get('objectType', ''),
                                           type=o.get('objectType', ''), place=o.get('_primaryPlace', ''), date=o.get('_primaryDate', ''),
                                           url=f'https://collections.vam.ac.uk/item/{sid}/', img=base + 'full/!1200,1200/0/default.jpg', license='V&A non-commercial')
            if page >= j.get('info', {}).get('pages', 1) or not rec: break
            page += 1
print('V&A:', sum(1 for k in items if k.startswith('vam_')), flush=True)

# ---- Метрополитен (только public domain) ----
ids = set()
for q in ('Caucasus', 'Azerbaijan', 'Shirvan', 'Kuba', 'Kazak', 'Karabagh', 'Baku', 'Daghestan', 'Talish', 'Gendje', 'Soumak', 'Tabriz carpet'):
    r = get('https://collectionapi.metmuseum.org/public/collection/v1/search', params={'hasImages': 'true', 'q': q})
    if r: ids |= set(r.json().get('objectIDs') or [])
print('Met: кандидатов', len(ids), flush=True)
for n, oid in enumerate(sorted(ids)):
    r = get(f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}')
    time.sleep(0.05)
    if not r: continue
    o = r.json()
    txt = ' '.join(str(o.get(k, '')) for k in ('title', 'objectName', 'classification'))
    geo = ' '.join(str(o.get(k, '')) for k in ('country', 'region', 'subregion', 'culture', 'title'))
    if not (o.get('isPublicDomain') and o.get('primaryImage') and RUG.search(txt) and CAU.search(geo)): continue
    items[f'met_{oid}'] = dict(id=f'met_{oid}', museum='Met', inv=o.get('accessionNumber', ''), title=o.get('title', ''), type=o.get('objectName', ''),
                               place=', '.join(x for x in (o.get('country'), o.get('region'), o.get('subregion')) if x), date=o.get('objectDate', ''),
                               url=o.get('objectURL', ''), img=o.get('primaryImage'), license='CC0')
    if n % 100 == 0: print('  Met', n, '/', len(ids), flush=True)
print('Met:', sum(1 for k in items if k.startswith('met_')), flush=True)

# ---- скачивание ----
ok = 0
for i, it in enumerate(items.values()):
    p = f'{OUT}/img/{it["id"]}.jpg'; it['file'] = f'img/{it["id"]}.jpg'
    if os.path.exists(p) and os.path.getsize(p) > 5000: ok += 1; continue
    r = get(it['img'])
    if r and len(r.content) > 5000:
        open(p, 'wb').write(r.content); ok += 1
    else: it['file'] = ''
    if i % 50 == 0: print('  фото', i, '/', len(items), flush=True)
L = list(items.values())
json.dump(L, open(f'{OUT}/catalog.json', 'w'), ensure_ascii=False, indent=1)
with open(f'{OUT}/catalog.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(L[0].keys())); w.writeheader(); w.writerows(L)
print(f'ИТОГ: предметов {len(L)} (V&A {sum(1 for x in L if x["museum"]=="V&A")}, Met {sum(1 for x in L if x["museum"]=="Met")}), фото скачано {ok}')
print('Готово. Напишите Claude: «музеи скачаны».')
