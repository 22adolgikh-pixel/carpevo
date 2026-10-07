# museum_photos_colab.py — скачать фото ковров из открытых коллекций (список: site/model/sources/museum_photos/catalog_photos.json)
# Результат на Drive (папка carpet-dna/photos/):
#   full/<id>.jpg     — фото для обучения (как отдаёт источник, до ~840 px)
#   thumbs/<id>.jpg   — миниатюры 320 px для сайта
#   thumbs_partN.zip  — миниатюры пачками ≤ 2 МБ (их забирает Claude через Google Drive)
#   manifest.json     — что скачано, размеры, ошибки
# Запуск в Colab:
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %run /content/carpevo/cps/tools/colab/museum_photos_colab.py
# Повторный запуск докачивает только недостающее.
import os, json, io, re, time, zipfile, hashlib
import requests
from PIL import Image
from google.colab import drive

drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/carpet-dna/photos'
for d in ('full', 'thumbs'):
    os.makedirs(f'{ROOT}/{d}', exist_ok=True)
items = json.load(open('/content/carpevo/site/model/sources/museum_photos/catalog_photos.json'))['items']
H = {'User-Agent': 'CarpetDNA/1.0 (research catalogue of Azerbaijani carpets; contact via github.com/22adolgikh-pixel)'}


def fname(pid):
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', pid)[:80]
    return s + '_' + hashlib.md5(pid.encode()).hexdigest()[:6]


def big_url(p):
    u = p['img']
    if p['src'] == 'wikimedia': return u.replace('/400px-', '/500px-')   # стандартный размер Wikimedia (уже в кэше) — не упирается в лимит 429
    if p['src'] == 'aic': return u.replace('/full/400,/', '/full/843,/')
    return u


man = {}
mp = f'{ROOT}/manifest.json'
if os.path.exists(mp): man = json.load(open(mp))
t0 = time.time()
for i, p in enumerate(items):
    fn = fname(p['id'])
    if man.get(p['id'], {}).get('ok') and os.path.exists(f'{ROOT}/thumbs/{fn}.jpg'):
        continue
    try:
        hd = dict(H)
        if p['src'] == 'aic':   # IIIF Чикаго отвечает 403 на «небраузерные» запросы
            hd = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36',
                  'AIC-User-Agent': H['User-Agent'], 'Referer': 'https://www.artic.edu/'}
        r = None
        for attempt in range(6):   # Wikimedia: 429 Too Many Requests → ждём и повторяем
            r = requests.get(big_url(p), headers=hd, timeout=60)
            if r.status_code == 429:
                w = min(int(r.headers.get('Retry-After', 0) or 0) or 3 * (attempt + 1), 20)
                print(f'   {p["id"][:50]}: 429, жду {w} с', flush=True); time.sleep(w); continue
            break
        if r.status_code != 200 and big_url(p) != p['img']:
            r = requests.get(p['img'], headers=hd, timeout=60)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert('RGB')
        im.save(f'{ROOT}/full/{fn}.jpg', quality=90)
        th = im.copy(); th.thumbnail((320, 320))
        th.save(f'{ROOT}/thumbs/{fn}.jpg', quality=78)
        man[p['id']] = {'ok': True, 'file': fn + '.jpg', 'w': im.width, 'h': im.height}
    except Exception as e:
        man[p['id']] = {'ok': False, 'err': str(e)[:200]}
    if p['src'] == 'wikimedia': time.sleep(1.0)
    done = sum(1 for v in man.values() if v.get('ok'))
    print(f'{i + 1}/{len(items)} · скачано {done} · {(time.time() - t0) / 60:.1f} мин · {"ок" if man[p["id"]].get("ok") else "ошибка"}', flush=True)
    if i % 10 == 0: json.dump(man, open(mp, 'w'))
json.dump(man, open(mp, 'w'))

# миниатюры пачками ≤ 2 МБ
for f in os.listdir(ROOT):
    if f.startswith('thumbs_part') and f.endswith('.zip'): os.remove(f'{ROOT}/{f}')
part, size, z = 1, 0, None
for pid, m in sorted(man.items()):
    if not m.get('ok'): continue
    fp = f'{ROOT}/thumbs/{m["file"]}'
    if not os.path.exists(fp): continue
    s = os.path.getsize(fp)
    if z is None or size + s > 2_000_000:
        if z: z.close(); part += 1
        z = zipfile.ZipFile(f'{ROOT}/thumbs_part{part}.zip', 'w', zipfile.ZIP_STORED); size = 0
    z.write(fp, m['file']); size += s
if z: z.close()
ok = sum(1 for m in man.values() if m.get('ok'))
print('ИТОГ', json.dumps({'всего': len(items), 'скачано': ok, 'ошибок': len(items) - ok, 'пачек': part}, ensure_ascii=False))
print('Готово. Напишите Claude: «фото скачаны».')
