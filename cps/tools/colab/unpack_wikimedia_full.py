# unpack_wikimedia_full.py — распаковать wikimedia_full_500.zip (скачан через браузер) в carpet-dna/photos/full и дописать manifest.json
# Запуск в Colab:  %run /content/carpevo/cps/tools/colab/unpack_wikimedia_full.py
import os, json, zipfile, re, hashlib
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/carpet-dna/photos'
items = json.load(open('/content/carpevo/site/model/sources/museum_photos/catalog_photos.json'))['items']
def fname(pid): return re.sub(r'[^A-Za-z0-9._-]+', '_', pid)[:80] + '_' + hashlib.md5(pid.encode()).hexdigest()[:6]
by = {fname(p['id']) + '.jpg': p['id'] for p in items}
mp = f'{ROOT}/manifest.json'
man = json.load(open(mp)) if os.path.exists(mp) else {}
os.makedirs(f'{ROOT}/full', exist_ok=True)
z = zipfile.ZipFile(f'{ROOT}/wikimedia_full_500.zip'); ok = bad = 0
for nm in z.namelist():
    pid = by.get(nm)
    if not pid: bad += 1; continue
    try:
        im = Image.open(z.open(nm)).convert('RGB'); im.save(f'{ROOT}/full/{nm}', quality=92)
        man[pid] = {'ok': True, 'file': nm, 'w': im.width, 'h': im.height}; ok += 1
    except Exception as e:
        man[pid] = {'ok': False, 'err': str(e)[:200]}; bad += 1
json.dump(man, open(mp, 'w'))
have = sum(1 for p in items if man.get(p['id'], {}).get('ok') and os.path.exists(f'{ROOT}/full/{man[p["id"]]["file"]}'))
print(f'распаковано {ok}, пропущено {bad}; полноразмерных фото теперь {have} из {len(items)}')
