# rug_bundle_colab.py — только собрать пакет для студии из уже готового каталога (carpet-dna/rug_catalog/catalog.json + _cache/img)
# и загрузить его на Диск с доступом «по ссылке». Нужен, если rug_catalog_colab.py дошёл до конца анализа, но оборвался на загрузке.
# Видеокарта не нужна (среда CPU годится). Время: 10–20 мин (чтение ~3200 фото с Диска + загрузка ~300–400 МБ).
import os, json, zipfile, time
from google.colab import drive, auth
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/carpet-dna/rug_catalog'; IMG = f'{OUT}/_cache/img'; BZ = '/content/rugs_bundle.zip'
t0 = time.time()
cat = json.load(open(f'{OUT}/catalog.json', encoding='utf-8'))
items = cat['items']; miss = 0
with zipfile.ZipFile(BZ, 'w', zipfile.ZIP_STORED) as z:
    z.writestr('catalog.json', json.dumps(cat, ensure_ascii=False))
    for n, it in enumerate(items):
        p = f'{IMG}/{it["id"]}.jpg'
        if os.path.exists(p): z.write(p, it['img'])
        else: miss += 1
        if n % 500 == 0: print(f'[{time.time() - t0:4.0f} c] фото', n, '/', len(items), flush=True)
print('пакет:', round(os.path.getsize(BZ) / 1e6), 'МБ | нет фото:', miss, flush=True)
auth.authenticate_user()
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
svc = build('drive', 'v3')
idf = f'{OUT}/bundle_id.txt'; fid = open(idf).read().strip() if os.path.exists(idf) else ''
media = MediaFileUpload(BZ, mimetype='application/zip', resumable=True, chunksize=50 * 1024 * 1024)
if fid:
    try: svc.files().update(fileId=fid, media_body=media).execute()
    except Exception as e: print('старый файл не обновился, создаю новый:', e); fid = ''
if not fid:
    folder = svc.files().list(q="name='rug_catalog' and mimeType='application/vnd.google-apps.folder' and trashed=false", fields='files(id)').execute()['files']
    req = svc.files().create(body={'name': 'rugs_bundle.zip', **({'parents': [folder[0]['id']]} if folder else {})}, media_body=media, fields='id')
    resp = None
    while resp is None:
        st, resp = req.next_chunk()
        if st: print(f'загрузка {int(st.progress() * 100)}%', flush=True)
    fid = resp['id']
    svc.permissions().create(fileId=fid, body={'type': 'anyone', 'role': 'reader'}).execute()
    open(idf, 'w').write(fid)
print('\nИТОГ: ковров', len(items), '| групп двойников', len(cat.get('groups', [])))
print('id пакета для студии:', fid)
