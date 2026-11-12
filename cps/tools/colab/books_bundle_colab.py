# books_bundle_colab.py — фото ковров из книг → пакет-дополнение для студии (CPS) + полный OCR энциклопедии для Carpet DNA.
#
# Что делает (одна ячейка, видеокарта не нужна):
#   1. Берёт архивы разбора книг (book_extract_colab v5): Энциклопедия т. II и Мурадов «Qarabağ qrupu».
#      Ищет в /content/out/ и на Диске (carpet-dna/output/books/**). Если архива Мурадова нет — заново режет книгу (≈3 мин).
#   2. Собирает books_bundle.zip = catalog.json (из carpet-dna/output/books/books_catalog.json, его готовит Claude) + img/<id>.jpg,
#      загружает в carpet-dna/rug_catalog/ с доступом «по ссылке» и печатает id. Основной rugs_bundle.zip не трогается.
#      В студии: «3 · Ковры» → «⟳ из Диска» → вставить этот id. Пакет помечен extra: основной id пакета в студии не меняется.
#   3. OCR всех страниц энциклопедии (tesseract aze+rus, 250 dpi) → carpet-dna/output/books/ensiklopediya_ii/ocr_pages.jsonl.
#      Пишет постранично и продолжает с места обрыва, если сессия прервалась (просто запустить ячейку ещё раз). ≈30–40 мин.
#
# Переменные окружения: STEPS=bundle,ocr (по умолч. обе), OCR_DPI=250, OCR_WORKERS=2.
import os, sys, json, glob, zipfile, subprocess, time, io, re
from concurrent.futures import ProcessPoolExecutor
from google.colab import drive, auth
drive.mount('/content/drive')
D = '/content/drive/MyDrive/carpet-dna'
BOOKS = f'{D}/output/books'
STEPS = os.environ.get('STEPS', 'bundle,ocr').split(',')
ENC, MUR = 'azerbaycan_xalchasi_ensiklopediyasi_ii_cild', 'azerbaycan_xalchalari_qarabag_qrupu'
NEW = ['intangible_2849', 'intangible_2850', 'intangible_2851', 'intangible_2854', 'azerbaycan_xalchasi', 'intangible_2885']   # альбомы/книги 12.11.2026 (build_series.py)
t0 = time.time()
def log(*a): print(f'[{time.time() - t0:5.0f} c]', *a, flush=True)
subprocess.run('apt-get -qq install -y tesseract-ocr tesseract-ocr-aze tesseract-ocr-rus > /dev/null 2>&1; pip -q install pymupdf pytesseract', shell=True)

def find_zip(slug):
    for p in [f'/content/out/{slug}.zip'] + glob.glob(f'{BOOKS}/**/{slug}.zip', recursive=True):
        if os.path.exists(p): return p
    return None

if 'bundle' in STEPS:
    zips = {ENC: find_zip(ENC), MUR: find_zip(MUR)}
    for _s in NEW: zips[_s] = find_zip(_s)
    if not zips[MUR]:
        log('архива Мурадова нет — режу книгу заново (≈3 мин)')
        ex = sorted(glob.glob(f'{D}/output/drive_audit/book_extract_colab_v5.py') + glob.glob(f'{D}/tools/book_extract_colab*.py'))[0]
        subprocess.run([sys.executable, ex], env={**os.environ, 'BOOK': 'Qarabag', 'STEP': 'all', 'ZIP': '1', 'PAGES': '',
                                                  'COPY_TO': f'{BOOKS}/{MUR}'}, check=False)
        zips[MUR] = find_zip(MUR)
    assert zips[ENC], 'нет архива энциклопедии (carpet-dna/output/books/ensiklopediya_ii/…zip)'
    log('архивы:', zips)
    cat = json.load(open(f'{BOOKS}/books_catalog.json', encoding='utf-8'))
    Z = {k: zipfile.ZipFile(v) for k, v in zips.items() if v}
    names = {k: set(z.namelist()) for k, z in Z.items()}
    BZ = '/content/books_bundle.zip'; miss = []; n = 0
    with zipfile.ZipFile(BZ, 'w', zipfile.ZIP_STORED) as out:
        for it in cat['items']:
            z = Z.get(it['book']); fn = it['id'][len(it['book']) + 1:] + '.jpg'      # p0056_1.jpg
            src = f"{it['book']}/images/{fn}"
            if z is None or src not in names[it['book']]:
                # на случай другой нумерации вырезок: единственная вырезка той же страницы
                pg = fn.split('_')[0]; alt = [x for x in names.get(it['book'], ()) if x.startswith(f"{it['book']}/images/{pg}_")]
                if len(alt) == 1: src = alt[0]
                else: miss.append(it['id']); continue
            out.writestr(it['img'], Z[it['book']].read(src)); n += 1
        cat['items'] = [it for it in cat['items'] if it['id'] not in set(miss)]
        cat['groups'] = [g for g in cat.get('groups', []) if g['members'][0] not in set(miss)]
        out.writestr('catalog.json', json.dumps(cat, ensure_ascii=False))
    log(f'пакет: {n} фото, {os.path.getsize(BZ) / 1e6:.0f} МБ; без фото {len(miss)}', miss[:10])
    auth.authenticate_user()
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    svc = build('drive', 'v3')
    idf = f'{D}/rug_catalog/books_bundle_id.txt'; fid = open(idf).read().strip() if os.path.exists(idf) else ''
    media = MediaFileUpload(BZ, mimetype='application/zip', resumable=True, chunksize=50 * 1024 * 1024)
    if fid:
        try: svc.files().update(fileId=fid, media_body=media).execute()
        except Exception as e: print('старый файл не обновился, создаю новый:', e); fid = ''
    if not fid:
        folder = svc.files().list(q="name='rug_catalog' and mimeType='application/vnd.google-apps.folder' and trashed=false", fields='files(id)').execute()['files']
        req = svc.files().create(body={'name': 'books_bundle.zip', **({'parents': [folder[0]['id']]} if folder else {})}, media_body=media, fields='id')
        resp = None
        while resp is None:
            st, resp = req.next_chunk()
            if st: print(f'загрузка {int(st.progress() * 100)}%', flush=True)
        fid = resp['id']
        svc.permissions().create(fileId=fid, body={'type': 'anyone', 'role': 'reader'}).execute()
        open(idf, 'w').write(fid)
    print('\nПАКЕТ КНИГ ГОТОВ: ковров', len(cat['items']), '| двойников с сайтом музея', len(cat.get('groups', [])))
    print('id пакета для студии («⟳ из Диска»):', fid, '\n')

def ocr_page(args):
    pdf, i, dpi = args
    import fitz, pytesseract
    from PIL import Image
    doc = fitz.open(pdf); pg = doc[i]; pix = pg.get_pixmap(dpi=dpi)
    im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    txt = pytesseract.image_to_string(im, lang='aze+rus', config='--psm 3')
    return {'page': i + 1, 'text': txt}

if 'ocr' in STEPS:
    import fitz
    pdf = [p for p in glob.glob('/content/drive/MyDrive/**/*.pdf', recursive=True) if 'ensiklopediyasi_ii' in os.path.basename(p).lower()][0]
    dst = f'{BOOKS}/ensiklopediya_ii/ocr_pages.jsonl'; os.makedirs(os.path.dirname(dst), exist_ok=True)
    done = set()
    if os.path.exists(dst):
        for l in open(dst, encoding='utf-8'):
            try: done.add(json.loads(l)['page'])
            except Exception: pass
    N = fitz.open(pdf).page_count; todo = [i for i in range(N) if i + 1 not in done]
    log(f'OCR: страниц {N}, уже есть {len(done)}, осталось {len(todo)}')
    dpi = int(os.environ.get('OCR_DPI', '250')); W = int(os.environ.get('OCR_WORKERS', '2'))
    with ProcessPoolExecutor(W) as ex, open(dst, 'a', encoding='utf-8') as f:
        for k, r in enumerate(ex.map(ocr_page, [(pdf, i, dpi) for i in todo], chunksize=2), 1):
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
            if k % 20 == 0: log(f'  OCR {k}/{len(todo)}')
    log('OCR ГОТОВО:', dst)
