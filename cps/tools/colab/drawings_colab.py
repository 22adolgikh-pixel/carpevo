# drawings_colab.py — вырезать зарисовки узоров (перо, без сетки) со страниц книги в 300 dpi для CPS.
# Сделано для Gans-Ruedin «Caucasian Carpets» (1986): на страницах-комментариях рядом с текстом — рисунок фрагмента ковра.
#
# Запуск в Colab (после book_pages_colab.py, видеокарта не нужна):
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %env BOOK=Caucasian_carpets
#   %run /content/carpevo/cps/tools/colab/drawings_colab.py
#
# Как ищет рисунок: страница 300 dpi → тёмные пиксели → вычитаются прямоугольники слов из текстового слоя (pdftotext -bbox)
# → оставшиеся «чернила» склеиваются → крупные пятна = рисунки. Страницы — из site/model/sources/gans_ruedin_1986/plates.json
# (страницы-комментарии) + все страницы с текстовым слоем в разделе пластин (на случай рисунков на других страницах).
# Результат на Drive: carpet-dna/output/gans_ruedin_drawings/ — PNG рисунков, manifest.json, _contact_sheet.jpg (обзор),
# drawings_for_cps.zip (PNG + manifest) с доступом «у кого есть ссылка» — его id печатается в конце для импорта в CPS.
import os, sys, glob, json, re, subprocess, html

def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)

print('1/5 ставлю poppler …', flush=True)
sh('apt-get -qq update && apt-get -qq install -y poppler-utils > /dev/null')
import numpy as np, cv2
from google.colab import drive, auth
drive.mount('/content/drive')

book = os.environ.get('BOOK', 'Caucasian_carpets')
cands = [p for p in glob.glob('/content/drive/**/*.pdf', recursive=True) if book.lower() in os.path.basename(p).lower()]
if not cands: sys.exit('PDF не найден')
src = max(cands, key=os.path.getsize)
sh(f'cp "{src}" /content/book.pdf')
out = os.path.join(os.path.dirname(os.path.dirname(src)), 'output', 'gans_ruedin_drawings')
os.makedirs(out, exist_ok=True)
plates = json.load(open('/content/carpevo/site/model/sources/gans_ruedin_1986/plates.json'))['plates']
by_page = {p['pdf_page']: p for p in plates}
pages = sorted(set(by_page) | set(range(30, 366)))
DPI = 300; S = DPI / 72.0

print('2/5 текстовые рамки слов …', flush=True)
def word_boxes(i):
    x = sh(f'pdftotext -bbox -f {i} -l {i} /content/book.pdf -', check=False).stdout
    return [tuple(float(v) for v in m) for m in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">', x)]

print('3/5 ищу рисунки на', len(pages), 'страницах …', flush=True)
manifest = []
for n, i in enumerate(pages):
    pl = by_page.get(i)
    sh(f'pdftoppm -r {DPI} -gray -f {i} -l {i} -singlefile -png /content/book.pdf /content/pg', check=False)
    g = cv2.imread('/content/pg.png', cv2.IMREAD_GRAYSCALE)
    if g is None: continue
    H, W = g.shape
    ink = (g < 150).astype(np.uint8)
    if ink.mean() > 0.35: continue                                   # фото на всю страницу — пропуск
    words = word_boxes(i)
    if not words and not pl: continue
    mask = np.zeros_like(ink)
    for x0, y0, x1, y1 in words:                                     # текст убрать (с запасом)
        cv2.rectangle(mask, (int(x0 * S) - 6, int(y0 * S) - 6), (int(x1 * S) + 6, int(y1 * S) + 6), 1, -1)
    rest = ink * (1 - mask)
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    blob = cv2.dilate(rest, np.ones((25, 25), np.uint8))
    k, lab, st, _ = cv2.connectedComponentsWithStats(blob)
    found = 0
    for j in range(1, k):
        x, y, w, h, a = st[j]
        if w < 0.12 * W or h < 0.08 * H: continue                     # мелочь: сноски, буквицы, грязь
        sub = rest[y:y + h, x:x + w]
        if sub.mean() < 0.02: continue                               # почти пусто
        if w > 0.95 * W and h > 0.9 * H: continue
        m = 15
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
        crop = g[y0:y1, x0:x1].copy()                                # текст внутри рамки НЕ белим: OCR находит «слова» в самих рисунках
        mid = float(((crop > 70) & (crop < 190)).mean())             # доля полутонов: у пера мало, у фото много
        white = float((crop > 215).mean())                           # чистая бумага: у пера много
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(crop, (3, 3), 0), cv2.CV_32F))
        lt = crop > 190; tex = float(lap[lt].mean()) if lt.any() else 99.0   # «шум» в светлых местах: у фото выше
        kind = 'photo' if (mid > 0.35 or (white < 0.35 and tex > 4.5)) else 'drawing'
        found += 1
        fid = 'gr86_p%03d_d%d' % (i, found)
        cv2.imwrite(os.path.join(out, fid + '.png'), crop)
        manifest.append({'id': fid, 'file': fid + '.png', 'kind': kind, 'midtones': round(mid, 3), 'paper': round(white, 3), 'texture': round(tex, 2), 'pdf_page': i, 'printed_page': i - 4, 'bbox_px300': [int(x0), int(y0), int(x1), int(y1)],
                         'plate_id': pl and pl.get('id'), 'title': pl and (pl.get('title_raw') or pl.get('type_name')),
                         'group': pl and pl.get('group'), 'knots_10cm_length': pl and pl.get('knots_10cm_length'),
                         'knots_10cm_width': pl and pl.get('knots_10cm_width'), 'dimensions_cm': pl and pl.get('dimensions_cm')})
    if (n + 1) % 25 == 0: print(f'   {n + 1}/{len(pages)} страниц, рисунков {len(manifest)}', flush=True)

json.dump(manifest, open(os.path.join(out, 'manifest.json'), 'w'), ensure_ascii=False, indent=1)
print('4/5 обзорный лист и zip …', flush=True)
th = []
for e in manifest:
    im = cv2.imread(os.path.join(out, e['file']), cv2.IMREAD_GRAYSCALE)
    s = 220 / max(im.shape); im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
    t = np.full((250, 240), 255, np.uint8); t[:im.shape[0], :im.shape[1]] = im
    cv2.putText(t, e['id'][5:] + (' PHOTO' if e['kind'] == 'photo' else ''), (2, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1); th.append(t)
cols = 8
while len(th) % cols: th.append(np.full((250, 240), 255, np.uint8))
if th:
    sheet = np.vstack([np.hstack(th[r:r + cols]) for r in range(0, len(th), cols)])
    cv2.imwrite(os.path.join(out, '_contact_sheet.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 70])
dr = list(manifest)   # в архив идут все вырезки; помеченные как фото CPS откроет «на пересмотр» — автоматика ошибается в обе стороны
json.dump(dr, open(os.path.join(out, 'manifest_drawings.json'), 'w'), ensure_ascii=False, indent=1)
sh(f'cd "{out}" && rm -f drawings_for_cps.zip && cp manifest_drawings.json /content/manifest.json && zip -q -j drawings_for_cps.zip ' + ' '.join('"%s"' % os.path.join(out, e['file']) for e in dr) + ' /content/manifest.json')

print('5/5 открываю доступ к zip по ссылке (для импорта в CPS) …', flush=True)
fid = None
try:
    auth.authenticate_user()
    from googleapiclient.discovery import build
    svc = build('drive', 'v3')
    r = svc.files().list(q="name='drawings_for_cps.zip' and trashed=false", fields='files(id,modifiedTime)', orderBy='modifiedTime desc').execute()
    if r.get('files'):
        fid = r['files'][0]['id']
        svc.permissions().create(fileId=fid, body={'type': 'anyone', 'role': 'reader'}).execute()
except Exception as ex:
    print('   не получилось открыть доступ автоматически:', ex)
    print('   Откройте доступ вручную: Drive → carpet-dna/output/gans_ruedin_drawings/drawings_for_cps.zip → «Поделиться» → «Все, у кого есть ссылка».')
summary = {'drawings': len(dr), 'photos_skipped': len(manifest) - len(dr), 'pages_with_drawings': len({e['pdf_page'] for e in manifest}),
           'with_knot_density': sum(1 for e in manifest if e['knots_10cm_length'] and e['knots_10cm_width']), 'zip_drive_id': fid, 'out': out}
json.dump(summary, open(os.path.join(out, '_summary.json'), 'w'), ensure_ascii=False, indent=1)
print('ИТОГ', json.dumps(summary, ensure_ascii=False))
