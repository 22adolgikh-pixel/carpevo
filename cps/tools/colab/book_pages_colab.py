# book_pages_colab.py — разложить книгу (PDF на Google Drive) на страницы: текст + картинка страницы.
# Результат кладётся рядом на Drive, откуда Claude читает страницы через подключённый Google Drive.
#
# Запуск в Colab (видеокарта не нужна):
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %env BOOK=Caucasian_carpets
#   %run /content/carpevo/cps/tools/colab/book_pages_colab.py
#
# BOOK — часть имени PDF (ищется на всём Drive). Необязательно: %env OCR_LANG=eng+deu+fra (по умолчанию eng+deu+fra+rus),
# %env OUT_NAME=raw_pages_ruedin (имя папки результата; по умолчанию raw_pages_<имя файла>).
# Повторный запуск продолжает с места остановки (готовые страницы пропускаются).
import os, sys, glob, json, re, subprocess, time

def sh(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)

print('1/4 ставлю poppler и tesseract …', flush=True)
langs = os.environ.get('OCR_LANG', 'eng+deu+fra+rus')
pk = ' '.join('tesseract-ocr-' + {'eng': 'eng', 'deu': 'deu', 'fra': 'fra', 'rus': 'rus', 'aze': 'aze', 'aze_cyrl': 'aze-cyrl'}.get(l, l) for l in langs.split('+'))
sh(f'apt-get -qq update && apt-get -qq install -y poppler-utils tesseract-ocr {pk} > /dev/null')

from google.colab import drive
drive.mount('/content/drive')

book = os.environ.get('BOOK', '').strip()
if not book: sys.exit('Укажите часть имени PDF: %env BOOK=...')
print('2/4 ищу PDF с «%s» в имени …' % book, flush=True)
cands = [p for p in glob.glob('/content/drive/**/*.pdf', recursive=True) if book.lower() in os.path.basename(p).lower()]
if not cands: sys.exit('PDF не найден на Drive. Проверьте имя.')
src = max(cands, key=os.path.getsize)
print('   нашёл:', src, round(os.path.getsize(src) / 1e6, 1), 'МБ')
local = '/content/book.pdf'
sh(f'cp "{src}" {local}')
info = sh(f'pdfinfo {local}').stdout
pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
name = os.environ.get('OUT_NAME') or 'raw_pages_' + re.sub(r'[^A-Za-z0-9]+', '_', os.path.splitext(os.path.basename(src))[0])[:40].strip('_').lower()
out = os.path.join(os.path.dirname(os.path.dirname(src)), 'output', name)   # carpet-dna/output/<name>
os.makedirs(out, exist_ok=True)
print('   страниц:', pages, '→', out)

print('3/4 страницы: текстовый слой или OCR + превью 110 dpi …', flush=True)
t0 = time.time(); stats = {'text_layer': 0, 'ocr': 0, 'empty': 0}
for i in range(1, pages + 1):
    jp = os.path.join(out, 'page_%03d.json' % i)
    if os.path.exists(jp):
        stats[json.load(open(jp)).get('method', 'ocr')] = stats.get(json.load(open(jp)).get('method', 'ocr'), 0) + 1
        continue
    txt = sh(f'pdftotext -layout -f {i} -l {i} {local} -', check=False).stdout
    method = 'text_layer'
    if len(re.sub(r'\s', '', txt)) < 40:
        sh(f'pdftoppm -r 300 -gray -f {i} -l {i} -png {local} /content/p', check=False)
        png = sorted(glob.glob('/content/p*.png'))
        txt = ''
        if png:
            txt = sh(f'tesseract {png[0]} - -l {langs} --psm 3', check=False).stdout
            for f in png: os.remove(f)
        method = 'ocr' if len(re.sub(r'\s', '', txt)) >= 10 else 'empty'
    sh(f'pdftoppm -r 110 -jpeg -jpegopt quality=80 -f {i} -l {i} -singlefile {local} {os.path.join(out, "page_%03d" % i)}', check=False)
    json.dump({'page': i, 'method': method, 'chars': len(txt), 'text': txt, 'image': 'page_%03d.jpg' % i},
              open(jp, 'w'), ensure_ascii=False)
    stats[method] += 1
    if i % 10 == 0 or i == pages:
        el = time.time() - t0
        print(f'   {i}/{pages}  ({el / 60:.1f} мин, осталось ≈{el / i * (pages - i) / 60:.0f} мин)', flush=True)

print('4/4 итог', flush=True)
summary = {'source_pdf': src, 'pages': pages, 'ocr_lang': langs, **stats, 'out': out}
json.dump(summary, open(os.path.join(out, '_summary.json'), 'w'), ensure_ascii=False, indent=1)
print('ИТОГ', json.dumps(summary, ensure_ascii=False))
print('Готово. Напишите Claude: «страницы книги готовы» — он прочитает их с Drive.')
