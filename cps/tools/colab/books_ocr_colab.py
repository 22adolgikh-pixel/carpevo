# -*- coding: utf-8 -*-
"""books_ocr_colab.py — полный OCR страниц книг-сканов без текстового слоя (CarpetsCat, 2026-11-12).

book_extract_colab.py для сканов без текстового слоя распознаёт только подписи к фото;
text.jsonl остаётся пустым. Этот скрипт распознаёт ВСЕ страницы и пишет
carpet-dna/output/books/<slug>/ocr_pages.jsonl ({"page": N, "text": "..."}),
с продолжением после обрыва (уже готовые страницы пропускаются).

Переменные окружения: SLUGS (через запятую, по умолч. четыре скана), OCR_DPI (230), OCR_WORKERS (2).
Запуск: через exec/%run после drive.mount (см. блок в чате).
"""
import os, glob, json, time
from concurrent.futures import ProcessPoolExecutor

D = '/content/drive/MyDrive/carpet-dna'
BOOKS = f'{D}/output/books'
SLUGS = os.environ.get('SLUGS', 'intangible_2851,intangible_2854,intangible_2850,intangible_2849').split(',')
DPI = int(os.environ.get('OCR_DPI', '230'))
W = int(os.environ.get('OCR_WORKERS', '2'))
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:5.0f} c]', *a, flush=True)


def ocr_page(args):
    pdf, i, dpi = args
    import fitz, pytesseract
    from PIL import Image
    doc = fitz.open(pdf)
    pix = doc[i].get_pixmap(dpi=dpi)
    im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    return {'page': i + 1, 'text': pytesseract.image_to_string(im, lang='aze+rus+eng', config='--psm 3')}


def find_pdf(slug):
    for p in glob.glob('/content/drive/MyDrive/scans/**/*.pdf', recursive=True):
        if slug.lower() in os.path.basename(p).lower():
            return p
    raise SystemExit(f'PDF для {slug} не найден')


if __name__ == '__main__':
    import fitz
    for slug in SLUGS:
        pdf = find_pdf(slug)
        dst = f'{BOOKS}/{slug}/ocr_pages.jsonl'
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        done = set()
        if os.path.exists(dst):
            for l in open(dst, encoding='utf-8'):
                try:
                    done.add(json.loads(l)['page'])
                except Exception:
                    pass
        N = fitz.open(pdf).page_count
        todo = [i for i in range(N) if i + 1 not in done]
        log(f'{slug}: страниц {N}, уже есть {len(done)}, осталось {len(todo)}')
        if not todo:
            continue
        with ProcessPoolExecutor(W) as ex, open(dst, 'a', encoding='utf-8') as f:
            for k, r in enumerate(ex.map(ocr_page, [(pdf, i, DPI) for i in todo], chunksize=2), 1):
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()
                if k % 25 == 0:
                    log(f'  {slug}: {k}/{len(todo)}')
        log(f'{slug}: ГОТОВО →', dst)
    print('\nOCR ВСЕХ КНИГ ГОТОВ')
