# azrugs_zip_colab.py — архивы для экрана «Ковры» в студии из папки Drive carpet-dna/photos/azerbaijan_rugs_guide.
# Фото уменьшаются до 1400 px, рядом кладётся одноимённый .txt (описание), папки сохраняются (по ним студия предлагает школу).
# Архивы по ≤ 90 МБ (ограничение загрузки на сервере 100 МБ): carpet-dna/photos/azrugs_zip/part_NN.zip.
# %env AZ_ONLY_TEXT=1 — только фото, у которых есть описание (их ~450; начать лучше с них).
import os, io, zipfile, cv2
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/carpet-dna/photos/azerbaijan_rugs_guide'
OUT = '/content/drive/MyDrive/carpet-dna/photos/azrugs_zip'; os.makedirs(OUT, exist_ok=True)
ONLY = os.environ.get('AZ_ONLY_TEXT') == '1'
LIM = 90 * 1024 * 1024
part, size, z, n, seen = 0, 0, None, 0, set()
def newzip():
    global part, size, z
    if z: z.close()
    part += 1; size = 0; z = zipfile.ZipFile(f'{OUT}/part_{part:02d}.zip', 'w', zipfile.ZIP_STORED)
newzip()
for dp, dn, fn in sorted(os.walk(ROOT)):
    names = set(fn)
    for f in sorted(fn):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in ('.jpg', '.jpeg', '.png', '.webp'): continue
        p = os.path.join(dp, f)
        if os.path.getsize(p) < 12000: continue
        txt = next((os.path.join(dp, c) for c in (stem + '.txt', stem + '.TXT') if c in names), None)
        if ONLY and not txt: continue
        if stem in seen: continue                       # одно и то же фото в нескольких папках сайта
        seen.add(stem)
        a = cv2.imread(p)
        if a is None: continue
        s = 1400 / max(a.shape[:2])
        if s < 1: a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        b = cv2.imencode('.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        rel = os.path.relpath(dp, ROOT).split('/')[0]
        if size + len(b) > LIM: newzip()
        z.writestr(f'{rel}/{stem}.jpg', b); size += len(b)
        if txt: z.writestr(f'{rel}/{stem}.txt', open(txt, encoding='utf-8', errors='ignore').read())
        n += 1
        if n % 200 == 0: print(n, 'фото, архив', part, flush=True)
z.close()
print(f'ИТОГ: {n} фото в {part} архив(ах): {OUT}/part_NN.zip — загрузите их в студии: «3 · Ковры» → «🗜 архив»')
