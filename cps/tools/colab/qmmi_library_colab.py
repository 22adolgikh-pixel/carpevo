# qmmi_library_colab.py — книги и каталоги о коврах из библиотеки Реестра нематериального наследия (intangible.az, раздел «Xalçaçılıq sənəti»)
# → Google Drive carpet-dna/sources/qmmi/<id>_<название>.pdf. Уже скачанные пропускаются; прерванные докачиваются заново.
# Запуск в Colab (GPU не нужен):
#   !rm -rf /content/carpevo && git clone --depth 1 -b cps-v5 https://github.com/22adolgikh-pixel/carpevo.git /content/carpevo
#   %run /content/carpevo/cps/tools/colab/qmmi_library_colab.py
# Порядок — сначала каталоги с фото ковров (их фото потом пойдут в каталог студии), потом история/текст.
import os, re, time, urllib.request
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/carpet-dna/sources/qmmi'; os.makedirs(OUT, exist_ok=True)
BOOKS = [  # id, название, МБ
    (3218, 'Qarabag xalcalari. Kataloq', 43.2),
    (3243, 'Baki, Sirvan, Quba xalcalari. Kataloq', 88.5),
    (3263, 'Azerbaycan xalcalari. Tebriz qrupu', 77.0),
    (3254, 'Azerbaycan xalcalari. Qarabag qrupu-1', 47.7),
    (3255, 'Azerbaycan xalcalari. Qarabag qrupu-2', 48.7),
    (3212, 'Azerbaycan xalcalari. Qarabag qrupu', 59.7),
    (9557, 'Azerbaycan xalcalari: Qarabag qrupu (2)', 66.9),
    (2866, 'Azerbaycan xalcalari, I cild', 60.3),
    (2867, 'Azerbaycan xalcalari, II cild', 60.7),
    (3244, 'Azerbaidzhanskie kovry. Shirvanskaya gruppa', 79.2),
    (3236, 'Azerbaycan xalcalari. Irevan qrupu', 53.3),
    (2854, 'Azerbaycan xalcalari: Irevan qrupu (2)', 34.8),
    (2915, 'Syuzhetnye kovry Azerbaidzhana (XIX - nachalo XX v.)', 92.7),
    (9491, 'Kavkazskie drakonovye kovry', 10.8),
    (9563, 'Qarabagin Qasimusagi xalcalarinin dizayn xususiyyetleri', 90.3),
    (9556, 'Qarabag xalcalari', 16.8),
    (3213, 'Naxcivan xalcacilari', 133.2),
    (2847, 'Kerimov. Azerbaycan xalcasi', 103.8),
    (2851, 'Azerbaycan xalca seneti: tarixi, mektebleri, texnologiyasi', 18.5),
    (2852, 'Azerbaycanin Irevan bolgesi: tarixi ve xalcalari', 14.7),
    (3219, 'Qarabag bolgesinin xalcaciliq tarixi', 83.4),
    (11620, 'Qarabagin elat dunyasi', 24.3),
    (2855, 'Abdullaeva. Kovrovoe iskusstvo Azerbaidzhana (fragment)', 0.7),
    (2886, 'Vidy azerbaidzhanskikh kovrov', 0.8),
    (2853, 'Azerbaycan xalcaciligi tarixine kulturoloji baxis', 0.3),
    (3183, 'Baki xalcalarinin yaradiciliq xususiyyetleri', 0.2),
    (11621, 'Toxuculuq memulatlari uzerindeki naxislar etnoqrafik menbe kimi', 0.4),
    (11633, 'Azerbaycan xalqinin meiset medeniyyeti', 0.5),
]
UA = {'User-Agent': 'Mozilla/5.0 (research; CarpetDNA catalogue)'}
ok, bad = 0, []
for i, (bid, name, mb) in enumerate(BOOKS):
    fn = f'{OUT}/{bid}_{re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:70]}.pdf'
    if os.path.exists(fn) and os.path.getsize(fn) > mb * 1e6 * 0.97: print('есть:', os.path.basename(fn)); ok += 1; continue
    url = f'https://intangible.az/front/az/downloadExampleLibrary/{bid}'
    t0 = time.time(); print(f'[{i + 1}/{len(BOOKS)}] {name} ({mb} МБ)…', flush=True)
    for att in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r, open(fn + '.part', 'wb') as f:
                n = 0
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); n += len(b)
            if n < mb * 1e6 * 0.97: raise IOError(f'получено {n / 1e6:.1f} МБ из {mb}')
            os.replace(fn + '.part', fn); ok += 1; print(f'   готово за {time.time() - t0:.0f} с'); break
        except Exception as e:
            print('   ошибка:', str(e)[:120]); time.sleep(10 * (att + 1))
    else:
        bad.append(name)
print('\nскачано:', ok, 'из', len(BOOKS))
if bad: print('не скачалось:', bad, '\nЕсли не скачалось ничего — сайт не пускает Colab; тогда откройте ссылки https://intangible.az/front/az/downloadExampleLibrary/<id> в Chrome и положите PDF в carpet-dna/sources/qmmi/ сами.')
print('Напишите Claude: «книги QMMI скачаны».')
