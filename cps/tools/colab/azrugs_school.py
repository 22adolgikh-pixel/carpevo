# azrugs_school.py — каталог и модель «фото → школа» по архиву azerbaijanrugs (Drive: carpet-dna/photos/azerbaijan_rugs_guide).
# Что делает:
#  1) обходит папки, к каждому фото берёт одноимённый .txt (описание) → таблица catalog.csv (папка, школа, век, музей, размер, текст);
#  2) школа берётся по приоритету: слово в имени файла → название папки → текст описания; при расхождении папка/файл/текст помечается;
#  3) считает признаки DINOv2 (кэш в features.npz — повторный запуск быстрый);
#  4) обучает логистическую регрессию; проверка ГРУППОВАЯ (почти одинаковые фото — один ковёр/копии — всегда в одной группе),
#     поэтому точность честнее, чем в первой модели;
#  5) пишет report.json, predictions.csv, disagreements.csv (где модель уверенно спорит с подписью) и model.npz (веса для CPS).
# Результат: Drive carpet-dna/photos/azrugs_school/. Время: 15–40 минут на T4 (читать тысячи фото с Drive небыстро).
import os, re, json, csv, collections
import numpy as np, torch
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/carpet-dna/photos/azerbaijan_rugs_guide'
OUT = '/content/drive/MyDrive/carpet-dna/photos/azrugs_school'; os.makedirs(OUT, exist_ok=True)
IMG = ('.jpg', '.jpeg', '.png', '.webp')
# ключевое слово (латиница, нижний регистр) → школа. Порядок важен: более конкретное раньше.
KEYS = [('karabagh', 'karabakh'), ('karabakh', 'karabakh'), ('gendje', 'ganja-gazakh'), ('ganja', 'ganja-gazakh'), ('genje', 'ganja-gazakh'),
        ('kazak', 'ganja-gazakh'), ('gazakh', 'ganja-gazakh'), ('borchaly', 'ganja-gazakh'), ('bordjalou', 'ganja-gazakh'),
        ('kuba', 'guba-shirvan'), ('quba', 'guba-shirvan'), ('guba', 'guba-shirvan'), ('shirvan', 'guba-shirvan'), ('zakatala', 'guba-shirvan'),
        ('baku', 'guba-shirvan'), ('derbend', 'guba-shirvan'), ('daghestan', 'guba-shirvan'), ('khyzy', 'guba-shirvan'), ('talish', 'guba-shirvan'),
        ('mughan', 'guba-shirvan'), ('tabriz', 'tabriz'), ('heriz', 'tabriz'), ('serapi', 'tabriz'), ('karaja', 'tabriz'), ('sarab', 'tabriz'),
        ('ardabil', 'tabriz'), ('bakhshaish', 'tabriz'), ('meshkin', 'tabriz'), ('shahsavan', 'tabriz'), ('zanjan', 'tabriz'), ('karadagh', 'tabriz'),
        ('arazbaran', 'tabriz'), ('nakhchivan', 'nakhchivan')]
# ВНИМАНИЕ: сопоставление «регион → школа» — рабочее допущение (Heriz/Serapi/Ardabil… → «табризская» группа по традиции; Mughan/Talish → Ширван-Куба).
# Точные названия регионов сохраняются в столбце region, школу можно перекроить в таблице без пересчёта признаков.


def school_of(s):
    s = s.lower()
    for k, v in KEYS:
        if k in s: return v, k
    return None, None


rows = []
for dp, dn, fn in os.walk(ROOT):
    names = set(fn)
    for f in fn:
        stem, ext = os.path.splitext(f)
        if ext.lower() not in IMG: continue
        if os.path.getsize(os.path.join(dp, f)) < 12000: continue    # значки, кнопки и миниатюры сайта
        txt = ''
        for cand in (stem + '.txt', stem + '.TXT'):
            if cand in names:
                try: txt = open(os.path.join(dp, cand), encoding='utf-8', errors='ignore').read().strip()
                except Exception: pass
                break
        folder = os.path.relpath(dp, ROOT)
        sf, kf = school_of(stem); sd, kd = school_of(folder); st, kt = school_of(txt[:200])
        label = sf or sd or st
        src = 'file' if sf else 'folder' if sd else 'text' if st else ''
        votes = {x for x in (sf, sd, st) if x}
        cent = re.search(r'(\d{1,2})(?:st|nd|rd|th)[ -]century', txt, re.I)
        size = re.search(r'(\d{2,3})\s*[x×]\s*(\d{2,3})\s*cm', txt)
        rows.append(dict(path=os.path.join(dp, f), folder=folder, stem=stem, label=label or '', label_from=src, region=kf or kd or kt or '',
                         conflict=int(len(votes) > 1), century=cent.group(1) if cent else '', size=f'{size.group(1)}x{size.group(2)}' if size else '',
                         has_text=int(bool(txt)), text=txt[:400].replace('\n', ' ')))
print('фото всего:', len(rows), '| с описанием:', sum(r['has_text'] for r in rows), '| со школой:', sum(1 for r in rows if r['label']),
      '| папка/файл/текст спорят:', sum(r['conflict'] for r in rows))
print('по школам:', dict(collections.Counter(r['label'] or '—' for r in rows)))
print('по папкам (топ-15):', collections.Counter(r['folder'] for r in rows).most_common(15))
with open(f'{OUT}/catalog.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- признаки DINOv2 ----
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(dev).eval()
mean = np.array([0.485, 0.456, 0.406], 'f'); std = np.array([0.229, 0.224, 0.225], 'f')
cache = {}
if os.path.exists(f'{OUT}/features.npz'):
    z = np.load(f'{OUT}/features.npz', allow_pickle=True); cache = dict(zip(z['paths'], z['feats']))


class DS(torch.utils.data.Dataset):
    def __init__(s, ps): s.ps = ps
    def __len__(s): return len(s.ps)
    def __getitem__(s, i):
        try:
            im = Image.open(s.ps[i]).convert('RGB'); im.draft('RGB', (448, 448)); im = im.resize((224, 224))
            a = (np.asarray(im, 'f') / 255 - mean) / std; return torch.from_numpy(a.transpose(2, 0, 1)), i, True
        except Exception: return torch.zeros(3, 224, 224), i, False


todo = [r['path'] for r in rows if r['path'] not in cache]
dl = torch.utils.data.DataLoader(DS(todo), batch_size=32, num_workers=8)
done = 0
for x, idx, ok in dl:
    with torch.no_grad(): f = model(x.to(dev)).cpu().numpy()
    for j, v, o in zip(idx.tolist(), f, ok.tolist()):
        if o: cache[todo[j]] = v / np.linalg.norm(v)
    done += len(idx)
    if done % 320 < 32: print('признаки', done, '/', len(todo), flush=True)
np.savez(f'{OUT}/features.npz', paths=np.array(list(cache)), feats=np.stack(list(cache.values())))
rows = [r for r in rows if r['path'] in cache]
X = np.stack([cache[r['path']] for r in rows])

# ---- группы: почти одинаковые фото (косинус ≥ 0.97) — одна группа ----
n = len(rows); par = list(range(n))
def fd(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
for i in range(0, n, 512):
    S = X[i:i + 512] @ X.T
    for a, b in zip(*np.nonzero(S >= 0.97)):
        if i + a < b: par[fd(i + a)] = fd(b)
groups = np.array([fd(i) for i in range(n)])
print('групп (после склейки почти одинаковых):', len(set(groups)), 'из', n)

# ---- обучение ----
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
lab = np.array([r['label'] for r in rows]); cnt = collections.Counter(lab)
keep = np.array([bool(l) and cnt[l] >= 25 and not r['conflict'] for l, r in zip(lab, rows)])    # чистые подписи: без спора папка/файл/текст
y = lab[keep]; Xk = X[keep]; gk = groups[keep]
clf = LogisticRegression(max_iter=4000, C=10, class_weight='balanced')
cv = cross_val_predict(clf, Xk, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), groups=gk)
labs = sorted(set(y))
base = max(collections.Counter(y).values()) / len(y)
rep = {'n_train': int(keep.sum()), 'classes': dict(collections.Counter(y)), 'cv_accuracy_grouped': float((cv == y).mean()), 'baseline_majority': base,
       'report': classification_report(y, cv, output_dict=True), 'confusion': {'labels': labs, 'matrix': confusion_matrix(y, cv, labels=labs).tolist()}}
json.dump(rep, open(f'{OUT}/report.json', 'w'), ensure_ascii=False, indent=1)
print('ТОЧНОСТЬ (5-fold, групповая):', round(rep['cv_accuracy_grouped'], 3), '| базовая (всё в самый частый класс):', round(base, 3), '| обучающих:', int(keep.sum()), rep['classes'])
for k in labs: print(f'  {k}: полнота {rep["report"][k]["recall"]:.2f}, точность {rep["report"][k]["precision"]:.2f}')
clf.fit(Xk, y)
pr = clf.predict_proba(X)
out = []
for r, pp in zip(rows, pr):
    k = int(pp.argmax()); out.append([r['path'].replace(ROOT + '/', ''), r['label'], r['label_from'], r['conflict'], clf.classes_[k], round(float(pp[k]), 3), r['text'][:80]])
hdr = ['path', 'label', 'label_from', 'conflict', 'model', 'p', 'text']
csv.writer(open(f'{OUT}/predictions.csv', 'w', newline='', encoding='utf-8')).writerows([hdr] + out)
dis = sorted([o for o in out if o[1] and o[1] != o[4] and o[5] >= 0.7], key=lambda o: -o[5])
csv.writer(open(f'{OUT}/disagreements.csv', 'w', newline='', encoding='utf-8')).writerows([hdr] + dis)
np.savez(f'{OUT}/model.npz', classes=clf.classes_, coef=clf.coef_, intercept=clf.intercept_)   # для CPS: p = softmax(X @ coef.T + intercept)
print('расхождений (p>=0.7):', len(dis), '| споров папка/файл/текст:', sum(r['conflict'] for r in rows))
print('Готово. Напишите Claude: «модель готова».')
