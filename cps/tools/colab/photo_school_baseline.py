# photo_school_baseline.py — первая модель «фото → школа» (признаки DINOv2 + логистическая регрессия).
# Учится на фото с высокой уверенностью (ковёр целиком), проверяет себя перекрёстно (5 частей),
# затем для ВСЕХ фото выдаёт предсказание и список расхождений с ИИ-разметкой (подсказка, где подпись, возможно, неверна).
# Результат: Drive carpet-dna/photos/school_baseline/{report.json, predictions.csv, disagreements.csv}
# Время: ~5-10 минут на GPU (Среда выполнения → T4), ~20 на CPU.
import os, json, re, hashlib, csv
import numpy as np, torch, requests
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/carpet-dna/photos'
OUT = f'{ROOT}/school_baseline'; os.makedirs(OUT, exist_ok=True)
items = requests.get('https://raw.githubusercontent.com/22adolgikh-pixel/carpevo/cps-v5/site/model/sources/museum_photos/catalog_photos.json').json()['items']
def fname(pid): return re.sub(r'[^A-Za-z0-9._-]+', '_', pid)[:80] + '_' + hashlib.md5(pid.encode()).hexdigest()[:6] + '.jpg'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(dev).eval()
mean = np.array([0.485, 0.456, 0.406], 'f'); std = np.array([0.229, 0.224, 0.225], 'f')
def prep(path):
    im = Image.open(path).convert('RGB').resize((224, 224))
    a = (np.asarray(im, 'f') / 255 - mean) / std
    return torch.from_numpy(a.transpose(2, 0, 1))
feats = {}
ids = [p['id'] for p in items if os.path.exists(f'{ROOT}/full/{fname(p["id"])}')]
for i in range(0, len(ids), 32):
    b = ids[i:i + 32]
    x = torch.stack([prep(f'{ROOT}/full/{fname(j)}') for j in b]).to(dev)
    with torch.no_grad(): f = model(x).cpu().numpy()
    for j, v in zip(b, f): feats[j] = v / np.linalg.norm(v)
    print(i + len(b), '/', len(ids), flush=True)
P = {p['id']: p for p in items}
train = [j for j in feats if P[j]['conf'] == 'high' and P[j]['type'] and P[j]['whole'] and P[j]['kind'] in ('pile', 'flatweave')]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
X = np.stack([feats[j] for j in train]); y = np.array([P[j]['type'] for j in train])
clf = LogisticRegression(max_iter=3000, C=10, class_weight='balanced')
cv = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
labs = sorted(set(y))
rep = {'n_train': len(train), 'classes': {k: int((y == k).sum()) for k in labs}, 'cv_accuracy': float((cv == y).mean()),
       'report': classification_report(y, cv, output_dict=True), 'confusion': {'labels': labs, 'matrix': confusion_matrix(y, cv, labels=labs).tolist()},
       'note': 'перекрёстная проверка может быть завышена: несколько фото одного ковра могут попасть и в обучение, и в проверку'}
json.dump(rep, open(f'{OUT}/report.json', 'w'), ensure_ascii=False, indent=1)
print('ТОЧНОСТЬ (5-fold):', round(rep['cv_accuracy'], 3), '| обучающих:', len(train), rep['classes'])
clf.fit(X, y)
allid = list(feats); pr = clf.predict_proba(np.stack([feats[j] for j in allid]))
rows = []
for j, pp in zip(allid, pr):
    k = int(pp.argmax()); rows.append([j, P[j]['title'][:60], P[j]['conf'], P[j]['whole'], P[j]['type'] or '', clf.classes_[k], round(float(pp[k]), 3), j in train])
csv.writer(open(f'{OUT}/predictions.csv', 'w', newline='')).writerows([['id', 'title', 'conf', 'whole', 'ai_label', 'model', 'p', 'in_train']] + rows)
dis = [r for r in rows if r[4] and r[4] != r[5] and r[6] >= 0.6]
dis.sort(key=lambda r: -r[6])
csv.writer(open(f'{OUT}/disagreements.csv', 'w', newline='')).writerows([['id', 'title', 'conf', 'whole', 'ai_label', 'model', 'p', 'in_train']] + dis)
print('расхождений с разметкой (p>=0.6):', len(dis), '| из них не в обучении:', sum(1 for r in dis if not r[7]))
print('Готово. Напишите Claude: «модель готова».')
