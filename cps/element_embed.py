# element_embed.py — «отпечатки» элементов узора: сеть, обученная на ответах людей в /m (v11.58).
#
# Обучение — tools/colab/element_search_colab.py (метки — tools/elements/make_labels.py). Сеть превращает вырезку
# элемента (рамка + 15 % поля, 96×96) в вектор из 128 чисел; близость векторов = «тот же элемент». Образец берётся
# во всех 8 положениях (поворот на 90° и зеркало) — у ковра элемент бывает повёрнут.
# Применение в студии:
#   verify — близость предложенных рамок к образцу группы: в карточке «элементы» лучшие идут первыми, сомнительные
#            (ниже порога thr_bad) заранее отмечены ✗;
#   search — по образцу найти на ковре остальные повторы (окна размера образца, шаг 1/3, три масштаба и поворот на 90°).
# Веса — DATA или models/element_embed_v1.pt; если нет — скачиваются с Диска (id в DRIVE). Нет torch или весов —
# available() = False, всё работает как раньше.
import json, os, threading
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CPS_DATA") or os.path.join(HERE, "data")
# v11.61: две сети, у каждой своя роль (замеры 13–14.11 на ответах людей):
#   "match"  — DINOv2 ViT-S/14 без обучения: сравнение вырезок (сортировка в «элементах», пары «между коврами»);
#              AUC «верная / отклонённая рамка» 0,733 против 0,704 у оценки машины; у элемента с именем тот же элемент
#              среди 5 ближайших на других коврах — в 56 % случаев;
#   "search" — своя сеть (обучена на 35 коврах): поиск повторов по образцу, порог 0,875.
# Роль можно переназначить файлом DATA/element_model.json {"match": "...", "search": "..."}.
MODELS = {"dinov3_elem": {"pt": "1O7MGlYcgyGDhAIiqGNZvaQHJjccjmVu1", "json": "1hMGl3iZXI483CRBcJfl_2JUV9IlSlKxx", "ver": 202,
                     # v11.64: вырезка ОВАЛОМ (вне овала серый) + 15 % поля — соседи в плотных узорах не мешают: AUC 0,785→0,956, any5 0,72→0,84;
                     # «заранее ✗»: при 5 % потерь верных ловит 75 % неверных (порог 0,844; замер 14.11 на 755 рамках, dino3_shape_colab.py)
                     "meta": {"S": 112, "ctx": 0.15, "shape": "ellipse", "thr_pre": 0.844}},
          "dinov2_s14": {"pt": "1ztqy9QC8c5RfN_Fi92H9nN3GA1JVwU_9", "json": "1Tdd1UjunQlB6fTKkFNV7xirg2OibwP6c", "ver": 101},
          "element_embed_v1": {"pt": "1M_5g-sChZHlGhzNYyiuiL3xzq81ybX-V", "json": "11_NpAjRhHnZSYBgln7JZxetOebGKWQ0J", "ver": 1}}
ROLE = {"match": "dinov3_elem", "search": "element_embed_v1"}
try:
    ROLE.update({k: v for k, v in json.load(open(os.path.join(DATA, "element_model.json"), encoding="utf-8")).items() if v in MODELS})
except (OSError, ValueError): pass
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
_lock = threading.Lock(); _ST = {n: {"net": None, "tried": False, "meta": {}, "err": ""} for n in MODELS}


def _n(m): return ROLE.get(m, m) if m else ROLE["match"]


def version(m="match"): return MODELS[_n(m)]["ver"]


VERSION = version("search")


def _path(n):
    for p in (os.path.join(DATA, n + ".pt"), os.path.join(HERE, "models", n + ".pt")):
        if os.path.exists(p): return p
    return None


def fetch(m="match", force=False):
    from rugs import drive_fetch
    n = _n(m); D = MODELS[n]; S = _ST[n]; dst = os.path.join(DATA, n + ".pt")
    if os.path.exists(dst) and not force: return True
    try:
        drive_fetch(D["pt"], dst + ".part"); os.replace(dst + ".part", dst)
        try: drive_fetch(D["json"], dst[:-3] + ".json")
        except Exception: pass
        S.update(tried=False, net=None, err=""); return True
    except Exception as e:
        S["err"] = str(e)[:200]; return False


def _load(m="match"):
    n = _n(m); S = _ST[n]
    if S["tried"]: return S["net"]
    with _lock:
        if S["tried"]: return S["net"]
        S["tried"] = True
        try:
            import torch
            P = _path(n)
            if P is None and fetch(n): P = _path(n)
            if P:
                torch.set_num_threads(2)
                S["net"] = torch.jit.load(P, map_location="cpu").eval()
                try: S["meta"] = json.load(open(P[:-3] + ".json", encoding="utf-8"))
                except (OSError, ValueError): S["meta"] = {}
                S["meta"].update(MODELS[n].get("meta") or {})                   # v11.64: параметры вырезки, замеренные позже выгрузки весов
        except Exception as e:
            S["net"] = None; S["err"] = S["err"] or str(e)[:200]
    return S["net"]


def available(m="match"):
    """готова ли сеть; первый вызов не ждёт загрузки (88 МБ у DINOv2) — качает в фоне, а пока отвечает «нет»"""
    S = _ST[_n(m)]
    if S["tried"]: return S["net"] is not None
    if not S.get("warming"):
        S["warming"] = True; threading.Thread(target=_load, args=(m,), daemon=True, name="element-embed-load").start()
    return False


def meta(m="match"): _load(m); return _ST[_n(m)]["meta"] or {}


def info():
    out = {"ok": any(_load(r) is not None for r in ROLE), "roles": dict(ROLE)}
    for r in ROLE:
        n = _n(r); S = _ST[n]; M = S["meta"] or {}
        out[r] = {"model": n, "ok": S["net"] is not None, "error": S["err"], **{k: M.get(k) for k in ("kind", "S", "trained", "thr_search", "shape", "thr_pre")}}
    return out


def small(img, m="match"):
    L = int(meta(m).get("long", 1400)); k = L / max(img.shape[:2])
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img


_CL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def _crop(a, box, S, ctx, gray=False, shape="rect"):
    if gray:
        c = _crop(a, box, S, ctx, False, shape); g = _CL.apply(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)); return cv2.merge([g, g, g])
    h, w = a.shape[:2]; x, y, bw, bh = box
    cx, cy = (x + bw / 2) * w, (y + bh / 2) * h; side = max(bw * w, bh * h) * (1 + 2 * ctx)
    M = np.array([[side / S, 0, cx - side / 2], [0, side / S, cy - side / 2]], np.float32)
    c = cv2.warpAffine(a, M, (S, S), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    if shape == "ellipse":                                                      # v11.64: овал по рамке, снаружи серое — чужие соседи не попадают в отпечаток
        m = np.zeros((S, S), np.uint8)
        cv2.ellipse(m, (S // 2, S // 2), (max(2, int(round(bw * w / side * S / 2))), max(2, int(round(bh * h / side * S / 2)))), 0, 0, 360, 255, -1)
        c = np.where(m[:, :, None] > 0, c, np.uint8(127))
    return c


def _prep(c): return ((np.ascontiguousarray(c)[:, :, ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def _run(arr, m):
    import torch
    net = _load(m); out = []
    with torch.no_grad():
        for i in range(0, len(arr), 128): out.append(net(torch.from_numpy(np.ascontiguousarray(np.stack(arr[i:i + 128])))).numpy())
    return np.concatenate(out) if out else np.zeros((0, 128), np.float32)


def emb(a, boxes, m="match"):
    M = meta(m); S, ctx = int(M.get("S", 96)), float(M.get("ctx", 0.15))
    return _run([_prep(_crop(a, b, S, ctx, bool(M.get("gray")), M.get("shape", "rect"))) for b in boxes], m)


def emb_rot(a, box, m="match"):
    M = meta(m); S, ctx = int(M.get("S", 96)), float(M.get("ctx", 0.15)); c = _crop(a, box, S, ctx, bool(M.get("gray")), M.get("shape", "rect"))
    return _run([_prep(np.rot90(c[:, ::-1] if f else c, r)) for f in (0, 1) for r in range(4)], m)


def verify(img, sample_box, boxes):
    """близость рамок к образцу (−1..1, выше — похожее); img — уже уменьшенное small()"""
    if _load("match") is None or not boxes: return None
    q = emb_rot(img, sample_box, "match"); return [float(v) for v in (emb(img, boxes, "match") @ q.T).max(1)]


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def search(img, sample_box, exclude=(), quad=None, thr=None, limit=24):
    """найти на ковре повторы образца → [(близость, рамка)], без наложений и без уже отмеченных мест"""
    m = "search"
    if _load(m) is None: return []
    if thr is None and meta(m).get("thr_search") is None: return []
    thr = thr if thr is not None else float(meta(m)["thr_search"])
    h, w = img.shape[:2]; bw, bh = sample_box[2], sample_box[3]; q = emb_rot(img, sample_box, m); C = []
    X0, X1, Y0, Y1 = 0.0, 1.0, 0.0, 1.0
    if quad:
        X0, X1 = max(0.0, min(p[0] for p in quad)), min(1.0, max(p[0] for p in quad)); Y0, Y1 = max(0.0, min(p[1] for p in quad)), min(1.0, max(p[1] for p in quad))
    for sc in (0.85, 1.0, 1.18):
        for ww, hh in {(bw * sc, bh * sc), (bh * sc * h / w, bw * sc * w / h)}:
            xs = np.arange(X0, X1 - ww, ww / 3); ys = np.arange(Y0, Y1 - hh, hh / 3)
            bs = [[float(x), float(y), ww, hh] for y in ys for x in xs]
            if not bs or len(bs) > 20000: continue
            s = (emb(img, bs, m) @ q.T).max(1)
            C += [(float(v), b) for v, b in zip(s, bs) if v >= thr]
    C.sort(key=lambda t: -t[0]); keep = []
    ex = list(exclude) + [sample_box]
    for v, b in C:
        if any(_iou(b, e) >= 0.3 for e in ex) or any(_iou(b, k[1]) >= 0.3 for k in keep): continue
        keep.append((v, [round(x, 5) for x in b]))
        if len(keep) >= limit: break
    return keep
