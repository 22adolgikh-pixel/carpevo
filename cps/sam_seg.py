# sam_seg.py — v11.28: точный контур элемента (MobileSAM: «Segment Anything», облегчённый энкодер TinyViT).
#
# Человек кликает по элементу (или машина даёт рамку находки) → SAM выделяет форму целиком: маска → контур → рамка
# по контуру. SAM возвращает 3 варианта (часть / элемент / элемент с окружением) — выбираем тот, что лучше подходит
# под подсказку; в студии можно переключать (1 / 2 / 3).
#
# Модель: vendor/mobile_sam (код MobileSAM и Segment Anything, Apache-2.0, без timm/torchvision); веса mobile_sam.pt
# (40 МБ) скачиваются один раз в data/models/ (проверяется sha256). Нужен torch (есть в контейнере студии).
# Энкодер — ≈1–2 с на фото на процессоре; признаки фото кэшируются (несколько кликов по одному фото — быстро).
import os, threading, hashlib, urllib.request, time
import numpy as np, cv2

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt"
SHA = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
_m = {"net": None, "torch": None, "err": ""}
_lock = threading.Lock()
_emb = {}                 # rid → (эмбеддинг, размер на входе, исходный размер), последние 6
_cfg = {"dir": "models"}


def configure(data_dir): _cfg["dir"] = os.path.join(data_dir, "models")


def available():
    if os.environ.get("CPS_SAM", "1") == "0": return False      # выключено (тесты)
    try:
        import torch  # noqa
        return True
    except Exception: return False


def _weights():
    os.makedirs(_cfg["dir"], exist_ok=True); p = os.path.join(_cfg["dir"], "mobile_sam.pt")
    if os.path.exists(p) and os.path.getsize(p) > 30_000_000: return p
    tmp = p + ".part"
    with urllib.request.urlopen(URL, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    h = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    if h != SHA: os.remove(tmp); raise RuntimeError("веса SAM скачались с другой контрольной суммой")
    os.replace(tmp, p); return p


def _load():
    if _m["net"] is not None: return _m["net"]
    import torch
    from vendor.mobile_sam.sam import Sam
    from vendor.mobile_sam.tiny_vit_sam import TinyViT
    from vendor.mobile_sam.prompt_encoder import PromptEncoder
    from vendor.mobile_sam.mask_decoder import MaskDecoder
    from vendor.mobile_sam.transformer import TwoWayTransformer
    net = Sam(image_encoder=TinyViT(img_size=1024, in_chans=3, num_classes=1000, embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
                                    num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7], mlp_ratio=4., drop_rate=0., drop_path_rate=0.0,
                                    use_checkpoint=False, mbconv_expand_ratio=4.0, local_conv_size=3, layer_lr_decay=0.8),
              prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16),
              mask_decoder=MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                                       transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256),
              pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375])
    net.load_state_dict(torch.load(_weights(), map_location="cpu", weights_only=True)); net.eval()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    _m.update(net=net, torch=torch)
    return net


def _embed(key, bgr):
    if key in _emb: return _emb[key]
    torch = _m["torch"]; net = _m["net"]
    H, W = bgr.shape[:2]; s = 1024.0 / max(H, W); nh, nw = int(round(H * s)), int(round(W * s))
    rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    x = (x - net.pixel_mean) / net.pixel_std
    x = torch.nn.functional.pad(x, (0, 1024 - nw, 0, 1024 - nh))
    with torch.no_grad(): e = net.image_encoder(x)
    if len(_emb) >= 6: _emb.pop(next(iter(_emb)))
    _emb[key] = (e, (nh, nw), (H, W))
    return _emb[key]


def segment(key, bgr, points=None, box=None):
    """points — [(x, y, 1|0)] в долях фото (1 — элемент, 0 — не элемент); box — [x, y, w, h] в долях.
    → [{poly: [[x,y],…] доли, box: [x,y,w,h], area: доля фото, score}] — 3 варианта, от меньшего к большему"""
    with _lock:
        net = _load(); torch = _m["torch"]
        e, (nh, nw), (H, W) = _embed(key, bgr)
        pts = lbl = bx = None
        if points:
            pts = torch.tensor([[[p[0] * nw, p[1] * nh] for p in points]], dtype=torch.float32)
            lbl = torch.tensor([[int(p[2]) for p in points]], dtype=torch.int32)
        if box:
            bx = torch.tensor([[box[0] * nw, box[1] * nh, (box[0] + box[2]) * nw, (box[1] + box[3]) * nh]], dtype=torch.float32)
        with torch.no_grad():
            sp, de = net.prompt_encoder(points=(pts, lbl) if pts is not None else None, boxes=bx, masks=None)
            lo, iou = net.mask_decoder(image_embeddings=e, image_pe=net.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sp,
                                       dense_prompt_embeddings=de, multimask_output=True)
            m = torch.nn.functional.interpolate(lo, (1024, 1024), mode="bilinear", align_corners=False)[..., :nh, :nw]
            m = (m[0] > net.mask_threshold).cpu().numpy(); sc = iou[0].cpu().numpy()
    out = []
    for i in range(m.shape[0]):
        mk = cv2.morphologyEx(m[i].astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk)
        if n < 2: continue
        # одна связная часть: под точкой (если есть), иначе самая крупная
        j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if points:
            px, py = int(points[0][0] * nw), int(points[0][1] * nh)
            if 0 <= py < nh and 0 <= px < nw and lab[py, px] > 0: j = lab[py, px]
        mk = (lab == j).astype(np.uint8)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: continue
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, max(1.0, 0.004 * cv2.arcLength(c, True)), True).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(c)
        out.append({"poly": [[round(float(px_) / nw, 5), round(float(py_) / nh, 5)] for px_, py_ in c],
                    "box": [round(x / nw, 5), round(y / nh, 5), round(w / nw, 5), round(h / nh, 5)],
                    "area": round(float(mk.sum()) / (nw * nh), 5), "score": round(float(sc[i]), 3)})
    out.sort(key=lambda o: o["area"])
    return out


def refine_box(key, bgr, box, grow=1.6, max_k=1.8):
    """рамка находки (часто обрезанная, сдвинутая или с запасом) → контур элемента целиком.
    Три подсказки SAM: (1) рамка ×grow + точка в центре, (2) сама рамка, (3) только точка в центре — 9 вариантов.
    Отбрасываются: упёршиеся в расширенную рамку со всех сторон (это фон/поле), крупнее 1/5 фото, меньше 1/4 рамки.
    Из оставшихся — самый уверенный (score). Проверка на фото: обрезанная рамка — совпадение с элементом 0.31 → 0.87,
    сдвинутая — 0.80 → 0.91, с запасом — 0.41 → 0.88 (против одной подсказки). → вариант или None"""
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    e = [max(0.0, cx - box[2] * grow / 2), max(0.0, cy - box[3] * grow / 2), 0, 0]
    e[2] = min(1.0, cx + box[2] * grow / 2) - e[0]; e[3] = min(1.0, cy + box[3] * grow / 2) - e[1]

    def edges(b, tol=0.01):
        return (b[0] <= e[0] + tol * e[2]) + (b[1] <= e[1] + tol * e[3]) + (b[0] + b[2] >= e[0] + e[2] * (1 - tol)) + (b[1] + b[3] >= e[1] + e[3] * (1 - tol))
    cands = segment(key, bgr, points=[(cx, cy, 1)], box=e) + segment(key, bgr, box=box) + segment(key, bgr, points=[(cx, cy, 1)])
    # v11.29: на плотном поле (пиребедиль) SAM охотно сливает соседние фигуры в одну — результат не должен уходить далеко
    # от рамки: стороны не больше max_k и не меньше 1/2 исходных, центр — в пределах исходной рамки
    def near(b):
        bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
        return (box[2] / 2 <= b[2] <= box[2] * max_k and box[3] / 2 <= b[3] <= box[3] * max_k
                and abs(bx - cx) <= box[2] / 2 and abs(by - cy) <= box[3] / 2)
    ok = [c for c in cands if edges(c["box"]) < 4 and c["area"] < 0.2 and near(c["box"])]
    return max(ok, key=lambda c: c["score"]) if ok else None


def best_for_box(cands, box):
    """из вариантов SAM — тот, чья рамка ближе всего к подсказке (перекрытие), при равенстве — с большим score"""
    def iou(a, b):
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        i = ix * iy; u = a[2] * a[3] + b[2] * b[3] - i
        return i / u if u > 0 else 0
    return max(cands, key=lambda c: iou(c["box"], box) + 0.2 * c["score"]) if cands else None
