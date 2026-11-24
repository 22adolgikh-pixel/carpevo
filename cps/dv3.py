"""v12.29: DINOv3-B на сервере — отпечатки ковров для модели кадра v2 и поиска двойников.

Веса — DATA/wcache/dinov3_b (safetensors из timm/vit_base_patch16_dinov3.lvd1689m, скачаны /api/open/wfetch).
Отпечаток ковра — (2, 1536) float16: строка 0 — всё фото, строка 1 — выпрямленный ковёр; в каждой [CLS | среднее патчей].
Картинка вписывается в квадрат 224 с серыми полями (124), как в ночной пробе 23–24.11.
"""
import os, threading
import numpy as np, cv2

NAME = "vit_base_patch16_dinov3.lvd1689m"
MU = np.array([.485, .456, .406], np.float32); SD = np.array([.229, .224, .225], np.float32)
_m = {"model": None, "err": None}; _lock = threading.Lock()


def weights(data):
    return os.path.join(data, "wcache", "dinov3_b")


def available(data):
    if not os.path.exists(weights(data)): return False
    try:
        import timm, torch  # noqa: F401
        return True
    except Exception:
        return False


def model(data):
    with _lock:
        if _m["model"] is None:
            import timm, torch
            from safetensors.torch import load_file
            torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
            m = timm.create_model(NAME, pretrained=False, num_classes=0, img_size=224, dynamic_img_size=True)
            m.load_state_dict(load_file(weights(data))); m.eval(); _m["model"] = m
        return _m["model"]


def square(a, S=224):
    h, w = a.shape[:2]; k = S / max(h, w)
    a = cv2.resize(a, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    c = np.full((S, S, 3), 124, np.uint8); y = (S - a.shape[0]) // 2; x = (S - a.shape[1]) // 2
    c[y:y + a.shape[0], x:x + a.shape[1]] = a
    return c


def _prep(ims):
    x = np.stack([(im[:, :, ::-1].astype(np.float32) / 255 - MU) / SD for im in ims]).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x, np.float32)


def embed(data, imgs):
    """imgs: список BGR → (n, 1536) float32: [CLS | среднее патчей]"""
    import torch
    m = model(data)
    with torch.no_grad():
        f = m.forward_features(torch.from_numpy(_prep([square(a) for a in imgs])))
    npre = getattr(m, "num_prefix_tokens", 1)
    return np.concatenate([f[:, 0].numpy(), f[:, npre:].mean(1).numpy()], 1)


def tokens(data, img, R=336):
    """плотные признаки выпрямленного ковра для проверки пары: (4 поворота, R/16, R/16, 768), нормированы"""
    import torch
    m = model(data); a = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
    ims = [np.ascontiguousarray(np.rot90(a, k)) for k in range(4)]
    with torch.no_grad():
        f = m.forward_features(torch.from_numpy(_prep(ims)))[:, m.num_prefix_tokens:].numpy()
    f /= np.linalg.norm(f, axis=-1, keepdims=True); g = R // 16
    return f.reshape(4, g, g, -1)


def nz(x):
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
