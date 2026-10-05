# cnn.py — свёрточная сеть клеток (v10.15). Обучена в Colab (tools/colab/cnn_colab.py), веса — models/cnn_net.pt
# (или data/cnn_net.pt, если положить свою). Замер на листах, которых сеть не видела (839 принятых ЧБ):
# ошибка по закрашенным клеткам: пороги 22,5% → маленькая модель 16,8% → сеть 9,0%.
# torch нужен только здесь и подключается лениво: нет torch или весов — available() = False, студия работает как раньше.
import os, threading
import numpy as np, cv2

S, PAD = 12, 2                      # пикселей на клетку при подаче в сеть; поле в клетках (как при обучении)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CPS_DATA") or os.path.join(HERE, "data")
_lock = threading.Lock(); _net = None; _tried = False


def _weights_path():
    for p in (os.path.join(DATA, "cnn_net.pt"), os.path.join(HERE, "models", "cnn_net.pt")):
        if os.path.exists(p): return p
    return None


def _build(torch):
    nn, F = torch.nn, torch.nn.functional

    class Net(nn.Module):             # та же архитектура, что в tools/colab/cnn_colab.py
        def __init__(s, c=32):
            super().__init__()
            s.pix = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, c, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.ReLU(), nn.AvgPool2d(3))
            s.ctx = nn.ModuleList([nn.Conv2d(2 * c, 2 * c, 3, padding=r, dilation=r) for r in (1, 2, 4, 8, 1)])
            s.bn = nn.ModuleList([nn.BatchNorm2d(2 * c) for _ in s.ctx])
            s.head = nn.Conv2d(2 * c, 3, 1)

        def forward(s, x):
            h = s.pix(x)
            for cv, bn in zip(s.ctx, s.bn): h = h + F.relu(bn(cv(h)))
            return s.head(h)
    return Net()


def _load():
    global _net, _tried
    with _lock:
        if _tried: return _net
        _tried = True
        p = _weights_path()
        if not p: return None
        try:
            import torch
            torch.set_num_threads(int(os.environ.get("CPS_TORCH_THREADS", "2")))
            net = _build(torch); net.load_state_dict(torch.load(p, map_location="cpu")); net.eval(); _net = net
        except Exception as e:
            print("cnn: недоступна:", e)
        return _net


def available():
    return _load() is not None


def predict(g, quad, grid, cols, rows):
    """g — серый кроп, quad — 4 угла блока клеток в кропе, grid {pw,ph,ox,oy} в выпрямленных координатах quad.
    → метки клеток (rows, cols) uint8: 0 фон, 1 контур, 2 серое. None, если сеть недоступна."""
    net = _load()
    if net is None: return None
    import torch
    import autogrid
    q = np.float32(quad); W, H = autogrid.quad_size([list(p) for p in q])
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), q)
    ox, oy, pw, ph = grid.get("ox", 0), grid.get("oy", 0), grid["pw"], grid["ph"]
    T = np.float64([[pw / S, 0, ox - PAD * pw], [0, ph / S, oy - PAD * ph], [0, 0, 1]])
    img = cv2.warpPerspective(g, Hm.astype(np.float64) @ T, ((cols + 2 * PAD) * S, (rows + 2 * PAD) * S),
                              flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    p90 = float(max(np.percentile(img.astype(np.uint8), 90), 1))
    x = torch.from_numpy(img.astype(np.uint8).astype(np.float32) / p90)[None, None]
    with torch.no_grad():
        lo = net(x)[0]
    pr = torch.softmax(lo, 0)[:, PAD:PAD + rows, PAD:PAD + cols].numpy()
    return pr.argmax(0).astype(np.uint8)


def predict_work(g, w):
    """Метки по итоговой сетке рабочего файла, в координатах матрицы (h, w) — как её видит человек."""
    gr, m = w["grid"], w["matrix"]; x0, y0 = m.get("origin", [0, 0])
    cols = max(int(gr.get("cols") or 0), x0 + int(m["w"])); rows = max(int(gr.get("rows") or 0), y0 + int(m["h"]))
    P = predict(g, w["quad"], gr, cols, rows)
    return None if P is None else P[y0:y0 + int(m["h"]), x0:x0 + int(m["w"])]
