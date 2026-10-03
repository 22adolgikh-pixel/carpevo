"""Возврат потерянной серой заливки в уже ПРИНЯТЫХ ч/б рисунках.
Причина: раньше прогон автомата шёл без серой заливки (dark_only), а такие рисунки принимали с белым телом. Проверка на 621 принятом
рисунке: у 44 из 138 «без серого» автомат находит заливку в >10% клеток, на оригинале она там действительно есть (смотрели глазами).
Контур и сетка человека не трогаются: серое берётся из автомата только в клетки, которые у человека пустые; рисунок уходит
обратно в проверку (done/reviewed=false, review=true), прежняя матрица сохраняется в matrix_before_gray."""
import numpy as np
import cv2
import autogrid
import learn

MIN_SHARE = 0.10      # доля пустых клеток, которые автомат считает серыми


def _enc(U):
    return ["".join("." if v == 0 else str(int(v)) for v in row) for row in U]


def propose(work, g):
    """-> (U_new, share) или None. U_new — матрица человека + серое автомата в его пустых клетках."""
    U = learn.matrix_labels(work)
    if (U == 2).any(): return None
    Cu, _ = learn.final_centers(work)
    G, C, M, info = autogrid.auto_figure(g, work.get("frame") or [0, 0, g.shape[1], g.shape[0]], dark_only=False, use_model=False)
    Hi = cv2.getPerspectiveTransform(np.array(G["corners"], np.float32), np.array([[0, 0], [G["cols"], 0], [G["cols"], G["rows"]], [0, G["rows"]]], np.float32))
    uv = cv2.perspectiveTransform(Cu.reshape(-1, 1, 2).astype(np.float32), Hi).reshape(Cu.shape[:2] + (2,))
    ii = np.floor(uv[..., 0]).astype(int); jj = np.floor(uv[..., 1]).astype(int)
    ok = (ii >= 0) & (ii < M.shape[1]) & (jj >= 0) & (jj < M.shape[0])
    A = np.zeros(U.shape, np.uint8); A[ok] = M[jj[ok], ii[ok]]
    add = (A == 2) & (U == 0)
    share = float(add.sum()) / max(1, int((U == 0).sum()))
    if share < MIN_SHARE: return None
    N = U.copy(); N[add] = 2
    return N, share


def apply(work, N, share):
    work["matrix_before_gray"] = work["matrix"]["rows"]
    work["matrix"]["rows"] = _enc(N)
    pal = list(work.get("palette") or ["#ffffff", "#1a1a1a"])
    if len(pal) < 3:
        pal = (pal + ["#ffffff", "#1a1a1a"])[:2] + ["#b8b8b8"]; work["palette"] = pal
        names = list(work.get("palette_names") or ["фон", "обводка"]); work["palette_names"] = (names + ["фон", "обводка"])[:2] + ["тело 1"]
    a = work["auto"] if isinstance(work.get("auto"), dict) else {}
    a["reviewed"] = False; a["gray_restored"] = round(share, 3)
    a["flags"] = [f for f in (a.get("flags") or []) if "серая заливка возвращена" not in f] + ["серая заливка возвращена автоматом — проверьте"]
    work["auto"] = a; work["done"] = False; work["review"] = True
    return work
