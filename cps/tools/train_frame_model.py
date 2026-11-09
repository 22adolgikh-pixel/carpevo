"""Обучить «целый ли ковёр» по ответам людей на /m (v11.44).

    python3 cps/tools/train_frame_model.py <папка data/rugs> [--out cps/models/frame_v1.json]

Ответы — r.micro.frame (голоса) или r.frame при frame_by=micro|hand. Цель: целый (whole) против остального
(обрезан / фрагмент / не ковёр). Модель — градиентный бустинг из неглубоких деревьев; качество — 5×5 перекрёстная проверка.
Деревья выгружаются в JSON и считаются на сервере без sklearn (cps/frame_model.py); сверка предсказаний — в конце.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import frame_model as FM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def label(r):
    if r.get("frame_by") in ("micro", "hand") and r.get("frame"): return r["frame"]
    v = list(((r.get("micro") or {}).get("frame") or {}).values())
    if not v: return None
    c = {}
    for x in v: c[x["a"]] = c.get(x["a"], 0) + x.get("w", 1)
    return max(c, key=c.get)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rugs"); ap.add_argument("--out", default=FM.PATH); a = ap.parse_args()
    X, Y, R = [], [], []
    for f in sorted(os.listdir(a.rugs)):
        if not f.endswith(".json"): continue
        try: r = json.load(open(os.path.join(a.rugs, f), encoding="utf-8"))
        except Exception: continue
        y = label(r)
        if y and r.get("auto"): X.append(FM.feat(r)); Y.append(y == "whole"); R.append(r)
    X, Y = np.array(X), np.array(Y)
    print(f"примеров {len(Y)}, целых {int(Y.sum())}")
    if len(Y) < 40: sys.exit("мало ответов — нужно хотя бы 40")
    mk = lambda s: GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=s)
    P = np.mean([cross_val_predict(mk(s), X, Y, cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1] for s in range(5)], 0)
    acc = float(((P > 0.5) == Y).mean()); rule = float(((X[:, FM.NAMES.index("auto_whole")] > 0.5) == Y).mean())
    thr = {}
    for t in (0.5, 0.6, 0.7, 0.8):
        s = P >= t; thr[str(t)] = {"picked": int(s.sum()), "precision": round(float(Y[s].mean()), 3) if s.any() else None, "missed_whole": int((Y & ~s).sum())}
    print(f"CV точность {acc:.3f} (правило машины {rule:.3f}, «всегда целый» {Y.mean():.3f})"); print(thr)
    m = mk(0).fit(X, Y)
    trees = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        trees.append({"l": t.children_left.tolist(), "r": t.children_right.tolist(), "f": t.feature.tolist(),
                      "th": [round(float(v), 6) for v in t.threshold], "v": [round(float(v[0][0]), 6) for v in t.value]})
    p0 = float(np.clip(Y.mean(), 1e-6, 1 - 1e-6))
    out = {"ver": 1, "trained": time.strftime("%Y-%m-%d %H:%M"), "n": int(len(Y)), "n_whole": int(Y.sum()), "cv_acc": round(acc, 3), "rule_acc": round(rule, 3),
           "thr": thr, "names": FM.NAMES, "init": float(np.log(p0 / (1 - p0))), "lr": m.learning_rate, "trees": trees}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"))
    FM._M["m"] = None; FM.PATH = a.out
    mine = np.array([FM.p_whole(r) for r in R]); ref = m.predict_proba(X)[:, 1]
    d = float(np.abs(mine - ref).max()); print(f"сверка с sklearn: макс. расхождение {d:.2e}")
    if d > 1e-4: sys.exit("JSON-модель считает не так, как sklearn")
    print("записано:", a.out)


if __name__ == "__main__":
    main()
