"""Полные наборы векторов. 3D только решает, кто столкнулся.

На столкновении перемножается весь внутренний вектор, без отрезания осей.
Смотрим, какая доля наборов обязана столкнуться, чтобы ответ слоя уцелел.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import progon

ROOT = Path(__file__).resolve().parent
FRACS = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00)


def column_scores(W: np.ndarray, U, S, Vt, vec: np.ndarray):
    norms = np.linalg.norm(W, axis=0)
    magnitude = np.abs(vec) * norms
    _, col = progon.positions_from(U, S, Vt, 3)
    distance = progon.geom_distances(col, vec, "point")
    order = np.arange(W.shape[1])
    return {
        "лучшие наборы": magnitude,
        "коллизия в 3D": -distance,
        "случайный порядок": order.astype(np.float64),
    }


def error_at(W, vec, score, frac, largest, rng):
    n_in = W.shape[1]
    k = max(1, min(n_in, int(round(frac * n_in))))
    if frac >= 1:
        idx = np.arange(n_in)
    elif largest:
        idx = np.argpartition(-score, k - 1)[:k]
    else:
        idx = rng.choice(n_in, size=k, replace=False)
    return progon.rel_l2(W[:, idx] @ vec[idx], W @ vec)


def main():
    rng = np.random.default_rng(9)
    W = np.load(ROOT / "веса" / "gpt2_h0_c_fc.npy").astype(np.float64)
    U, S, Vt = progon.svd_full(W)
    vecs = []
    for _ in range(8):
        v = rng.normal(size=W.shape[1])
        vecs.append(v / np.linalg.norm(v))
    methods = ("лучшие наборы", "коллизия в 3D", "случайный порядок")
    curve = {m: [] for m in methods}
    for frac in FRACS:
        bucket = {m: [] for m in methods}
        for vec in vecs:
            scores = column_scores(W, U, S, Vt, vec)
            for name, score in scores.items():
                largest = name != "случайный порядок"
                bucket[name].append(error_at(W, vec, score, frac, largest, rng))
        for name in methods:
            curve[name].append({"доля_столкновений": frac, "ошибка": float(np.mean(bucket[name]))})
            print(f"{name} {frac:.2f} {curve[name][-1]['ошибка']:.3f}", flush=True)
    dest = ROOT / "rezultat_kollizia.json"
    dest.write_text(json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
