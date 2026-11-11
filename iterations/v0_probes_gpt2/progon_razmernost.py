"""Какая размерность каркаса отделяет существенные связи от пустых.

Для каждой матрицы берём d главных сингулярных осей и смотрим:
энергию ближайших 10% пар, различимость близкого и далёкого,
ошибку умножения при отборе 10% столбцов по расстоянию.
Рядом — стоимость самого расстояния относительно плотного умножения.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import progon

ROOT = Path(__file__).resolve().parent
DIMS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
FRAC = 0.10


def share_at(S: np.ndarray, k: int) -> float:
    energy = S.astype(np.float64) ** 2
    k = min(k, len(S))
    return float(energy[:k].sum() / (energy.sum() + 1e-30))


def distance_cv(pos: np.ndarray, rng: np.random.Generator, n: int = 4000) -> float:
    count = len(pos)
    if count < 3:
        return 0.0
    left = rng.integers(0, count, n)
    right = rng.integers(0, count, n)
    mask = left != right
    dist = np.linalg.norm(pos[left[mask]] - pos[right[mask]], axis=1)
    return float(dist.std() / (dist.mean() + 1e-12))


def gemv_error(W, col_pos, vectors) -> float:
    n_in = W.shape[1]
    k = max(1, min(n_in, int(round(FRAC * n_in))))
    errors = []
    for vec in vectors:
        truth = W @ vec
        dist = progon.geom_distances(col_pos, vec, "point")
        idx = progon.topk_idx(dist, k, largest=False)
        errors.append(progon.rel_l2(W[:, idx] @ vec[idx], truth))
    return float(np.mean(errors))


def recall_at_dim(X: np.ndarray, dim: int, k: int = 8, n_queries: int = 24, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    vecs = X.astype(np.float64)
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    centered = vecs - vecs.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    dim = min(dim, vt.shape[0])
    pos = centered @ vt[:dim].T
    queries = rng.choice(len(vecs), size=min(n_queries, len(vecs)), replace=False)
    hits = []
    for q in queries:
        sim = vecs @ vecs[q]
        sim[q] = -np.inf
        true = set(np.argpartition(-sim, k)[:k].tolist())
        delta = pos - pos[q]
        dist2 = np.sum(delta * delta, axis=1)
        dist2[q] = np.inf
        pred = set(np.argpartition(dist2, k)[:k].tolist())
        hits.append(len(true & pred) / k)
    return float(np.mean(hits))


def sweep_matrix(name: str, W: np.ndarray, vectors: dict, rng, oracle_pos=None) -> dict:
    print(f"  {name} {W.shape}", flush=True)
    U, S, Vt = progon.svd_full(W)
    energy = W * W
    n_out, n_in = W.shape
    dense = int(n_out * n_in)
    rows = []
    for d in DIMS:
        if d > len(S):
            break
        row_pos, col_pos = progon.positions_from(U, S, Vt, d)
        item = {
            "d": d,
            "доля_спектра": share_at(S, d),
            "энергия_ближних_пар": progon.captured(
                energy, progon.pair_distances(row_pos, col_pos), FRAC, largest=False
            ),
            "разброс_расстояний": distance_cv(col_pos, rng),
            "ошибка": {key: gemv_error(W, col_pos, vecs) for key, vecs in vectors.items()},
            "доля_стоимости_расстояния": float(d * n_in / dense),
        }
        rows.append(item)
    report = {
        "name": name,
        "shape": [int(n_out), int(n_in)],
        "случайные_пары": FRAC,
        "кривая": rows,
    }
    if oracle_pos is not None and len(oracle_pos) == n_in == n_out:
        oracle_rows = []
        for d in (1, 2, 3):
            pos = oracle_pos[:, :d]
            oracle_rows.append(
                {
                    "d": d,
                    "энергия_ближних_пар": progon.captured(
                        energy, progon.pair_distances(pos, pos), FRAC, largest=False
                    ),
                    "ошибка_сгусток": gemv_error(W, pos, vectors["один сгусток"])
                    if "один сгусток" in vectors
                    else None,
                }
            )
        report["оракул"] = oracle_rows
    best = min(rows, key=lambda item: item["ошибка"]["гаусс"])
    report["лучшая_d_по_гауссу"] = best["d"]
    report["ошибка_лучшей"] = best["ошибка"]["гаусс"]
    return report


def gauss_vectors(n_in: int, rng: np.random.Generator, count: int):
    vecs = rng.normal(size=(count, n_in))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return list(vecs)


def blob_vectors(pts: np.ndarray):
    center = pts[len(pts) // 3]
    one = np.exp(-np.sum((pts - center) ** 2, axis=1) / (2 * 0.20**2))
    one = one / np.linalg.norm(one)
    left = np.array([-0.85, -0.85, -0.85])
    right = np.array([0.85, 0.85, 0.85])
    two = np.exp(-np.sum((pts - left) ** 2, axis=1) / (2 * 0.18**2))
    two += np.exp(-np.sum((pts - right) ** 2, axis=1) / (2 * 0.18**2))
    two = two / np.linalg.norm(two)
    return [one], [two]


def main():
    rng = np.random.default_rng(7)
    reports = []
    sharp, sharp_pts = progon.mat_sharp(progon.DIM, rng)
    smooth, smooth_pts = progon.mat_smooth(progon.DIM, rng)
    mixed, mixed_pts = progon.mat_mixed(progon.DIM, rng)
    synthetic = [
        ("гаусс", progon.mat_gauss(progon.DIM, rng), None),
        ("крутой спектр", progon.mat_steep(progon.DIM, rng), None),
        ("резкая 3D", sharp, sharp_pts),
        ("гладкая 3D", smooth, smooth_pts),
        ("смесь", mixed, mixed_pts),
    ]
    for name, matrix, pts in synthetic:
        vecs = {"гаусс": gauss_vectors(matrix.shape[1], rng, 8)}
        if pts is not None:
            one, two = blob_vectors(pts)
            vecs["один сгусток"] = one
            vecs["два сгустка"] = two
        reports.append(sweep_matrix(name, matrix, vecs, rng, pts))

    for name, filename in (
        ("gpt2 слой0 c_fc", "gpt2_h0_c_fc.npy"),
        ("gpt2 слой0 c_proj", "gpt2_h0_c_proj.npy"),
        ("gpt2 слой1 c_fc", "gpt2_h1_c_fc.npy"),
        ("gpt2 слой1 c_proj", "gpt2_h1_c_proj.npy"),
    ):
        matrix = np.load(ROOT / "веса" / filename).astype(np.float64)
        vecs = {"гаусс": gauss_vectors(matrix.shape[1], rng, 8)}
        reports.append(sweep_matrix(name, matrix, vecs, rng))

    print("  соседство по размерности", flush=True)
    clouds = progon.synthetic_clouds(rng)
    gpt_rows = np.load(ROOT / "веса" / "gpt2_h0_c_fc.npy")
    gpt_rows = gpt_rows[rng.choice(gpt_rows.shape[0], size=120, replace=False)]
    clouds["строки gpt2 c_fc"] = gpt_rows
    neighbors = {}
    for name, cloud in clouds.items():
        chance = 8 / (len(cloud) - 1)
        neighbors[name] = {
            "случайный": chance,
            "кривая": [
                {"d": d, "полнота": recall_at_dim(cloud, d, seed=4)}
                for d in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
                if d < cloud.shape[1]
            ],
        }

    payload = {"кривые": reports, "соседи": neighbors}
    dest = ROOT / "rezultat_razmernost.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports:
        print(
            f"\n{report['name']}: лучшая d по гауссу = {report['лучшая_d_по_гауссу']}, "
            f"ошибка {report['ошибка_лучшей']:.3f}"
        )
        for row in report["кривая"]:
            if row["d"] not in (1, 2, 3, 4, 8, 16, 32, 64):
                continue
            err = ", ".join(f"{k}={v:.3f}" for k, v in row["ошибка"].items())
            print(
                f"  d={row['d']:2d} спектр={row['доля_спектра']:.3f} "
                f"ближние={row['энергия_ближних_пар']:.3f} "
                f"разброс={row['разброс_расстояний']:.3f} "
                f"цена={row['доля_стоимости_расстояния']:.4f} {err}"
            )
        if "оракул" in report:
            print("  оракул", report["оракул"])
    for name, item in neighbors.items():
        curve = ", ".join(f"{r['d']}:{r['полнота']:.2f}" for r in item["кривая"])
        print(f"соседи {name} (случайный {item['случайный']:.3f}): {curve}")
    print("записано", dest)


if __name__ == "__main__":
    main()
