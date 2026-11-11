"""Проверка гипотезы: можно ли заменить плотное умножение слоя
отсечением по близости в 3D-каркасе из главных ортогональных направлений.

Считаем не «красоту метафоры», а ошибку при одинаковом числе умножений.
Положительный контроль — матрица, которая и правда локальна в 3D.
Отрицательный — плотный гауссов шум и крутой, но непространственный спектр.
Реальные матрицы — MLP GPT-2, слои 0 и 1, уже лежавшие в проекте.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DIM = 128
GAUSS_SYNTH = 8
GAUSS_REAL = 12
FRACS = (0.05, 0.10, 0.20)
PAIR_FRACS = (0.05, 0.10, 0.20)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def spectrum_report(S: np.ndarray) -> dict:
    energy = S.astype(np.float64) ** 2
    total = float(energy.sum()) + 1e-30
    cum = np.cumsum(energy) / total

    def at(p: float) -> int:
        return int(np.searchsorted(cum, p) + 1)

    def share(k: int) -> float:
        return float(cum[min(k, len(cum)) - 1])

    return {
        "rank": int(len(S)),
        "top3": share(3),
        "top16": share(16),
        "top64": share(64),
        "rank_10": at(0.10),
        "rank_50": at(0.50),
        "rank_90": at(0.90),
    }


def svd_full(W: np.ndarray):
    U, S, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    return U, S, Vt


def positions_from(U, S, Vt, k: int = 3):
    k = min(k, len(S))
    scale = S[:k]
    row = U[:, :k] * scale
    col = Vt[:k].T * scale
    return row, col


def captured(energy: np.ndarray, score: np.ndarray, frac: float, largest: bool) -> float:
    flat_e = np.ascontiguousarray(energy).ravel()
    flat_s = np.ascontiguousarray(score).ravel()
    n_keep = max(1, min(flat_e.size, int(round(frac * flat_e.size))))
    if largest:
        idx = np.argpartition(-flat_s, n_keep - 1)[:n_keep]
    else:
        idx = np.argpartition(flat_s, n_keep - 1)[:n_keep]
    return float(flat_e[idx].sum() / (flat_e.sum() + 1e-30))


def pair_distances(row_pos: np.ndarray, col_pos: np.ndarray) -> np.ndarray:
    r2 = np.sum(row_pos * row_pos, axis=1)[:, None]
    c2 = np.sum(col_pos * col_pos, axis=1)[None, :]
    return np.maximum(r2 + c2 - 2.0 * (row_pos @ col_pos.T), 0.0)


def column_norms(W: np.ndarray) -> np.ndarray:
    return np.linalg.norm(W, axis=0)


def topk_idx(score: np.ndarray, K: int, largest: bool) -> np.ndarray:
    K = max(1, min(K, score.size))
    if largest:
        return np.argpartition(-score, K - 1)[:K]
    return np.argpartition(score, K - 1)[:K]


def geom_distances(col_pos: np.ndarray, x: np.ndarray, mode: str) -> np.ndarray:
    weight = np.abs(x)
    weight = weight / (weight.sum() + 1e-12)
    if mode == "point":
        center = (weight[:, None] * col_pos).sum(axis=0)
        return np.linalg.norm(col_pos - center, axis=1)
    k_sup = max(4, int(round(0.05 * len(x))))
    k_sup = min(k_sup, len(x))
    support = np.argpartition(-weight, k_sup - 1)[:k_sup]
    diff = col_pos[:, None, :] - col_pos[support][None, :, :]
    dist = np.linalg.norm(diff, axis=-1).min(axis=1)
    # при равной дистанции оставляем более сильные координаты входа
    dist = dist - 1e-6 * (weight / (weight.max() + 1e-12))
    return dist


def multiframe_score(Vt: np.ndarray, S: np.ndarray, x: np.ndarray, n_frames: int = 4) -> np.ndarray:
    weight = np.abs(x)
    weight = weight / (weight.sum() + 1e-12)
    score = np.zeros(Vt.shape[1], dtype=np.float64)
    frames = min(n_frames, len(S) // 3)
    for frame in range(frames):
        sl = slice(3 * frame, 3 * frame + 3)
        pos = Vt[sl].T * S[sl]
        center = (weight[:, None] * pos).sum(axis=0)
        dist2 = np.sum((pos - center) ** 2, axis=1)
        energy = float(np.sum(S[sl] ** 2))
        score += energy / (dist2 + 1e-8)
    return score


def lowrank_apply(U, S, Vt, x, rank: int) -> np.ndarray:
    rank = max(1, min(rank, len(S)))
    coeff = Vt[:rank] @ x
    return U[:, :rank] @ (S[:rank] * coeff)


def lod_apply(W, U, S, Vt, x, idx: np.ndarray) -> np.ndarray:
    y = lowrank_apply(U, S, Vt, x, 3)
    base = (U[:, :3] * S[:3]) @ Vt[:3, idx]
    residual = W[:, idx] - base
    return y + residual @ x[idx]


def match_rank(budget: int, n_out: int, n_in: int, cap: int) -> int:
    rank = int(round(budget / max(1, n_in + n_out)))
    return max(1, min(cap, rank))


def orthonormal(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(n, max(k, 1))))
    return q[:, :k]


def grid_points(n: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.ceil(n ** (1.0 / 3.0)))
    axis = np.linspace(-1.0, 1.0, side)
    pts = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    rng.shuffle(pts)
    pts = np.asarray(pts[:n], dtype=np.float64)
    pts = pts + rng.normal(scale=0.01, size=pts.shape)
    return pts


def mat_gauss(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n, n)) / np.sqrt(n)


def mat_steep(n: int, rng: np.random.Generator) -> np.ndarray:
    left = orthonormal(n, n, rng)
    right = orthonormal(n, n, rng)
    singular = 1.0 / np.arange(1, n + 1) ** 0.85
    return (left * singular) @ right.T


def mat_sharp(n: int, rng: np.random.Generator, radius: float = 0.55):
    pts = grid_points(n, rng)
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    mask = dist <= radius
    degree = mask.sum(axis=1, keepdims=True)
    weights = rng.normal(size=(n, n)) / np.sqrt(degree + 1.0)
    return weights * mask, pts


def mat_smooth(n: int, rng: np.random.Generator, sigma: float = 0.55):
    pts = grid_points(n, rng)
    dist2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    return np.exp(-dist2 / (2.0 * sigma * sigma)), pts


def mat_mixed(n: int, rng: np.random.Generator):
    left = orthonormal(n, 16, rng)
    right = orthonormal(n, 16, rng)
    singular = np.linspace(3.0, 1.0, 16)
    global_part = (left * singular) @ right.T
    local_part, pts = mat_sharp(n, rng)
    global_part *= np.linalg.norm(local_part) / (np.linalg.norm(global_part) + 1e-12)
    return global_part + local_part, pts


def make_queries(rng, n_in, Vt, oracle_pos, n_gauss: int):
    queries = []
    for _ in range(n_gauss):
        vec = rng.normal(size=n_in)
        queries.append(("гаусс", vec / np.linalg.norm(vec)))
    spikes = np.zeros(n_in)
    hot = rng.choice(n_in, size=min(8, n_in), replace=False)
    spikes[hot] = rng.normal(size=hot.size)
    queries.append(("всплески", spikes / np.linalg.norm(spikes)))
    main_axis = Vt[0].copy()
    queries.append(("главная ось", main_axis / np.linalg.norm(main_axis)))
    tail = Vt[-1].copy()
    queries.append(("хвост", tail / np.linalg.norm(tail)))
    if oracle_pos is not None and len(oracle_pos) == n_in:
        pts = oracle_pos
        center = pts[int(rng.integers(0, len(pts)))]
        blob = np.exp(-np.sum((pts - center) ** 2, axis=1) / (2.0 * 0.20 ** 2))
        queries.append(("один сгусток", blob / np.linalg.norm(blob)))
        left = np.array([-0.85, -0.85, -0.85])
        right = np.array([0.85, 0.85, 0.85])
        two = np.exp(-np.sum((pts - left) ** 2, axis=1) / (2.0 * 0.18 ** 2))
        two += np.exp(-np.sum((pts - right) ** 2, axis=1) / (2.0 * 0.18 ** 2))
        queries.append(("два сгустка", two / np.linalg.norm(two)))
    return queries


def empty_bucket():
    return {}


def add_error(bucket, query_name, frac, method, err):
    slot = bucket.setdefault(query_name, {}).setdefault(str(frac), {}).setdefault(method, [])
    slot.append(err)


def mean_bucket(bucket):
    out = {}
    for query_name, fracs in bucket.items():
        out[query_name] = {}
        for frac, methods in fracs.items():
            out[query_name][frac] = {m: float(np.mean(v)) for m, v in methods.items()}
    return out


def evaluate(name: str, W: np.ndarray, oracle_pos, rng, n_gauss: int) -> dict:
    n_out, n_in = W.shape
    print(f"  матрица {name} {W.shape}", flush=True)
    U, S, Vt = svd_full(W)
    U_abs, S_abs, Vt_abs = svd_full(np.abs(W))
    row_s, col_s = positions_from(U, S, Vt)
    row_a, col_a = positions_from(U_abs, S_abs, Vt_abs)
    energy = W * W
    norms = column_norms(W)
    report = {
        "name": name,
        "shape": [int(n_out), int(n_in)],
        "spectrum_signed": spectrum_report(S),
        "spectrum_abs": spectrum_report(S_abs),
        "pair_energy": {},
        "ranks": {},
        "gemv": {},
    }
    for frac in PAIR_FRACS:
        item = {
            "случайный": frac,
            "по величине": captured(energy, energy, frac, largest=True),
            "svd": captured(energy, pair_distances(row_s, col_s), frac, largest=False),
            "svd_модуля": captured(energy, pair_distances(row_a, col_a), frac, largest=False),
        }
        if oracle_pos is not None and len(oracle_pos) == n_out == n_in:
            item["оракул_3d"] = captured(
                energy, pair_distances(oracle_pos, oracle_pos), frac, largest=False
            )
        report["pair_energy"][str(frac)] = item

    queries = make_queries(rng, n_in, Vt, oracle_pos, n_gauss)
    # контроль: полный набор столбцов обязан совпасть с плотным умножением
    probe = queries[0][1]
    exact = W @ probe
    if not np.allclose(exact, W[:, np.arange(n_in)] @ probe, rtol=1e-8, atol=1e-8):
        raise RuntimeError("контроль полного набора столбцов не сошёлся")
    if not np.allclose(exact, lod_apply(W, U, S, Vt, probe, np.arange(n_in)), rtol=1e-6, atol=1e-6):
        raise RuntimeError("контроль LOD на всех столбцах не сошёлся")

    bucket = empty_bucket()
    cap = len(S)
    for frac in FRACS:
        K = max(1, min(n_in, int(round(frac * n_in))))
        budget_cols = 3 * n_in + n_out * K
        budget_lod = 3 * n_in + 3 * n_out + n_out * K
        rank_cols = match_rank(budget_cols, n_out, n_in, cap)
        rank_lod = match_rank(budget_lod, n_out, n_in, cap)
        report["ranks"][str(frac)] = {
            "K": K,
            "rank_против_столбцов": rank_cols,
            "rank_против_lod": rank_lod,
            "умножения_столбцы": int(budget_cols),
            "умножения_lod": int(budget_lod),
            "умножения_плотные": int(n_out * n_in),
        }
        for query_name, vec in queries:
            truth = W @ vec
            idx_mag = topk_idx(np.abs(vec) * norms, K, largest=True)
            idx_svd = topk_idx(geom_distances(col_s, vec, "point"), K, largest=False)
            idx_abs = topk_idx(geom_distances(col_a, vec, "point"), K, largest=False)
            idx_sup = topk_idx(geom_distances(col_s, vec, "support"), K, largest=False)
            idx_frames = topk_idx(multiframe_score(Vt, S, vec), K, largest=True)
            idx_rand = rng.choice(n_in, size=K, replace=False)
            packed = {
                "величина": rel_l2(W[:, idx_mag] @ vec[idx_mag], truth),
                "случайные": rel_l2(W[:, idx_rand] @ vec[idx_rand], truth),
                "3d_точка": rel_l2(W[:, idx_svd] @ vec[idx_svd], truth),
                "3d_модуль": rel_l2(W[:, idx_abs] @ vec[idx_abs], truth),
                "3d_объём": rel_l2(W[:, idx_sup] @ vec[idx_sup], truth),
                "4_кадра": rel_l2(W[:, idx_frames] @ vec[idx_frames], truth),
                "низкий_ранг": rel_l2(lowrank_apply(U, S, Vt, vec, rank_cols), truth),
                "lod": rel_l2(lod_apply(W, U, S, Vt, vec, idx_svd), truth),
                "низкий_ранг_как_у_lod": rel_l2(lowrank_apply(U, S, Vt, vec, rank_lod), truth),
            }
            if oracle_pos is not None and len(oracle_pos) == n_in:
                idx_orc = topk_idx(geom_distances(oracle_pos, vec, "point"), K, largest=False)
                packed["оракул_3d"] = rel_l2(W[:, idx_orc] @ vec[idx_orc], truth)
            for method, err in packed.items():
                add_error(bucket, query_name, frac, method, err)
    report["gemv"] = mean_bucket(bucket)
    return report


def neighbor_recall(X: np.ndarray, k: int = 8, n_frames: int = 1, n_queries: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    vecs = X.astype(np.float64)
    vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    centered = vecs - vecs.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    dim = min(3 * n_frames, vt.shape[0])
    pos = centered @ vt[:dim].T
    queries = rng.choice(len(vecs), size=min(n_queries, len(vecs)), replace=False)
    recalls = []
    for q in queries:
        sim = vecs @ vecs[q]
        sim[q] = -np.inf
        true = set(np.argpartition(-sim, k)[:k].tolist())
        score = np.zeros(len(vecs))
        frames = min(n_frames, dim // 3)
        for frame in range(max(1, frames)):
            sl = slice(3 * frame, min(3 * frame + 3, dim))
            if sl.start >= sl.stop:
                break
            delta = pos[:, sl] - pos[q, sl]
            dist2 = np.sum(delta * delta, axis=1)
            energy = float(np.sum(singular[sl] ** 2)) + 1e-12
            score += energy / (dist2 + 1e-8)
        score[q] = -np.inf
        pred = set(np.argpartition(-score, k)[:k].tolist())
        recalls.append(len(true & pred) / k)
    chance = k / max(1, len(vecs) - 1)
    return {"recall": float(np.mean(recalls)), "случайный": float(chance), "кадры": n_frames}


def synthetic_clouds(rng: np.random.Generator):
    n, d = 80, 96
    gauss = rng.normal(size=(n, d))
    latent = rng.normal(size=(n, 3))
    manifold = latent @ rng.normal(size=(3, d))
    manifold += rng.normal(scale=0.01 * manifold.std(), size=manifold.shape)
    centers = rng.normal(size=(8, d))
    labels = rng.integers(0, 8, size=n)
    clusters = centers[labels] + rng.normal(scale=0.05, size=(n, d))
    return {"гаусс": gauss, "многообразие_3d": manifold, "8_кластеров": clusters}


def banded_batch(W: np.ndarray, X: np.ndarray, bandwidth: int) -> np.ndarray:
    _, dim = X.shape
    out = np.zeros((X.shape[0], dim), dtype=np.float64)
    base = W.astype(np.float64)
    data = X.astype(np.float64)
    for offset in range(-bandwidth, bandwidth + 1):
        diag = np.diagonal(base, offset=offset)
        if offset >= 0:
            out[:, : dim - offset] += data[:, offset:] * diag
        else:
            shift = -offset
            out[:, shift:] += data[:, : dim - shift] * diag
    return out


def self_check():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 6))
    band = 1
    mask = np.abs(np.subtract.outer(np.arange(6), np.arange(6))) <= band
    clipped = base * mask
    data = rng.normal(size=(4, 6))
    got = banded_batch(clipped, data, band)
    truth = data @ clipped.T
    if not np.allclose(got, truth):
        raise RuntimeError("контроль ленточного умножения не сошёлся")
    W = rng.normal(size=(5, 7))
    U, S, Vt = svd_full(W)
    vec = rng.normal(size=7)
    exact = W @ vec
    if not np.allclose(lowrank_apply(U, S, Vt, vec, len(S)), exact, rtol=1e-6, atol=1e-6):
        raise RuntimeError("контроль полного SVD не сошёлся")


def time_layouts(rng: np.random.Generator) -> dict:
    dim = 1536
    batch = 64
    print(f"  тайминг D={dim}, пакет={batch}", flush=True)
    pts = grid_points(dim, rng)
    dist = np.linalg.norm(pts[:, None, :].astype(np.float32) - pts[None, :, :].astype(np.float32), axis=-1)
    mask = dist <= np.float32(0.42)
    degree = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    weights = (rng.normal(size=(dim, dim)).astype(np.float32) / np.sqrt(degree)) * mask
    order = np.argsort(pts[:, 0] * 4 + pts[:, 1] * 2 + pts[:, 2])
    ordered = weights[np.ix_(order, order)]
    bandwidth = max(1, int(round(0.08 * (dim - 1))))
    data = rng.normal(size=(batch, dim)).astype(np.float32)

    def median_seconds(fn, repeats: int = 3) -> float:
        fn()
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return float(np.median(samples))

    dense_t = median_seconds(lambda: data @ ordered.T)
    band_t = median_seconds(lambda: banded_batch(ordered, data, bandwidth))
    dense_y = data @ ordered.T
    band_y = banded_batch(ordered, data, bandwidth)
    # ошибка ленты относительно ПОЛНОЙ локальной матрицы, не относительно ленты
    return {
        "D": dim,
        "пакет": batch,
        "полуширина_доли": 0.08,
        "умножения_плотные": int(batch * dim * dim),
        "умножения_лента": int(batch * dim * (2 * bandwidth + 1)),
        "сек_плотное": dense_t,
        "сек_лента": band_t,
        "ошибка_ленты": rel_l2(band_y, dense_y),
        "ненулевых_на_строку": float(mask.sum(axis=1).mean()),
    }


def shorten(report: dict) -> str:
    spec = report["spectrum_signed"]
    lines = [
        f"{report['name']}: форма {report['shape']}, "
        f"энергия top3={spec['top3']:.3f}, top16={spec['top16']:.3f}, "
        f"ранг на 50%={spec['rank_50']}, на 90%={spec['rank_90']}"
    ]
    pair = report["pair_energy"]["0.1"]
    pair_txt = ", ".join(f"{k}={v:.3f}" for k, v in pair.items())
    lines.append(f"  энергия пар в ближайших 10%: {pair_txt}")
    gauss = report["gemv"].get("гаусс", {}).get("0.1", {})
    if gauss:
        err_txt = ", ".join(f"{k}={v:.3f}" for k, v in gauss.items())
        lines.append(f"  ошибка GEMV, гаусс, бюджет 10%: {err_txt}")
    blob = report["gemv"].get("один сгусток", {}).get("0.1")
    if blob:
        err_txt = ", ".join(f"{k}={v:.3f}" for k, v in blob.items())
        lines.append(f"  ошибка GEMV, один сгусток, 10%: {err_txt}")
    two = report["gemv"].get("два сгустка", {}).get("0.1")
    if two:
        err_txt = ", ".join(f"{k}={v:.3f}" for k, v in two.items())
        lines.append(f"  ошибка GEMV, два сгустка, 10%: {err_txt}")
    return "\n".join(lines)


def main():
    self_check()
    rng = np.random.default_rng(7)
    cases = []
    synth = [
        ("гаусс", mat_gauss(DIM, rng), None, GAUSS_SYNTH),
        ("крутой спектр", mat_steep(DIM, rng), None, GAUSS_SYNTH),
    ]
    sharp, sharp_pts = mat_sharp(DIM, rng)
    smooth, smooth_pts = mat_smooth(DIM, rng)
    mixed, mixed_pts = mat_mixed(DIM, rng)
    synth += [
        ("резкая 3D-локальность", sharp, sharp_pts, GAUSS_SYNTH),
        ("гладкая 3D-локальность", smooth, smooth_pts, GAUSS_SYNTH),
        ("смесь: ранг 16 + резкая 3D", mixed, mixed_pts, GAUSS_SYNTH),
    ]
    for name, matrix, pts, n_gauss in synth:
        cases.append(evaluate(name, matrix, pts, rng, n_gauss))

    for name, filename in (
        ("gpt2 слой0 c_fc", "gpt2_h0_c_fc.npy"),
        ("gpt2 слой0 c_proj", "gpt2_h0_c_proj.npy"),
        ("gpt2 слой1 c_fc", "gpt2_h1_c_fc.npy"),
        ("gpt2 слой1 c_proj", "gpt2_h1_c_proj.npy"),
    ):
        matrix = np.load(ROOT / "веса" / filename).astype(np.float64)
        cases.append(evaluate(name, matrix, None, rng, GAUSS_REAL))

    print("  соседство и внимание", flush=True)
    clouds = synthetic_clouds(rng)
    attention = {}
    for name, cloud in clouds.items():
        attention[name] = {
            "1_кадр": neighbor_recall(cloud, n_frames=1, seed=1),
            "4_кадра": neighbor_recall(cloud, n_frames=4, seed=1),
        }
    row_sample = sharp[rng.choice(sharp.shape[0], size=80, replace=False)]
    gpt_rows = np.load(ROOT / "веса" / "gpt2_h0_c_fc.npy")
    gpt_rows = gpt_rows[rng.choice(gpt_rows.shape[0], size=120, replace=False)]
    rows = {
        "строки резкой 3D-матрицы": {
            "1_кадр": neighbor_recall(row_sample, n_frames=1, seed=2),
            "4_кадра": neighbor_recall(row_sample, n_frames=4, seed=2),
        },
        "строки gpt2 c_fc слой0": {
            "1_кадр": neighbor_recall(gpt_rows, n_frames=1, seed=3),
            "4_кадра": neighbor_recall(gpt_rows, n_frames=4, seed=3),
        },
    }
    timing = time_layouts(np.random.default_rng(11))
    payload = {
        "машина": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "что_сравниваем": (
            "Ошибка приближённого умножения y=Wx. "
            "3D-методы берут столбцы рядом с центром текущего входа "
            "в координатах трёх главных сингулярных векторов. "
            "Низкий ранг тратит столько же умножений на глобальные компоненты. "
            "Величина берёт столбцы с наибольшим |x_j| * нормой столбца."
        ),
        "матрицы": cases,
        "соседи": {"синтетика": attention, "строки": rows},
        "тайминг": timing,
    }
    dest = ROOT / "rezultat.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(shorten(c) for c in cases))
    print("соседи", json.dumps(payload["соседи"], ensure_ascii=False))
    print("тайминг", json.dumps(timing, ensure_ascii=False))
    print("записано", dest)


if __name__ == "__main__":
    main()
