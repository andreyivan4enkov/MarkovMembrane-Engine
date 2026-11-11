"""Навязанное 3D-пространство для уже обученного слоя.

Оси не ищутся в весах. Их задают снаружи: случайное облако,
решётка по номеру нейрона или случайная проекция.
Дальше гасятся все пары, которые в этом объёме не столкнулись.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import progon

ROOT = Path(__file__).resolve().parent
FRAC = 0.10


def grid_positions(count: int) -> np.ndarray:
    side = int(np.ceil(count ** (1.0 / 3.0)))
    axis = np.linspace(-1.0, 1.0, side)
    pts = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    return np.asarray(pts[:count], dtype=np.float64)


def random_positions(count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(count, 3))


def mask_nearest(row_pos: np.ndarray, col_pos: np.ndarray, frac: float) -> np.ndarray:
    dist = progon.pair_distances(row_pos, col_pos)
    flat = dist.ravel()
    keep = max(1, int(round(frac * flat.size)))
    border = np.partition(flat, keep - 1)[keep - 1]
    return dist <= border


def errors(W: np.ndarray, approx, vectors) -> float:
    vals = [progon.rel_l2(approx(v), W @ v) for v in vectors]
    return float(np.mean(vals))


def masked_apply(masked):
    return lambda v: masked @ v


def rank3_apply(U, S, Vt):
    return lambda v: progon.lowrank_apply(U, S, Vt, v, 3)


def random_rank3_apply(W, rng):
    n_out, n_in = W.shape
    right, _ = np.linalg.qr(rng.normal(size=(n_in, 3)))
    left, _ = np.linalg.qr(rng.normal(size=(n_out, 3)))
    core = left.T @ W @ right
    return lambda v: left @ (core @ (right.T @ v))


def evaluate(name: str, W: np.ndarray, vectors, rng, oracle=None) -> dict:
    print(name, W.shape, flush=True)
    energy = W * W
    U, S, Vt = progon.svd_full(W)
    n_out, n_in = W.shape
    layouts = {
        "случайное облако": (random_positions(n_out, rng), random_positions(n_in, rng)),
        "решётка по номеру": (grid_positions(n_out), grid_positions(n_in)),
        "лучшие 3 оси слоя": progon.positions_from(U, S, Vt, 3),
    }
    if oracle is not None:
        layouts["настоящие координаты"] = (oracle, oracle)
    report = {"name": name, "shape": [n_out, n_in], "схемы": {}}
    for title, (row_pos, col_pos) in layouts.items():
        mask = mask_nearest(row_pos, col_pos, FRAC)
        kept = progon.captured(energy, progon.pair_distances(row_pos, col_pos), FRAC, largest=False)
        report["схемы"][title] = {
            "энергия_после_гашения": kept,
            "доля_оставленных_связей": float(mask.mean()),
            "ошибка": errors(W, masked_apply(W * mask), vectors),
        }
    report["схемы"]["самые большие веса"] = {
        "энергия_после_гашения": progon.captured(energy, energy, FRAC, largest=True),
        "доля_оставленных_связей": FRAC,
        "ошибка": errors(
            W,
            masked_apply(W * (energy >= np.partition(energy.ravel(), -max(1, int(FRAC * energy.size)))[-max(1, int(FRAC * energy.size))])),
            vectors,
        ),
    }
    report["проекция_в_3_числа"] = {
        "лучшие_оси_слоя": errors(W, rank3_apply(U, S, Vt), vectors),
        "случайные_оси": errors(W, random_rank3_apply(W, rng), vectors),
        "энергия_трёх_осей": progon.spectrum_report(S)["top3"],
    }
    return report


def main():
    rng = np.random.default_rng(5)
    W = np.load(ROOT / "веса" / "gpt2_h0_c_fc.npy").astype(np.float64)
    vecs = []
    for _ in range(8):
        v = rng.normal(size=W.shape[1])
        vecs.append(v / np.linalg.norm(v))
    reports = [evaluate("gpt2 слой0 c_fc", W, vecs, rng)]
    sharp, pts = progon.mat_sharp(96, rng)
    blob = np.exp(-np.sum((pts - pts[20]) ** 2, axis=1) / (2 * 0.2**2))
    blob /= np.linalg.norm(blob)
    sharp_vecs = [blob]
    for _ in range(4):
        v = rng.normal(size=sharp.shape[1])
        sharp_vecs.append(v / np.linalg.norm(v))
    reports.append(evaluate("резкая решётка, контроль", sharp, sharp_vecs, rng, pts))
    dest = ROOT / "rezultat_iskusstvennoe.json"
    dest.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
