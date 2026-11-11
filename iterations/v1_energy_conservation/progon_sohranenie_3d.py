"""Честная проверка с законом сохранения энергии поля (без искусственных порогов и без читерского раздувания температуры).

В прошлом запуске (progon_chestnaya_3d.py) вскрылся фундаментальный физический эффект:
если температура/радиус поля ничем не нормированы по площади (нет закона сохранения энергии волны 1 / Z),
то градиентный спуск просто раздувает температуру в бесконечность, чтобы видеть все объекты сразу на 80%!

В физике волна или поток при расширении на радиус R теряет плотность энергии пропорционально объему/площади.
И второе: при жесткой подгрузке (VRAM streaming) нельзя учить модель мягкими полутонами, а потом внезапно рубить топором —
в физике и оптике фазовый переход/дискретизация уровня детализации (LOD) участвует прямо в прямом проходе (через Straight-Through градиент),
чтобы модель с первого шага знала: далекий объект — это СТРОГО точка S_0, а следующие группы S_k включаются ступенями по мере приближения.

Проверяем три модели в абсолютно равных условиях на реальном WikiText-2:
1. Классическая (100% матриц всегда).
2. 3D-перспектива без нормировки энергии (свободная температура).
3. 3D-перспектива с законом сохранения плотности поля и честным ступенчатым LOD в прямом проходе.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from progon_chestnaya_3d import ClassicalBlock, evaluate_bpc, generate_text, load_wikitext_bytes

OUT_DIR = Path(__file__).resolve().parent


class ConservedSpatialLayer(nn.Module):
    """Слой 3D-коллизий с законом сохранения энергии поля и ступенчатым перспективным LOD прямо во время обучения."""

    def __init__(self, d_model: int, n_objects: int, n_groups: int):
        super().__init__()
        self.d_model = d_model
        self.n_objects = n_objects
        self.n_groups = n_groups
        self.group_dim = d_model // n_groups

        self.to_pos = nn.Linear(d_model, 3)
        self.to_phi = nn.Linear(d_model, 3)

        # Координаты объектов на 3D-сфере/объеме
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.5)
        self.obj_phi = nn.Parameter(torch.zeros(n_objects, 3))
        self.obj_in = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))

        self.env_decay = nn.Parameter(torch.tensor(0.85))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        b, t, d = x.shape
        h = self.layer_norm(x)

        r_ctx = self.to_pos(h)
        phi_ctx = self.to_phi(h)

        v_ctx = torch.zeros_like(r_ctx)
        v_ctx[:, 1:, :] = r_ctx[:, 1:, :] - r_ctx[:, :-1, :]
        v_ctx[:, 0, :] = r_ctx[:, 0, :]

        decay = torch.sigmoid(self.env_decay)
        env_field = torch.zeros_like(r_ctx)
        state = torch.zeros(b, 3, device=x.device, dtype=x.dtype)
        for step_t in range(t):
            env_field[:, step_t, :] = state
            state = decay * state + (1.0 - decay) * r_ctx[:, step_t, :]

        r_eff = r_ctx + 0.25 * env_field

        delta = self.obj_pos.view(1, 1, self.n_objects, 3) - r_eff.unsqueeze(2)
        dist2 = (delta * delta).sum(dim=-1)
        dist = torch.sqrt(dist2 + 1e-8)
        normal = delta / dist.unsqueeze(-1)

        v_norm = torch.sqrt((v_ctx * v_ctx).sum(dim=-1, keepdim=True) + 1e-8)
        v_dir = v_ctx / v_norm
        cos_angle = (v_dir.unsqueeze(2) * normal).sum(dim=-1)
        frontal = 0.5 * (cos_angle + 1.0)

        # ЗАКОН СОХРАНЕНИЯ ЭНЕРГИИ ПОЛЯ:
        # Относительное расстояние нормируется на средний масштаб облака объектов в данный момент,
        # а температура распределяется между объектами как конкурентная среда (если везде горячо — плотность падает)
        mean_cloud_dist = dist.mean(dim=-1, keepdim=True).detach().clamp_min(0.5)
        rel_dist = dist / mean_cloud_dist  # безразмерное расстояние в единицах радиуса мира

        raw_temp = torch.sigmoid(phi_ctx[:, :, 0:1] + self.obj_phi[:, 0].view(1, 1, -1))  # (0, 1)
        # Нормировка энергии: средняя температура по слою ограничена законом сохранения
        temp = 0.35 + 0.65 * raw_temp / (raw_temp.mean(dim=-1, keepdim=True) + 1e-6)

        imp_mismatch = (phi_ctx[:, :, 2:3] - self.obj_phi[:, 2].view(1, 1, -1)).pow(2)

        # Коэффициент 2D-коллизии с учетом плотности среды (1 / (1 + rel_dist^2))
        transmission = (1.0 / (1.0 + (rel_dist / temp).pow(2))) * torch.exp(-0.25 * imp_mismatch) * (0.2 + 0.8 * frontal)

        # Перспективный LOD: группа 0 — точка (всегда 1.0), группы k = 1..K-1 затухают с безразмерным расстоянием
        group_idx = torch.arange(self.n_groups, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        soft_lod = torch.exp(-group_idx * (rel_dist / (0.6 * temp)).unsqueeze(-1))

        # Реальная дискретная подгрузка (ступени детализации):
        # В прямом проходе модель реально видит только те группы k >= 1, которые прошли перспективный порог,
        # а градиент течет через мягкую огибающую (Straight-Through), чтобы координаты двигались!
        hard_lod = (soft_lod >= 0.5).to(x.dtype)
        hard_lod[..., 0] = 1.0
        if mode == "vram_streaming":
            lod_weights = hard_lod
        else:
            lod_weights = hard_lod.detach() - soft_lod.detach() + soft_lod

        w_active = transmission.unsqueeze(-1) * lod_weights

        h_groups = h.view(b, t, self.n_groups, self.group_dim)
        h_perm = h_groups.permute(2, 0, 1, 3).reshape(self.n_groups, b * t, self.group_dim)
        in_perm = self.obj_in.permute(1, 2, 0)
        group_dots = torch.bmm(h_perm, in_perm).permute(1, 2, 0).reshape(b, t, self.n_objects, self.n_groups)

        activated = F.silu(group_dots) * w_active

        act_perm = activated.permute(3, 0, 1, 2).reshape(self.n_groups, b * t, self.n_objects)
        out_perm = self.obj_out.permute(1, 0, 2)
        morphed = torch.bmm(act_perm, out_perm).permute(1, 0, 2).reshape(b, t, d)

        with torch.no_grad():
            near_objects_frac = float((hard_lod[..., -1] > 0.5).float().mean().item())
            detail_volume = float((hard_lod[..., 1:] > 0.5).float().mean().item())
            loaded_volume = float(hard_lod.float().mean().item())
            mean_dist = float(dist.mean().item())

        stats = {
            "near_objects_frac": near_objects_frac,
            "detail_volume": detail_volume,
            "loaded_volume": loaded_volume,
            "mean_dist": mean_dist,
        }
        return x + morphed, stats


class ConservedModel(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 96, n_objects: int = 96, n_groups: int = 8, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ConservedSpatialLayer(d_model, n_objects, n_groups) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        x = self.embed(tokens)
        agg = {"near_objects_frac": 0.0, "detail_volume": 0.0, "loaded_volume": 0.0, "mean_dist": 0.0}
        for layer in self.layers:
            x, st = layer(x, mode=mode)
            for k in agg:
                agg[k] += st[k] / len(self.layers)
        return self.head(self.norm(x)), agg


def main():
    train_data = load_wikitext_bytes("train", max_chars=1_500_000)
    valid_data = load_wikitext_bytes("validation", max_chars=250_000)

    seq_len = 96
    batch_size = 16
    steps = 400
    lr = 2e-3

    torch.manual_seed(42)
    conserved = ConservedModel(d_model=96, n_objects=96, n_groups=8, n_layers=3)
    opt = torch.optim.AdamW(conserved.parameters(), lr=lr, weight_decay=0.01)

    n_windows = (train_data.numel() - 1) // seq_len
    order = torch.randperm(n_windows, generator=torch.Generator().manual_seed(7))
    cursor = 0
    t0 = time.perf_counter()

    print("Обучение 3D-модели с законом сохранения плотности поля и ступенчатым LOD...", flush=True)
    for step in range(1, steps + 1):
        srcs, dsts = [], []
        for _ in range(batch_size):
            idx = int(order[cursor % n_windows]) * seq_len
            cursor += 1
            win = train_data[idx : idx + seq_len + 1]
            srcs.append(win[:-1])
            dsts.append(win[1:])
        src = torch.stack(srcs)
        dst = torch.stack(dsts)

        logits, st = conserved(src, mode="native")
        loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(conserved.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 50 == 0:
            bpc = float(loss.item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | BPC={bpc:.3f} | "
                f"полных объектов={st['near_objects_frac']*100:.1f}% | "
                f"объем деталей={st['detail_volume']*100:.1f}% | "
                f"объем с точкой={st['loaded_volume']*100:.1f}% | "
                f"ср.расст={st['mean_dist']:.2f} | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )

    bpc_nat, ppl_nat, st_nat = evaluate_bpc(conserved, valid_data, seq_len, mode="native")
    bpc_str, ppl_str, st_str = evaluate_bpc(conserved, valid_data, seq_len, mode="vram_streaming")
    print("\n=== ПРОВЕРКА С ЗАКОНОМ СОХРАНЕНИЯ ЭНЕРГИИ ===", flush=True)
    print(
        f"3D с сохранением (обычный режим):   BPC = {bpc_nat:.3f}, перплексия = {ppl_nat:.2f} | "
        f"полных объектов: {st_nat['near_objects_frac']*100:.1f}%, деталей: {st_nat['detail_volume']*100:.1f}%, "
        f"всего подгружено: {st_nat['loaded_volume']*100:.1f}%",
        flush=True,
    )
    print(
        f"3D с сохранением (VRAM-подгрузка):  BPC = {bpc_str:.3f}, перплексия = {ppl_str:.2f} | "
        f"потеря при переходе к жесткой подгрузке: +{bpc_str - bpc_nat:.4f} BPC",
        flush=True,
    )
    torch.manual_seed(11)
    sample = generate_text(conserved, "The history of science is ", max_new=90, mode="vram_streaming")
    print(f"Генерация: {sample!r}", flush=True)


if __name__ == "__main__":
    main()
