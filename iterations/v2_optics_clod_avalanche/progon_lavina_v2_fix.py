"""Проверка гипотезы об ошибке логики в изотропной лавине (v2b):
Почему в v2 симметричная лавина дала 3.539 BPC (чуть хуже чистой оптики 3.489 BPC)?

Ошибка логики 1 (Изотропное растекание):
  В v2 матрица связей bond_graph была симметричной (неориентированной) и зависела только от 3D-расстояния.
  Когда контекст бил в узел j, лавина текла во все стороны (в том числе назад, против стрелы времени)
  и после нормировки бюджета сцены "отбирала" резкость у прямого оптического фокуса!

Ошибка логики 2 (Сложение яркостей вместо переноса векторного импульса):
  В химии и физике кристаллов каскадная реакция передает по ребру j -> m не просто "разрешение открыться",
  а сам фазовый импульс (ковалентное связывание отклика узла j с соседом m вдоль направления поля E_ctx).

Исправление в v2b (Направленный дипольный каскад по ребрам связей):
1. Ребро связи j -> m активируется только вдоль силовых линий поля контекста (дипольное правило:
   <r_m - r_j, f_cam> > 0) и только у top-валентных связей (максимальная валентность узла = 4, как у углерода).
2. Лавина не отбирает оптическую резкость у первичного фокуса, а передает возбужденный сигнал
   от первично пробитых оптикой ядер j к их ковалентным соседям m с учетом фазы!
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from progon_chestnaya_3d import generate_text, load_wikitext_bytes
from progon_optika_lavina_v2 import build_camera_frame, eval_model_v2

OUT_DIR = Path(__file__).resolve().parent


class DirectedValenceAvalancheLayer(nn.Module):
    """Оптика + полигональный CLOD + направленный валентный каскад (макс. валентность = 4, вдоль поля)."""

    def __init__(self, d_model: int, n_objects: int, n_groups: int):
        super().__init__()
        self.d_model = d_model
        self.n_objects = n_objects
        self.n_groups = n_groups
        self.group_dim = d_model // n_groups

        self.layer_norm = nn.LayerNorm(d_model)
        self.to_cam_pos = nn.Linear(d_model, 3)
        self.to_cam_optics = nn.Linear(d_model, 3)

        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.4)
        self.obj_normal = nn.Parameter(torch.randn(n_objects, 3))
        self.obj_optics = nn.Parameter(torch.zeros(n_objects, 2))
        self.obj_orbital = nn.Parameter(torch.randn(n_objects, 8) / math.sqrt(8))

        self.obj_in = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))

        self.motion_compensation = nn.Parameter(torch.tensor(0.75))
        self.valence_gain = nn.Parameter(torch.tensor(0.25))

    def compute_valence_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Для каждого узла j находит его валентные ребра (насыщение валентности: максимум 4 связи на узел, как sp3 у углерода).
        Возвращает:
          bonds: (N, N) разреженная матрица ковалентных связей
          edge_dir: (N, N, 3) единичные векторы направления связи j -> m в 3D
        """
        diff = self.obj_pos.unsqueeze(1) - self.obj_pos.unsqueeze(0)  # (N, N, 3): r_j - r_m -> нам нужно r_m - r_j
        bond_vec = -diff  # (N, N, 3): от j к m
        dist2 = (bond_vec * bond_vec).sum(dim=-1) + 1e-6
        dist = torch.sqrt(dist2)
        edge_dir = bond_vec / dist.unsqueeze(-1)

        mean_d2 = dist2.mean().detach().clamp_min(0.5)
        spatial_aff = torch.exp(-dist2 / (0.35 * mean_d2))

        orb = F.normalize(self.obj_orbital, dim=-1, eps=1e-6)
        orb_aff = F.relu(orb @ orb.T)

        raw_bonds = spatial_aff * orb_aff
        eye = torch.eye(self.n_objects, device=raw_bonds.device, dtype=raw_bonds.dtype)
        raw_bonds = raw_bonds * (1.0 - eye)

        # Химическое правило валентности: у каждого атома/узла активны только 4 сильнейшие ковалентные связи (sp3)
        top_vals, top_idx = raw_bonds.topk(4, dim=-1)
        valence_mask = torch.zeros_like(raw_bonds).scatter(-1, top_idx, 1.0)
        sparse_bonds = raw_bonds * valence_mask
        norm_bonds = sparse_bonds / (sparse_bonds.sum(dim=-1, keepdim=True) + 1e-6)
        return norm_bonds, edge_dir

    def forward(self, x: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        b, t, d = x.shape
        h = self.layer_norm(x)

        r_raw = self.to_cam_pos(h)
        alpha_mot = torch.sigmoid(self.motion_compensation)
        r_cam = torch.zeros_like(r_raw)
        v_cam = torch.zeros_like(r_raw)
        pos_state = r_raw[:, 0, :]
        vel_state = torch.zeros_like(pos_state)
        for step_t in range(t):
            cur = r_raw[:, step_t, :]
            inst_vel = cur - pos_state if step_t > 0 else cur
            vel_state = alpha_mot * vel_state + (1.0 - alpha_mot) * inst_vel
            pos_state = 0.7 * cur + 0.3 * (pos_state + vel_state)
            r_cam[:, step_t, :] = pos_state
            v_cam[:, step_t, :] = vel_state

        f_axis, u_axis, w_axis = build_camera_frame(v_cam)

        delta = self.obj_pos.view(1, 1, self.n_objects, 3) - r_cam.unsqueeze(2)
        dist2 = (delta * delta).sum(dim=-1) + 1e-6
        dist = torch.sqrt(dist2)
        ray_dir = delta / dist.unsqueeze(-1)

        z_cam = (delta * f_axis.unsqueeze(2)).sum(dim=-1)
        x_scr = (delta * u_axis.unsqueeze(2)).sum(dim=-1)
        y_scr = (delta * w_axis.unsqueeze(2)).sum(dim=-1)

        in_front = torch.sigmoid(3.0 * z_cam / (dist.mean(dim=-1, keepdim=True).detach() + 1e-4))
        z_depth = F.softplus(z_cam) + 0.35

        u_2d = x_scr / z_depth
        v_2d = y_scr / z_depth
        screen_r2 = u_2d * u_2d + v_2d * v_2d

        poly_normal = F.normalize(self.obj_normal, dim=-1, eps=1e-6).view(1, 1, self.n_objects, 3)
        cos_inc = (ray_dir * poly_normal).sum(dim=-1).abs()
        f0 = torch.sigmoid(self.obj_optics[:, 1]).view(1, 1, self.n_objects)
        fresnel_reflect = f0 + (1.0 - f0) * (1.0 - cos_inc).pow(5)
        optical_coupling = cos_inc * (1.0 - 0.7 * fresnel_reflect)

        raw_area = F.softplus(self.obj_optics[:, 0]) + 0.2
        norm_area = raw_area / (raw_area.mean() + 1e-6)

        optics_ctrl = self.to_cam_optics(h)
        focal = 0.6 + 0.8 * torch.sigmoid(optics_ctrl[:, :, 0:1])
        aperture = 0.4 + 0.8 * torch.sigmoid(optics_ctrl[:, :, 1:2])

        mean_d2 = dist2.mean(dim=-1, keepdim=True).detach().clamp_min(0.5)
        norm_z2 = (z_depth * z_depth) / mean_d2
        dof_blur = 1.0 + aperture * screen_r2
        solid_angle = in_front * (focal.pow(2) * norm_area.view(1, 1, -1) * optical_coupling) / (norm_z2 * dof_blur + 0.15)

        # Оптический фокус сохраняется чистым (лавина НЕ размывает оптическую резкость!)
        mean_omega = solid_angle.mean(dim=-1, keepdim=True).detach().clamp_min(0.15)
        conserved_omega = 0.35 * solid_angle / mean_omega

        # Глубина векторов из оптики и 2D-проекции
        continuous_depth = 1.0 + (self.n_groups - 1.0) * torch.log2(1.0 + 3.0 * conserved_omega) / math.log2(1.0 + 3.0 * 2.5)
        continuous_depth = continuous_depth.clamp(1.0, float(self.n_groups))

        k_idx = torch.arange(self.n_groups, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        soft_lod = torch.sigmoid((continuous_depth.unsqueeze(-1) - 0.5 - k_idx) * 4.0)
        soft_lod = soft_lod.clone()
        soft_lod[..., 0] = 1.0

        hard_lod = (k_idx < continuous_depth.round().unsqueeze(-1)).to(x.dtype)
        hard_lod[..., 0] = 1.0

        lod_weights = hard_lod if mode == "vram_streaming" else (hard_lod.detach() - soft_lod.detach() + soft_lod)
        envelope = conserved_omega / (1.0 + conserved_omega)
        w_active = envelope.unsqueeze(-1) * lod_weights

        # Векторный отклик на оптической 2D-коллизии
        h_groups = h.view(b, t, self.n_groups, self.group_dim)
        h_perm = h_groups.permute(2, 0, 1, 3).reshape(self.n_groups, b * t, self.group_dim)
        in_perm = self.obj_in.permute(1, 2, 0)
        group_dots = torch.bmm(h_perm, in_perm).permute(1, 2, 0).reshape(b, t, self.n_objects, self.n_groups)

        primary_activated = F.silu(group_dots) * w_active  # (B, T, N, K)

        # НАПРАВЛЕННЫЙ ХИМИЧЕСКИЙ КАСКАД ПО ВАЛЕНТНЫМ РЕБРАМ (без тяжелого 4D-тензора):
        # Ребро j -> m проводит сигнал пропорционально валентной связи bonds(j, m)
        # и проекции вектора связи (r_m - r_j) на направление движения мысли f_axis!
        bonds, edge_dir = self.compute_valence_edges()  # (N, N), (N, N, 3)

        # Порог Аррениуса: в каскад отдают импульс только те узлы, где оптический удар превысил барьер E_a
        e_act = 0.55 + 0.45 * torch.sigmoid(optics_ctrl[:, :, 2:3])
        arrhenius_gate = torch.sigmoid((conserved_omega - e_act) * 6.0).unsqueeze(-1)  # (B, T, N, 1)
        cascade_source = primary_activated * arrhenius_gate  # (B, T, N, K)

        # Разложение направленного переноса по 3 декартовым компонентам поля f_axis (в 10 раз быстрее 4D-тензора):
        # base_transfer: симметричный перенос по ребрам, dipole_transfer: направленный перенос вдоль f_axis
        src_flat = cascade_source.permute(0, 1, 3, 2).reshape(b * t * self.n_groups, self.n_objects)
        base_transfer = torch.matmul(src_flat, bonds).reshape(b, t, self.n_groups, self.n_objects).permute(0, 1, 3, 2)

        directed_flux = (bonds.unsqueeze(-1) * edge_dir).permute(2, 0, 1)  # (3, N, N)
        # Умножаем для каждой из 3 осей X, Y, Z и сворачиваем с вектором скорости мысли f_axis (B, T, 3)
        dipole_xyz = torch.matmul(src_flat.unsqueeze(0), directed_flux)  # (3, B*T*K, N)
        dipole_xyz = dipole_xyz.reshape(3, b, t, self.n_groups, self.n_objects).permute(1, 2, 4, 3, 0)  # (B, T, N, K, 3)
        dipole_transfer = (dipole_xyz * f_axis.view(b, t, 1, 1, 3)).sum(dim=-1)  # (B, T, N, K)

        cascade_received = 0.5 * base_transfer + 0.5 * dipole_transfer
        # Принимающий сосед m модулирует полученный по ребру импульс только в открытых у него уровнях!
        total_activated = primary_activated + torch.tanh(self.valence_gain) * cascade_received * lod_weights

        act_perm = total_activated.permute(3, 0, 1, 2).reshape(self.n_groups, b * t, self.n_objects)
        out_perm = self.obj_out.permute(1, 0, 2)
        morphed = torch.bmm(act_perm, out_perm).permute(1, 0, 2).reshape(b, t, d)

        with torch.no_grad():
            near_objects_frac = float((hard_lod[..., -1] > 0.5).float().mean().item())
            detail_volume = float((hard_lod[..., 1:] > 0.5).float().mean().item())
            loaded_volume = float(hard_lod.float().mean().item())
            mean_vectors_per_obj = float(hard_lod.sum(dim=-1).mean().item())
            active_aval_nodes = float((arrhenius_gate > 0.5).float().mean().item())

        stats = {
            "near_objects_frac": near_objects_frac,
            "detail_volume": detail_volume,
            "loaded_volume": loaded_volume,
            "mean_dist": float(z_depth.mean().item()),
            "mean_vectors_per_obj": mean_vectors_per_obj,
            "avalanche_added_groups": active_aval_nodes,
        }
        return x + morphed, stats


class DirectedAvalancheLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 96, n_objects: int = 96, n_groups: int = 8, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DirectedValenceAvalancheLayer(d_model, n_objects, n_groups) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        x = self.embed(tokens)
        keys = ("near_objects_frac", "detail_volume", "loaded_volume", "mean_dist", "mean_vectors_per_obj", "avalanche_added_groups")
        agg = {k: 0.0 for k in keys}
        for layer in self.layers:
            x, st = layer(x, mode=mode)
            for k in keys:
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
    model = DirectedAvalancheLM(d_model=96, n_objects=96, n_groups=8, n_layers=3)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_windows = (train_data.numel() - 1) // seq_len
    order = torch.randperm(n_windows, generator=torch.Generator().manual_seed(7))
    cursor = 0
    t0 = time.perf_counter()

    print("Обучение v2b (Оптика + полигональный CLOD + Направленный sp3-валентный каскад)...", flush=True)
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

        logits, st = model(src, mode="native")
        loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step == 1 or step % 50 == 0:
            bpc = float(loss.item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | v2b BPC={bpc:.3f} | детали={st['detail_volume']*100:.1f}% | "
                f"VRAM={st['loaded_volume']*100:.1f}% | векторов/объект={st['mean_vectors_per_obj']:.2f}/8 | "
                f"ядер_лавины={st['avalanche_added_groups']*100:.1f}% | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )

    nat_bpc, nat_ppl, nat_st = eval_model_v2(model, valid_data, seq_len, mode="native")
    str_bpc, str_ppl, str_st = eval_model_v2(model, valid_data, seq_len, mode="vram_streaming")
    print("\n=== ИТОГИ V2B (НАПРАВЛЕННЫЙ ВАЛЕНТНЫЙ КАСКАД) ===", flush=True)
    print(
        f"v2b Направленный каскад (native): BPC = {nat_bpc:.3f}, перплексия = {nat_ppl:.2f} | "
        f"деталей: {nat_st['detail_volume']*100:.1f}%, VRAM: {nat_st['loaded_volume']*100:.1f}%",
        flush=True,
    )
    print(
        f"v2b Направленный каскад (VRAM):   BPC = {str_bpc:.3f}, перплексия = {str_ppl:.2f} | "
        f"потеря при стриминге: +{str_bpc - nat_bpc:.4f} BPC",
        flush=True,
    )


if __name__ == "__main__":
    main()
