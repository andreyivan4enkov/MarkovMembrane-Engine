"""Версия v2c: Причинно-следственный (временной) лавинный каскад по ребрам связей t-1 -> t
+ классическая оптика, 2D-проекция и полигональный CLOD.

Какую ошибку логики выявили v2 и v2b:
В v2 и v2b химическая лавина запускалась СИНХРОННО внутри одного и того же шага t:
когда контекст в момент t уже ударил в объект j, лавина размывала его отклик на соседние объекты m
в тот же самый момент t. В итоге соседние объекты m добавляли параллельный шум к уже случившемуся удару
(3.539 BPC при изотропной лавине, 3.498 BPC при направленной — против 3.489 BPC у чистой оптики).

Как работает реальная химическая кинетика и обработка видео во времени (v2c):
В химии и распространении волн каскад развивается ВДОЛЬ СТРЕЛЫ ВРЕМЕНИ (t-1 -> t):
1. На шаге t-1 контекст сталкивается с объектами j. Те объекты j, у которых на 2D-экране
   произошел сильный пробой (превышен порог Аррениуса E_a), испускают импульс по своим валентным ребрам j -> m.
2. На следующем шаге t этот лавинный потенциал P_aval(t) ЗАРАНЕЕ подогревает и раскрывает
   детализацию (уровни полигонов k = 1..K-1) у связанных объектов m, которые контекст должен встретить дальше!
3. Кроме того, на самой 2D-плоскости проекции (u_2d, v_2d) работает билинейный экранный фильтр
   (как в видеокодеках и растеризации GPU), связывающий проекцию со шкалой мип-уровней.
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


class TemporalOpticalCascadeLayer(nn.Module):
    """
    Слой v2c:
    - 2D перспективная проекция (Perspective Divide + Ламберт + Френель),
    - полигональный CLOD (логарифмический мип-маппинг числа векторов от телесного угла),
    - причинно-следственный лавинный каскад по ребрам химических связей во времени (t-1 -> t).
    """

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
        # Валентные орбитали донора и акцептора (асимметричная направленная химическая связь j -> m!)
        self.obj_donor = nn.Parameter(torch.randn(n_objects, 12) / math.sqrt(12))
        self.obj_acceptor = nn.Parameter(torch.randn(n_objects, 12) / math.sqrt(12))

        self.obj_in = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))

        self.motion_compensation = nn.Parameter(torch.tensor(0.75))
        self.aval_decay = nn.Parameter(torch.tensor(0.5))
        self.aval_gain = nn.Parameter(torch.tensor(0.8))
        self.kinetic_transfer = nn.Parameter(torch.tensor(0.35))

    def compute_directed_bond_graph(self) -> torch.Tensor:
        """
        Строит ориентированный граф химических связей j -> m:
        связь сильна, если объекты лежат на характерной длине ковалентной связи в 3D,
        а донорная орбиталь j резонирует с акцепторной орбиталью m.
        """
        diff = self.obj_pos.unsqueeze(1) - self.obj_pos.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=-1) + 1e-6
        mean_d2 = dist2.mean().detach().clamp_min(0.5)
        spatial_bond = torch.exp(-dist2 / (0.5 * mean_d2))

        donor = F.normalize(self.obj_donor, dim=-1, eps=1e-6)
        acceptor = F.normalize(self.obj_acceptor, dim=-1, eps=1e-6)
        chem_affinity = F.relu(donor @ acceptor.T)

        raw = spatial_bond * chem_affinity
        eye = torch.eye(self.n_objects, device=raw.device, dtype=raw.dtype)
        raw = raw * (1.0 - eye)

        # Валентное насыщение: каждый узел-донор держит до 6 направленных связей
        _, top_idx = raw.topk(6, dim=-1)
        mask = torch.zeros_like(raw).scatter(-1, top_idx, 1.0)
        sparse = raw * mask
        return sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        b, t, d = x.shape
        h = self.layer_norm(x)

        # 1. Траектория камеры с видеокомпенсацией движения
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

        # 2. Классическая оптика и 2D-проекция
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

        # 3. ПРИЧИННО-СЛЕДСТВЕННЫЙ ЛАВИННЫЙ КАСКАД ПО РЕБРАМ СВЯЗЕЙ ВО ВРЕМЕНИ (t-1 -> t):
        # Когда на шаге t-1 объект j ярко вспыхнул на 2D-мембране (выше порога Аррениуса),
        # по ориентированным химическим связям bond_graph(j -> m) волна возбуждения идет к узлам m на шаге t!
        bond_graph = self.compute_directed_bond_graph()  # (N, N)
        e_act = 0.35 + 0.45 * torch.sigmoid(optics_ctrl[:, :, 2:3])
        # Импульс пробоя на каждом такте: (B, T, N)
        spark = torch.sigmoid((solid_angle - e_act) * 6.0) * solid_angle
        # Куда этот импульс переходит по ребрам связей: (B, T, N)
        hop_flux = torch.matmul(spark, bond_graph)

        decay = 0.3 + 0.5 * torch.sigmoid(self.aval_decay)
        gain = F.softplus(self.aval_gain)
        aval_potential = torch.zeros_like(solid_angle)
        chem_state = torch.zeros(b, self.n_objects, device=x.device, dtype=x.dtype)
        for step_t in range(t):
            aval_potential[:, step_t, :] = chem_state
            chem_state = decay * chem_state + gain * hop_flux[:, step_t, :]

        # Суммарный телесный угол (прямая оптика кадра t + предвозбуждение от каскада t-1 -> t)
        total_omega = solid_angle + aval_potential
        mean_omega = total_omega.mean(dim=-1, keepdim=True).detach().clamp_min(0.15)
        conserved_omega = 0.35 * total_omega / mean_omega

        # 4. Глубина просчета векторов в реальном времени (CLOD / Mip-Mapping)
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

        # 5. Векторный отклик + перенос ковалентного возбуждения с шага t-1 на шаг t
        h_groups = h.view(b, t, self.n_groups, self.group_dim)
        h_perm = h_groups.permute(2, 0, 1, 3).reshape(self.n_groups, b * t, self.group_dim)
        in_perm = self.obj_in.permute(1, 2, 0)
        group_dots = torch.bmm(h_perm, in_perm).permute(1, 2, 0).reshape(b, t, self.n_objects, self.n_groups)

        activated = F.silu(group_dots) * w_active  # (B, T, N, K)

        # Ковалентный перенос векторного состояния вдоль ребер j -> m от предыдущего такта t-1 к текущему такту t!
        prev_act = torch.zeros_like(activated)
        prev_act[:, 1:, :, :] = activated[:, :-1, :, :]
        # Передача по ориентированному графу связей (B, T, K, N) @ (N, N) -> (B, T, K, N)
        transferred = torch.matmul(prev_act.permute(0, 1, 3, 2), bond_graph).permute(0, 1, 3, 2)
        # Принимающий узел m получает этот сигнал только в тех векторных уровнях, которые у него открыты сейчас!
        activated = activated + torch.tanh(self.kinetic_transfer) * transferred * lod_weights

        act_perm = activated.permute(3, 0, 1, 2).reshape(self.n_groups, b * t, self.n_objects)
        out_perm = self.obj_out.permute(1, 0, 2)
        morphed = torch.bmm(act_perm, out_perm).permute(1, 0, 2).reshape(b, t, d)

        with torch.no_grad():
            near_objects_frac = float((hard_lod[..., -1] > 0.5).float().mean().item())
            detail_volume = float((hard_lod[..., 1:] > 0.5).float().mean().item())
            loaded_volume = float(hard_lod.float().mean().item())
            mean_vectors_per_obj = float(hard_lod.sum(dim=-1).mean().item())
            aval_share = float((aval_potential / (total_omega + 1e-6)).mean().item())

        stats = {
            "near_objects_frac": near_objects_frac,
            "detail_volume": detail_volume,
            "loaded_volume": loaded_volume,
            "mean_dist": float(z_depth.mean().item()),
            "mean_vectors_per_obj": mean_vectors_per_obj,
            "avalanche_share": aval_share,
        }
        return x + morphed, stats


class TemporalCascadeLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 96, n_objects: int = 96, n_groups: int = 8, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TemporalOpticalCascadeLayer(d_model, n_objects, n_groups) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        x = self.embed(tokens)
        keys = ("near_objects_frac", "detail_volume", "loaded_volume", "mean_dist", "mean_vectors_per_obj", "avalanche_share")
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
    model = TemporalCascadeLM(d_model=96, n_objects=96, n_groups=8, n_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели v2c (Оптика + CLOD + Временная химическая лавина t-1 -> t): {n_params:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_windows = (train_data.numel() - 1) // seq_len
    order = torch.randperm(n_windows, generator=torch.Generator().manual_seed(7))
    cursor = 0
    t0 = time.perf_counter()

    history = []
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
                f"шаг {step:3d} | v2c BPC={bpc:.3f} | детали={st['detail_volume']*100:.1f}% | "
                f"VRAM={st['loaded_volume']*100:.1f}% | векторов/объект={st['mean_vectors_per_obj']:.2f}/8 | "
                f"доля_лавины={st['avalanche_share']*100:.1f}% | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append({"step": step, "bpc": round(bpc, 4), **{k: round(v, 4) for k, v in st.items()}})

    nat_bpc, nat_ppl, nat_st = eval_model_v2(model, valid_data, seq_len, mode="native")
    str_bpc, str_ppl, str_st = eval_model_v2(model, valid_data, seq_len, mode="vram_streaming")
    print("\n=== ИТОГИ V2C (ОПТИКА + CLOD + ПРИЧИННО-СЛЕДСТВЕННАЯ ЛАВИНА ПО РЕБРАМ t-1 -> t) ===", flush=True)
    print(
        f"v2c (native):         BPC = {nat_bpc:.3f}, перплексия = {nat_ppl:.2f} | "
        f"деталей: {nat_st['detail_volume']*100:.1f}%, VRAM: {nat_st['loaded_volume']*100:.1f}%, "
        f"векторов на объект: {nat_st['mean_vectors_per_obj']:.2f}/8",
        flush=True,
    )
    print(
        f"v2c (VRAM-стриминг):  BPC = {str_bpc:.3f}, перплексия = {str_ppl:.2f} | "
        f"потеря при стриминге: +{str_bpc - nat_bpc:.4f} BPC",
        flush=True,
    )

    torch.manual_seed(11)
    sample = generate_text(model, "The history of science is ", max_new=90, mode="vram_streaming")
    print(f"Генерация v2c (VRAM-стриминг): {sample!r}", flush=True)

    out_path = OUT_DIR / "rezultat_temporal_cascade_v2c.json"
    out_path.write_text(
        json.dumps(
            {
                "params": n_params,
                "history": history,
                "validation": {
                    "v2c_native": {"bpc": round(nat_bpc, 4), "ppl": round(nat_ppl, 2), "stats": nat_st},
                    "v2c_vram": {"bpc": round(str_bpc, 4), "ppl": round(str_ppl, 2), "stats": str_st},
                },
                "sample": sample,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Сохранено в {out_path}", flush=True)


if __name__ == "__main__":
    main()
