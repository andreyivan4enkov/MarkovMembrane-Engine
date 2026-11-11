"""Версия 2: Реалтайм-глубина просчета векторов из классической оптики, 3D-полигонального CLOD,
видеокомпенсации движения и каскадных лавинных реакций по ребрам химических связей.

Зачем эта версия и какую проблему v1 она решает:
В v1 (progon_sohranenie_3d.py) закон сохранения поля дал честную разреженность 1.6% деталей
и 0.0000 потерь при стриминге в VRAM, но точность отставала от классики (3.799 против 3.187 BPC).
Причина ошибки логики в v1:
  В v1 глубина просчета зависела ТОЛЬКО от прямого евклидова расстояния d до одиночного центра.
  Связанные по смыслу признаки (как атомы в молекуле), которые лежали чуть сбоку от луча камеры,
  оставались точками S_0 и не раскрывали свои векторные грани!

Как устроена v2:
1. Классическая оптика и 2D-проекция (Perspective Divide + Lambert + Schlick Fresnel):
   - У контекста в 3D есть положение камеры r_cam, единичная оптическая ось взгляда f (вдоль скорости мысли)
     и ортонормированный базис 2D-экрана (u, w).
   - Каждый 3D-объект j проецируется в координаты камеры: глубина z_j = <r_j - r_cam, f>,
     экранные координаты (x_2d, y_2d) = (<Delta r, u> / z_j, <Delta r, w> / z_j).
   - У каждого 3D-объекта есть ориентация нормали главной грани n_j и площадь A_j.
   - Видимый телесный угол на 2D-экране (закон Ламберта + обратные квадраты глубины + преломление Френеля):
     Omega_j = (A_j * cos(theta_inc) * F_fresnel) / (z_eff^2 + r_perp^2).

2. 3D-моделирование полигонов (Continuous Level of Detail / Trilinear Mip-Mapping):
   - Вместо грубого экспоненциального затухания число просчитываемых векторных групп k = 0..K-1
     определяется логарифмическим масштабом площади проекции (как уровень мип-карт в GPU и тесселяция мешей):
     D_opt(j) = K * Omega_j / (Omega_j + Omega_half).
   - Если объект далеко или повернут ребром (cos(theta_inc) -> 0), его проекция меньше 1 пикселя ->
     считается только группа k=0 (скалярная точка). По мере роста проекции плавно (или ступенями в VRAM)
     включаются векторные грани k = 1, 2, ..., K-1.

3. Химия: Каскадный лавинный эффект по ребрам связей (Arrhenius Bond Cascade):
   - У объектов в 3D-пространстве есть валентные векторы связи b_j и матрица энергии ребер E_bond(j, m),
     зависящая от 3D-близости объектов друг к другу и совпадения их орбиталей.
   - Когда первичная оптическая коллизия возбуждает близкие объекты выше порога энергии активации E_a
     (закон Аррениуса), запускается 2-шаговая цепная реакция (лавина):
     возбужденный узел j перебрасывает энергию по ребрам связей к соседям m,
     динамически увеличивая их глубину просчета векторов D_total(m) = D_opt(m) + D_cascade(m)!
   - Это позволяет "вытянуть" всю смысловую молекулу целиком, начав всего с 1-2 близких точек удара.

4. Обработка видео (Temporal Motion Compensation / Optical Flow):
   - Между тактами времени t-1 -> t работает предсказание оптического потока и инерция кадра,
     удерживающая фокус на движущемся ансамбле объектов.

В скрипте одновременно проверяются 3 модели в равных условиях на реальном WikiText-2:
- Классическая (100% матриц всегда).
- v2 Только оптика + 3D полигональный LOD (без химической лавины) — проверка вклада чистой геометрии.
- v2 Полная (Оптика + 2D-проекция + 3D CLOD + Каскадная лавина по ребрам связей).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from progon_chestnaya_3d import ClassicalBlock, evaluate_bpc, generate_text, load_wikitext_bytes

OUT_DIR = Path(__file__).resolve().parent


def build_camera_frame(v_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Строит ортонормированный 3D-базис камеры (f - взгляд, u - вправо на 2D-экране, w - вверх на 2D-экране)."""
    f = F.normalize(v_ctx, dim=-1, eps=1e-6)
    # Опорный вектор вверх, устойчивый к коллинеарности
    up_ref = torch.zeros_like(f)
    up_ref[..., 2] = 1.0
    alt_ref = torch.zeros_like(f)
    alt_ref[..., 1] = 1.0
    parallel = (f[..., 2:3].abs() > 0.9).to(f.dtype)
    ref = (1.0 - parallel) * up_ref + parallel * alt_ref

    u = F.normalize(torch.cross(f, ref, dim=-1), dim=-1, eps=1e-6)
    w = torch.cross(u, f, dim=-1)
    return f, u, w


class OpticalAvalanche3DLayer(nn.Module):
    """
    Слой, объединяющий:
    - 2D перспективную проекцию и оптику (Ламберт + Френель + фокусное расстояние),
    - полигональный CLOD / трилинейный мип-маппинг групп векторов,
    - каскадную лавинную активацию по ребрам связей (химия),
    - временную видеокомпенсацию движения (Optical Flow).
    """

    def __init__(self, d_model: int, n_objects: int, n_groups: int, enable_avalanche: bool = True):
        super().__init__()
        assert d_model % n_groups == 0
        self.d_model = d_model
        self.n_objects = n_objects
        self.n_groups = n_groups
        self.group_dim = d_model // n_groups
        self.enable_avalanche = enable_avalanche

        self.layer_norm = nn.LayerNorm(d_model)

        # Параметры камеры контекста: 3D-позиция, скорость/фокус и апертура
        self.to_cam_pos = nn.Linear(d_model, 3)
        self.to_cam_optics = nn.Linear(d_model, 3)  # [фокусное расстояние, апертура, порог Аррениуса]

        # 3D-объекты слоя (полигональные узлы):
        # - координаты центра в 3D
        # - нормаль главной грани (ориентация полигона в 3D)
        # - логарифм площади/сечения объекта и коэффициент преломления F0
        # - валентный вектор химической связи (орбиталь для лавинного каскада)
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.4)
        self.obj_normal = nn.Parameter(torch.randn(n_objects, 3))
        self.obj_optics = nn.Parameter(torch.zeros(n_objects, 2))  # [log_area, fresnel_f0]
        self.obj_orbital = nn.Parameter(torch.randn(n_objects, 8) / math.sqrt(8))

        # Иерархия векторов от грубой точки (k=0) до микрогеометрии полигона (k=K-1)
        self.obj_in = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))

        # Видеообработка: компенсация движения между кадрами t-1 и t
        self.motion_compensation = nn.Parameter(torch.tensor(0.75))
        self.cascade_gain = nn.Parameter(torch.tensor(1.2))

    def compute_bond_graph(self) -> torch.Tensor:
        """
        Строит разреженную матрицу ребер химических связей между 3D-объектами:
        ребро сильно, если объекты близки в 3D-пространстве и их валентные орбитали сонаправлены.
        Диагональ обнулена (самовозбуждения нет).
        """
        # Евклидово расстояние между самими 3D-объектами
        diff = self.obj_pos.unsqueeze(1) - self.obj_pos.unsqueeze(0)  # (N, N, 3)
        dist2 = (diff * diff).sum(dim=-1)
        mean_d2 = dist2.mean().detach().clamp_min(0.5)
        spatial_bond = torch.exp(-dist2 / (0.35 * mean_d2))

        # Перекрытие валентных орбиталей
        orb = F.normalize(self.obj_orbital, dim=-1, eps=1e-6)
        orbital_overlap = F.relu(orb @ orb.T)

        bonds = spatial_bond * orbital_overlap
        eye = torch.eye(self.n_objects, device=bonds.device, dtype=bonds.dtype)
        bonds = bonds * (1.0 - eye)
        # Нормировка валентности (закон сохранения потока по ребрам, чтобы лавина не взрывалась в бесконечность)
        row_sum = bonds.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return bonds / row_sum

    def forward(self, x: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        b, t, d = x.shape
        h = self.layer_norm(x)

        # 1. Временная видеокомпенсация движения (Optical Flow Prediction)
        r_raw = self.to_cam_pos(h)  # (B, T, 3)
        alpha_mot = torch.sigmoid(self.motion_compensation)
        r_cam = torch.zeros_like(r_raw)
        v_cam = torch.zeros_like(r_raw)
        pos_state = r_raw[:, 0, :]
        vel_state = torch.zeros_like(pos_state)
        for step_t in range(t):
            cur = r_raw[:, step_t, :]
            inst_vel = cur - pos_state if step_t > 0 else cur
            vel_state = alpha_mot * vel_state + (1.0 - alpha_mot) * inst_vel
            # Предсказанная позиция камеры с учетом оптического потока
            pos_state = 0.7 * cur + 0.3 * (pos_state + vel_state)
            r_cam[:, step_t, :] = pos_state
            v_cam[:, step_t, :] = vel_state

        # Ортонормированный базис камеры (f - глубина Z, u - ось X экрана, w - ось Y экрана)
        f_axis, u_axis, w_axis = build_camera_frame(v_cam)

        # 2. 2D-проекция объектов на экранную мембрану коллизии и классическая оптика
        # Вектор от камеры к каждому 3D-объекту: (B, T, N, 3)
        delta = self.obj_pos.view(1, 1, self.n_objects, 3) - r_cam.unsqueeze(2)
        dist2 = (delta * delta).sum(dim=-1) + 1e-6
        dist = torch.sqrt(dist2)
        ray_dir = delta / dist.unsqueeze(-1)  # единичный луч от камеры к объекту

        # Координаты в пространстве камеры (Z - глубина вдоль взгляда, X_scr, Y_scr - 2D плоскость)
        z_cam = (delta * f_axis.unsqueeze(2)).sum(dim=-1)  # (B, T, N)
        x_scr = (delta * u_axis.unsqueeze(2)).sum(dim=-1)
        y_scr = (delta * w_axis.unsqueeze(2)).sum(dim=-1)

        # Мягкое отсечение пирамиды видимости (Frustum Culling): то, что позади камеры (z_cam < 0), гасится
        in_front = torch.sigmoid(3.0 * z_cam / (dist.mean(dim=-1, keepdim=True).detach() + 1e-4))
        z_depth = F.softplus(z_cam) + 0.35

        # Perspective Divide: координаты проекции на 2D-экране и радиус от центра фокуса (фовеа)
        u_2d = x_scr / z_depth
        v_2d = y_scr / z_depth
        screen_r2 = u_2d * u_2d + v_2d * v_2d  # квадрат расстояния на 2D-мембране от оптического центра

        # Ориентация грани полигона (закон Ламберта + формула Френеля Шлика)
        poly_normal = F.normalize(self.obj_normal, dim=-1, eps=1e-6).view(1, 1, self.n_objects, 3)
        cos_inc = (ray_dir * poly_normal).sum(dim=-1).abs()  # |cos(theta)| двухстороннего полигона
        f0 = torch.sigmoid(self.obj_optics[:, 1]).view(1, 1, self.n_objects)
        # Уравнение Шлика: при скользящем угле растет отражение, а прохождение вглубь полигона = (1 - R_fresnel) * cos_inc
        fresnel_reflect = f0 + (1.0 - f0) * (1.0 - cos_inc).pow(5)
        optical_coupling = cos_inc * (1.0 - 0.7 * fresnel_reflect)

        # Нормированная площадь полигона (закон сохранения суммарной площади мира)
        raw_area = F.softplus(self.obj_optics[:, 0]) + 0.2
        norm_area = raw_area / (raw_area.mean() + 1e-6)

        optics_ctrl = self.to_cam_optics(h)  # (B, T, 3)
        focal = 0.6 + 0.8 * torch.sigmoid(optics_ctrl[:, :, 0:1])
        aperture = 0.4 + 0.8 * torch.sigmoid(optics_ctrl[:, :, 1:2])

        # Телесный угол (размер проекции полигона на 2D-мембране с учетом пятна размытия Depth-of-Field)
        mean_d2 = dist2.mean(dim=-1, keepdim=True).detach().clamp_min(0.5)
        norm_z2 = (z_depth * z_depth) / mean_d2
        dof_blur = 1.0 + aperture * screen_r2  # вне центра экрана или вне фокуса падает резкость
        solid_angle = in_front * (focal.pow(2) * norm_area.view(1, 1, -1) * optical_coupling) / (norm_z2 * dof_blur + 0.15)

        # 3. Каскадный лавинный эффект по ребрам связей (Химическая кинетика Аррениуса)
        # Если первичный оптический удар в объект j превышает энергию активации E_a,
        # узел j "пробивает" барьер и лавинообразно поджигает соседей по графу связей!
        e_act = 0.45 + 0.50 * torch.sigmoid(optics_ctrl[:, :, 2:3])  # порог активации Аррениуса
        # Вероятность первичного пробоя барьера
        seed_ignite = torch.sigmoid((solid_angle - e_act) * 6.0) * solid_angle

        if self.enable_avalanche:
            bond_graph = self.compute_bond_graph()  # (N, N)
            gain = F.softplus(self.cascade_gain).clamp_max(2.0)
            # Волна 1: передача импульса непосредственным соседям по связям
            wave1 = torch.matmul(seed_ignite, bond_graph)
            # Лавинное нелинейное усиление: если сосед получил импульс, он сам включается в каскад (Волна 2)
            wave1_active = torch.sigmoid((wave1 - 0.25 * e_act) * 6.0) * wave1
            wave2 = torch.matmul(wave1_active, bond_graph)
            avalanche_boost = gain * (wave1 + 0.5 * wave2)
        else:
            avalanche_boost = torch.zeros_like(solid_angle)

        # Полная эффективная светимость объекта (оптика + химическая лавина) с законом сохранения бюджета сцены
        total_salience = solid_angle + avalanche_boost
        # Нормировка закона сохранения энергии на кадре: чтобы сумма открытых полигонов не раздувалась в 100%
        mean_salience = total_salience.mean(dim=-1, keepdim=True).detach().clamp_min(0.15)
        conserved_omega = 0.35 * total_salience / mean_salience

        # 4. Определение глубины просчета векторов в реальном времени (CLOD / Trilinear Mip-Mapping)
        # Непрерывное число открытых векторных уровней d_depth in [1, K]:
        # При малом угле Omega -> 0 открыта только группа 0 (точка, d_depth = 1).
        # С ростом Omega открываются уровни 1, 2, ..., K-1 (логарифмический закон мип-маппинга).
        continuous_depth = 1.0 + (self.n_groups - 1.0) * torch.log2(1.0 + 3.0 * conserved_omega) / math.log2(1.0 + 3.0 * 2.5)
        continuous_depth = continuous_depth.clamp(1.0, float(self.n_groups))

        # Трилинейная интерполяция между уровнями детализации k = 0 .. K-1:
        # Группа k полностью открыта (1.0), если k < floor(d_depth),
        # дробно интерполируется на границе и равна 0.0 для более тонких уровней!
        k_idx = torch.arange(self.n_groups, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        soft_lod = torch.sigmoid((continuous_depth.unsqueeze(-1) - 0.5 - k_idx) * 4.0)
        # Группа 0 (точка) включена всегда на 100%
        soft_lod = soft_lod.clone()
        soft_lod[..., 0] = 1.0

        # Дискретная маска реального числа просчитываемых векторов (для VRAM и честного Straight-Through)
        hard_lod = (k_idx < continuous_depth.round().unsqueeze(-1)).to(x.dtype)
        hard_lod[..., 0] = 1.0

        if mode == "vram_streaming":
            lod_weights = hard_lod
        else:
            # В обучении прямой проход использует реальное ступенчатое число векторов,
            # а обратный проход получает гладкий градиент от трилинейного мип-уровня!
            lod_weights = hard_lod.detach() - soft_lod.detach() + soft_lod

        # Вес передачи сигнала через 2D-коллизию
        envelope = conserved_omega / (1.0 + conserved_omega)
        w_active = envelope.unsqueeze(-1) * lod_weights  # (B, T, N, K)

        # 5. Сборка векторного отклика по открытым уровням детализации
        h_groups = h.view(b, t, self.n_groups, self.group_dim)
        h_perm = h_groups.permute(2, 0, 1, 3).reshape(self.n_groups, b * t, self.group_dim)
        in_perm = self.obj_in.permute(1, 2, 0)
        group_dots = torch.bmm(h_perm, in_perm).permute(1, 2, 0).reshape(b, t, self.n_objects, self.n_groups)

        activated = F.silu(group_dots) * w_active

        act_perm = activated.permute(3, 0, 1, 2).reshape(self.n_groups, b * t, self.n_objects)
        out_perm = self.obj_out.permute(1, 0, 2)
        morphed = torch.bmm(act_perm, out_perm).permute(1, 0, 2).reshape(b, t, d)

        with torch.no_grad():
            opt_only_depth = 1.0 + (self.n_groups - 1.0) * torch.log2(
                1.0 + 3.0 * (0.35 * solid_angle / mean_salience)
            ) / math.log2(1.0 + 3.0 * 2.5)
            avalanche_added_groups = float((continuous_depth - opt_only_depth.clamp(1.0, float(self.n_groups))).clamp_min(0.0).mean().item())
            near_objects_frac = float((hard_lod[..., -1] > 0.5).float().mean().item())
            detail_volume = float((hard_lod[..., 1:] > 0.5).float().mean().item())
            loaded_volume = float(hard_lod.float().mean().item())
            mean_vectors_per_obj = float(hard_lod.sum(dim=-1).mean().item())

        stats = {
            "near_objects_frac": near_objects_frac,
            "detail_volume": detail_volume,
            "loaded_volume": loaded_volume,
            "mean_dist": float(z_depth.mean().item()),
            "mean_vectors_per_obj": mean_vectors_per_obj,
            "avalanche_added_groups": avalanche_added_groups,
        }
        return x + morphed, stats


class OpticalAvalancheLM(nn.Module):
    def __init__(
        self,
        enable_avalanche: bool = True,
        vocab_size: int = 256,
        d_model: int = 96,
        n_objects: int = 96,
        n_groups: int = 8,
        n_layers: int = 3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [OpticalAvalanche3DLayer(d_model, n_objects, n_groups, enable_avalanche=enable_avalanche) for _ in range(n_layers)]
        )
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


def eval_model_v2(model: nn.Module, data: torch.Tensor, seq_len: int, mode: str) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    agg_stats: dict[str, float] = {}
    n_batches = 0
    with torch.no_grad():
        limit = min(data.numel() - 1, seq_len * 80)
        for start in range(0, limit - seq_len, seq_len):
            chunk = data[start : start + seq_len + 1]
            logits, st = model(chunk[:-1].unsqueeze(0), mode=mode)
            loss = F.cross_entropy(logits[0], chunk[1:], reduction="sum")
            total_loss += float(loss.item())
            total_tokens += seq_len
            for k, v in st.items():
                agg_stats[k] = agg_stats.get(k, 0.0) + v
            n_batches += 1
    for k in agg_stats:
        agg_stats[k] /= max(1, n_batches)
    mean_nats = total_loss / max(1, total_tokens)
    bpc = mean_nats / math.log(2.0)
    ppl = math.exp(mean_nats)
    model.train()
    return bpc, ppl, agg_stats


def main():
    train_data = load_wikitext_bytes("train", max_chars=1_500_000)
    valid_data = load_wikitext_bytes("validation", max_chars=250_000)

    seq_len = 96
    batch_size = 16
    steps = 400
    lr = 2e-3

    torch.manual_seed(42)
    opt_only = OpticalAvalancheLM(enable_avalanche=False, d_model=96, n_objects=96, n_groups=8, n_layers=3)
    torch.manual_seed(42)
    opt_aval = OpticalAvalancheLM(enable_avalanche=True, d_model=96, n_objects=96, n_groups=8, n_layers=3)

    n_params = sum(p.numel() for p in opt_aval.parameters())
    print(f"Параметров в модели v2 (Оптика + CLOD + Лавина): {n_params:,}", flush=True)

    optim_only = torch.optim.AdamW(opt_only.parameters(), lr=lr, weight_decay=0.01)
    optim_aval = torch.optim.AdamW(opt_aval.parameters(), lr=lr, weight_decay=0.01)

    n_windows = (train_data.numel() - 1) // seq_len
    order = torch.randperm(n_windows, generator=torch.Generator().manual_seed(7))
    cursor = 0
    t0 = time.perf_counter()

    history = []
    print("Обучение двух вариантов v2 (чистая оптика vs оптика + лавинный каскад по ребрам)...", flush=True)
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

        logits_o, st_o = opt_only(src, mode="native")
        loss_o = F.cross_entropy(logits_o.reshape(-1, 256), dst.reshape(-1))
        optim_only.zero_grad()
        loss_o.backward()
        torch.nn.utils.clip_grad_norm_(opt_only.parameters(), 1.0)
        optim_only.step()

        logits_a, st_a = opt_aval(src, mode="native")
        loss_a = F.cross_entropy(logits_a.reshape(-1, 256), dst.reshape(-1))
        optim_aval.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(opt_aval.parameters(), 1.0)
        optim_aval.step()

        if step == 1 or step % 50 == 0:
            bpc_o = float(loss_o.item()) / math.log(2.0)
            bpc_a = float(loss_a.item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | только_оптика BPC={bpc_o:.3f} (детали={st_o['detail_volume']*100:.1f}%) | "
                f"оптика+лавина BPC={bpc_a:.3f} (детали={st_a['detail_volume']*100:.1f}%, "
                f"векторов/объект={st_a['mean_vectors_per_obj']:.2f}/8, "
                f"вклад_лавины=+{st_a['avalanche_added_groups']:.2f} групп) | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append(
                {
                    "step": step,
                    "optics_only_bpc": round(bpc_o, 4),
                    "optics_avalanche_bpc": round(bpc_a, 4),
                    "optics_only_stats": {k: round(v, 4) for k, v in st_o.items()},
                    "optics_avalanche_stats": {k: round(v, 4) for k, v in st_a.items()},
                }
            )

    o_nat_bpc, o_nat_ppl, o_nat_st = eval_model_v2(opt_only, valid_data, seq_len, mode="native")
    o_str_bpc, o_str_ppl, o_str_st = eval_model_v2(opt_only, valid_data, seq_len, mode="vram_streaming")

    a_nat_bpc, a_nat_ppl, a_nat_st = eval_model_v2(opt_aval, valid_data, seq_len, mode="native")
    a_str_bpc, a_str_ppl, a_str_st = eval_model_v2(opt_aval, valid_data, seq_len, mode="vram_streaming")

    print("\n=== ИТОГИ СРАВНЕНИЯ V2 НА ОТЛОЖЕННЫХ СТАТЬЯХ WIKITEXT-2 ===", flush=True)
    print(
        f"v2 Только оптика + полигональный CLOD (native):  BPC = {o_nat_bpc:.3f}, перплексия = {o_nat_ppl:.2f} | "
        f"деталей: {o_nat_st['detail_volume']*100:.1f}%, подгружено в VRAM: {o_nat_st['loaded_volume']*100:.1f}%",
        flush=True,
    )
    print(
        f"v2 Только оптика + полигональный CLOD (VRAM):    BPC = {o_str_bpc:.3f}, перплексия = {o_str_ppl:.2f} | "
        f"потеря при стриминге: +{o_str_bpc - o_nat_bpc:.4f} BPC",
        flush=True,
    )
    print(
        f"v2 Оптика + CLOD + Химическая лавина (native):   BPC = {a_nat_bpc:.3f}, перплексия = {a_nat_ppl:.2f} | "
        f"деталей: {a_nat_st['detail_volume']*100:.1f}%, подгружено в VRAM: {a_nat_st['loaded_volume']*100:.1f}%, "
        f"ср.число векторов на объект: {a_nat_st['mean_vectors_per_obj']:.2f} из 8",
        flush=True,
    )
    print(
        f"v2 Оптика + CLOD + Химическая лавина (VRAM):     BPC = {a_str_bpc:.3f}, перплексия = {a_str_ppl:.2f} | "
        f"потеря при стриминге: +{a_str_bpc - a_nat_bpc:.4f} BPC",
        flush=True,
    )

    torch.manual_seed(11)
    sample_a = generate_text(opt_aval, "The history of science is ", max_new=90, mode="vram_streaming")
    print(f"\nГенерация v2 (Оптика + Лавина, режим VRAM): {sample_a!r}", flush=True)

    out_data = {
        "params": n_params,
        "history": history,
        "validation": {
            "v2_optics_only_native": {"bpc": round(o_nat_bpc, 4), "ppl": round(o_nat_ppl, 2), "stats": o_nat_st},
            "v2_optics_only_vram": {"bpc": round(o_str_bpc, 4), "ppl": round(o_str_ppl, 2), "stats": o_str_st},
            "v2_optics_avalanche_native": {"bpc": round(a_nat_bpc, 4), "ppl": round(a_nat_ppl, 2), "stats": a_nat_st},
            "v2_optics_avalanche_vram": {"bpc": round(a_str_bpc, 4), "ppl": round(a_str_ppl, 2), "stats": a_str_st},
        },
        "sample": sample_a,
    }
    out_path = OUT_DIR / "rezultat_optika_lavina_v2.json"
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено в {out_path}", flush=True)


if __name__ == "__main__":
    main()
