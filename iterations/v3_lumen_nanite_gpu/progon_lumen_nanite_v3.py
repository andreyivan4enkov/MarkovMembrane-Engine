"""Версия 3 (v3): Полная архитектура Nanite (полигональное дерево огрубления векторов)
+ Трассировка лучей с перекрытием (Ray Tracing with Occlusion)
+ Технология глобального освещения Unreal Engine 5 Lumen (6-гранный Surface Cache + World-Space Radiosity Probes).

Ключевые механизмы v3:
1. Огрубление признака по логике полигонов (Nanite Hierarchical Cluster DAG / Edge Collapse):
   - В отличие от простого отсечения хвоста вектора, каждый признак состоит из K=8 микрограней v_0..v_7,
     собранных в бинарное дерево уровней детализации (как полигональные кластеры Nanite в UE5):
     * LOD 0 (1 вершина-силуэт, 12.5% VRAM): грубый родительский полигон P0 = среднее всех 8 микрограней.
     * LOD 1 (2 крупных полигона, 25% VRAM): P0 + 1 вектор расщепления левой/правой половины меша.
     * LOD 2 (4 квада, 50% VRAM): + 2 вектора расщепления четвертей меша.
     * LOD 3 (8 микрограней, 100% VRAM): + 4 вектора тонкой микрогеометрии.
   - Когда объект далеко, он НЕ теряет связь со своими гранями — все его 8 граней сжаты в 1 вершинную нормаль P0
     (Exact Geometric Mesh Simplification). По мере приближения экранная ошибка (Screen-Space Error) превышает порог,
     и родительский полигон плавно (или ступенчато в VRAM) расщепляется на дочерние грани!

2. Активации как трассировка лучей (Ray/Cone Tracing + BVH Sphere Intersection + Beer-Lambert Occlusion):
   - Контекст выпускает направленный конус лучей из точки камеры r_cam вдоль вектора взгляда d_ray.
   - Для каждого 3D-объекта j вычисляется расстояние вдоль луча t_hit, прицельный промах b_perp
     и аналитическое пересечение конуса со сферой объекта R_j.
   - Перекрытие (Occlusion / Shadowing): объекты, стоящие ближе вдоль луча (малый t_hit и малый b_perp),
     поглощают часть энергии луча (закон Бугера-Ламберта-Бера), отбрасывая настоящую 3D-тень на объекты позади них!

3. Технология Unreal Lumen (6-Card Surface Cache + World-Space Radiosity Probes):
   - Surface Cache (Кэш поверхностей): у каждого объекта есть 6 ориентированных карточек поверхности
     (+X, -X, +Y, -Y, +Z, -Z), которые накапливают падающее излучение между кадрами без полного пересчета.
   - Radiosity Probes (Световые зонды в 3D-мире): в пространстве стоят 8 зондов глобального освещения,
     которые за O(N) собирают переотраженный свет (Indirect Bounce) с освещенных карточек Surface Cache
     и подсвечивают связанные объекты, даже если они находятся в тени или вне прямого луча!
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
from progon_optika_lavina_v2 import eval_model_v2

OUT_DIR = Path(__file__).resolve().parent


def build_haar_nanite_basis(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Строит ортогональный базис дерева огрубления полигонов (8 -> 4 -> 2 -> 1, аналог вейвлетов Хаара / Nanite DAG):
    - Индекс 0: LOD 0 (средний вектор-силуэт всего 8-гранного меша)
    - Индекс 1: LOD 1 (разлом меша на 2 крупные грани)
    - Индексы 2..3: LOD 2 (разлом на 4 квада)
    - Индексы 4..7: LOD 3 (4 детальных ребра для всех 8 микрограней)
    Это гарантирует, что даже на самом грубом уровне (LOD 0, 1 вектор) в точке честно представлена сумма всех граней,
    а добавление уровней 1..7 уточняет геометрию без потери информации!
    """
    h = torch.zeros(8, 8, device=device, dtype=dtype)
    # LOD 0: 1 общий вектор
    h[0, :] = 1.0 / math.sqrt(8.0)
    # LOD 1: 1 вектор диполя (левая vs правая половина полигона)
    h[1, :4] = 1.0 / math.sqrt(8.0)
    h[1, 4:] = -1.0 / math.sqrt(8.0)
    # LOD 2: 2 вектора квадов
    h[2, 0:2] = 0.5
    h[2, 2:4] = -0.5
    h[3, 4:6] = 0.5
    h[3, 6:8] = -0.5
    # LOD 3: 4 вектора тонких ребер
    for m in range(4):
        h[4 + m, 2 * m] = 1.0 / math.sqrt(2.0)
        h[4 + m, 2 * m + 1] = -1.0 / math.sqrt(2.0)
    return h  # (8, 8), ортонормированная матрица H @ H.T = I


class LumenNanite3DLayer(nn.Module):
    """
    Слой v3:
    - Nanite DAG огрубления полигонов (базис 1 -> 2 -> 4 -> 8 граней по Screen-Space Error),
    - Трассировка конуса лучей с аналитическим пересечением сфер и 3D-перекрытием (Occlusion),
    - Unreal Lumen: 6-гранный Surface Cache на объектах + 8 World-Space Radiosity Probes для глобального освещения (GI).
    """

    def __init__(self, d_model: int, n_objects: int, n_groups: int = 8):
        super().__init__()
        assert n_groups == 8 and d_model % 8 == 0
        self.d_model = d_model
        self.n_objects = n_objects
        self.n_groups = n_groups
        self.group_dim = d_model // n_groups

        self.layer_norm = nn.LayerNorm(d_model)

        # Источник лучей (камера контекста): точка испускания r_orig, направление луча d_ray и угол раскрытия конуса
        self.to_ray_orig = nn.Linear(d_model, 3)
        self.to_ray_dir = nn.Linear(d_model, 3)
        self.to_cone_params = nn.Linear(d_model, 3)  # [cone_angle, error_threshold, gi_bounce_strength]

        # 3D-объекты сцены (полигональные кластеры Nanite):
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.3)
        self.obj_log_radius = nn.Parameter(torch.zeros(n_objects))
        # Альбедо 6 граней Surface Cache каждого объекта (+X, -X, +Y, -Y, +Z, -Z)
        self.obj_card_albedo = nn.Parameter(torch.ones(n_objects, 6) * 0.5)

        # Полигональные векторы признаков (хранятся сразу в иерархическом базисе Nanite: 0=силуэт, 1=LOD1, 2..3=LOD2, 4..7=LOD3)
        self.obj_in = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))

        # 8 зондов глобального освещения Lumen (World-Space Radiosity Probes в вершинах 3D-куба)
        probe_coords = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("probe_pos", probe_coords * 1.2)
        self.register_buffer("nanite_basis", build_haar_nanite_basis(torch.device("cpu"), torch.float32))

        self.cache_decay = nn.Parameter(torch.tensor(0.65))
        self.gi_transfer_gain = nn.Parameter(torch.tensor(0.45))

    def get_card_weights(self, direction: torch.Tensor) -> torch.Tensor:
        """
        Разлагает 3D-единичный вектор направления по 6 карточкам Lumen Surface Cache (+X, -X, +Y, -Y, +Z, -Z).
        direction: (..., 3) -> возвращает (..., 6), сумма квадратов по активным граням = 1.
        """
        px = F.relu(direction[..., 0])
        nx = F.relu(-direction[..., 0])
        py = F.relu(direction[..., 1])
        ny = F.relu(-direction[..., 1])
        pz = F.relu(direction[..., 2])
        nz = F.relu(-direction[..., 2])
        cards = torch.stack([px, nx, py, ny, pz, nz], dim=-1)
        return cards / (cards.sum(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        b, t, d = x.shape
        h = self.layer_norm(x)

        # 1. ИСПУСКАНИЕ ЛУЧЕЙ КОНТЕКСТА (Ray Tracing + Temporal Ray Guiding)
        r_orig = self.to_ray_orig(h)  # (B, T, 3)
        raw_dir = self.to_ray_dir(h)
        # Складываем собственное направление луча со сдвигом камеры во времени
        motion_dir = torch.zeros_like(r_orig)
        motion_dir[:, 1:, :] = r_orig[:, 1:, :] - r_orig[:, :-1, :]
        motion_dir[:, 0, :] = r_orig[:, 0, :]
        ray_dir = F.normalize(raw_dir + 0.5 * motion_dir, dim=-1, eps=1e-6)  # (B, T, 3)

        ctrl = self.to_cone_params(h)  # (B, T, 3)
        cone_tan = 0.15 + 0.35 * torch.sigmoid(ctrl[:, :, 0:1])  # тангенс угла раскрытия конуса трассировки
        err_scale = 0.6 + 0.8 * torch.sigmoid(ctrl[:, :, 1:2])  # чувствительность к экранной ошибке Nanite
        gi_strength = 0.3 + 0.7 * torch.sigmoid(ctrl[:, :, 2:3])  # сила непрямого освещения Lumen GI

        # 2. АНАЛИТИЧЕСКАЯ ТРАССИРОВКА КОНУСА ЛУЧЕЙ И ПЕРЕКРЫТИЕ (OCCLUSION / SHADOWING)
        # Вектор от точки испускания луча к центру каждого 3D-объекта: (B, T, N, 3)
        to_obj = self.obj_pos.view(1, 1, self.n_objects, 3) - r_orig.unsqueeze(2)
        dist2 = (to_obj * to_obj).sum(dim=-1) + 1e-6
        dist = torch.sqrt(dist2)

        # Проекция вдоль луча t_hit и прицельное расстояние (промах луча мимо центра сферы) b_perp
        t_hit = (to_obj * ray_dir.unsqueeze(2)).sum(dim=-1)  # (B, T, N)
        t_pos = F.softplus(t_hit) + 0.25  # эффективная дистанция вдоль луча в передней полусфере
        in_front = torch.sigmoid(3.5 * t_hit / (dist.mean(dim=-1, keepdim=True).detach() + 1e-4))

        # Перпендикулярное отклонение центра объекта от оси луча
        b_perp2 = (dist2 - t_hit * t_hit).clamp_min(0.0)
        b_perp = torch.sqrt(b_perp2 + 1e-6)

        # Радиус сферы объекта и радиус конуса луча на глубине t_pos (Cone Tracing из Lumen)
        raw_r = F.softplus(self.obj_log_radius) + 0.25
        obj_radius = 0.55 * raw_r / (raw_r.mean() + 1e-6)  # закон сохранения объема объектов: (N,)
        cone_radius = cone_tan * t_pos + obj_radius.view(1, 1, -1)

        # Доля пересечения конуса луча со сферой объекта (аналитическое пересечение Cone-Sphere):
        hit_overlap = torch.exp(-0.5 * (b_perp / cone_radius).pow(2)) * in_front

        # ОПТИЧЕСКОЕ ПЕРЕКРЫТИЕ (Occlusion / Z-Buffer Shadowing):
        # Объекты, попавшие в конус луча на малой глубине t_pos, ослабляют луч для более далеких объектов!
        # Сортируем объекты по глубине t_pos вдоль луча и считаем накопленную оптическую толщину (закон Бера)
        sorted_t, sort_idx = torch.sort(t_pos, dim=-1)
        sorted_hit = torch.gather(hit_overlap, -1, sort_idx)
        # Накопленная тень от впереди стоящих объектов на том же конусе луча
        optical_depth = torch.cumsum(sorted_hit * 0.35, dim=-1) - sorted_hit * 0.35
        transmittance_sorted = torch.exp(-optical_depth)
        transmittance = torch.zeros_like(transmittance_sorted).scatter(-1, sort_idx, transmittance_sorted)

        # Прямая освещенность объекта лучом контекста (Direct Ray Illumination)
        mean_d2 = dist2.mean(dim=-1, keepdim=True).detach().clamp_min(0.5)
        direct_radiance = hit_overlap * transmittance / (1.0 + (t_pos * t_pos) / mean_d2)

        # 3. ТЕХНОЛОГИЯ UNREAL LUMEN: SURFACE CACHE (6 КАРТОЧЕК) + WORLD-SPACE RADIOSITY PROBES
        # Направление падающего луча на поверхность каждого объекта: (B, T, N, 3)
        inc_dir = -to_obj / dist.unsqueeze(-1)
        card_weights = self.get_card_weights(inc_dir)  # (B, T, N, 6)
        albedo = torch.sigmoid(self.obj_card_albedo).view(1, 1, self.n_objects, 6)

        # Мгновенная засветка 6 карточек поверхности от прямого луча
        instant_cards = direct_radiance.unsqueeze(-1) * card_weights * albedo  # (B, T, N, 6)

        # Геометрия связи между N объектами и P=8 зондами глобального освещения Lumen (вычисляется за O(8*N))
        # Vec от объекта j к зонду p: (N, 8, 3)
        obj_to_probe = self.probe_pos.unsqueeze(0) - self.obj_pos.unsqueeze(1)
        op_dist2 = (obj_to_probe * obj_to_probe).sum(dim=-1) + 0.25  # (N, 8)
        op_dir = obj_to_probe / torch.sqrt(op_dist2).unsqueeze(-1)
        # Какие карточки объекта j смотрят в сторону зонда p: (N, 8, 6)
        obj_to_probe_cards = self.get_card_weights(op_dir)
        # Матрица переноса излучения от карточек объектов к 8 зондам: (N, 6, 8)
        to_probe_kernel = (obj_to_probe_cards / op_dist2.unsqueeze(-1)).permute(0, 2, 1)
        to_probe_flat = to_probe_kernel.reshape(self.n_objects * 6, 8)
        to_probe_flat = to_probe_flat / (to_probe_flat.sum(dim=0, keepdim=True) + 1e-6)

        # И обратно: от 8 зондов на 6 карточек каждого объекта (принимают грани, обращенные к зонду, т.е. -op_dir)
        probe_to_obj_cards = self.get_card_weights(-op_dir)  # (N, 8, 6)
        from_probe_kernel = (probe_to_obj_cards / op_dist2.unsqueeze(-1)).permute(1, 0, 2)  # (8, N, 6)
        from_probe_flat = from_probe_kernel.reshape(8, self.n_objects * 6)
        from_probe_flat = from_probe_flat / (from_probe_flat.sum(dim=0, keepdim=True) + 1e-6)

        # Обновление Lumen Surface Cache во времени t-1 -> t и раздача непрямого освещения (Multi-Bounce GI)
        decay = 0.4 + 0.5 * torch.sigmoid(self.cache_decay)
        surface_cache = torch.zeros_like(instant_cards)  # (B, T, N, 6)
        cache_state = torch.zeros(b, self.n_objects, 6, device=x.device, dtype=x.dtype)
        for step_t in range(t):
            # 1) Собираем свет с текущего Surface Cache на 8 мировых зондов Radiosity: (B, N*6) @ (N*6, 8) -> (B, 8)
            probe_irradiance = torch.matmul(cache_state.reshape(b, self.n_objects * 6), to_probe_flat)
            # 2) Зонды раздают непрямой свет (GI Bounce) обратно на карточки всех объектов: (B, 8) @ (8, N*6) -> (B, N, 6)
            gi_bounce = torch.matmul(probe_irradiance, from_probe_flat).reshape(b, self.n_objects, 6) * albedo.squeeze(1)
            # 3) Полный кэш поверхности на кадре step_t = накопленный кэш + непрямой свет зондов + прямой луч
            cur_cache = decay * cache_state + (1.0 - decay) * (instant_cards[:, step_t] + gi_strength[:, step_t].unsqueeze(-1) * gi_bounce)
            surface_cache[:, step_t] = cur_cache
            cache_state = cur_cache

        # Считываем освещенность объекта со стороны камеры (видимые карточки Surface Cache)
        indirect_radiance = (surface_cache * card_weights).sum(dim=-1)  # (B, T, N)
        total_radiance = direct_radiance + gi_strength * indirect_radiance

        # Закон сохранения энергии освещенности кадра
        mean_rad = total_radiance.mean(dim=-1, keepdim=True).detach().clamp_min(0.12)
        norm_radiance = 0.35 * total_radiance / mean_rad

        # 4. NANITE CLOD: ВЫБОР ГЛУБИНЫ РАЗБИЕНИЯ ПОЛИГОНОВ ПО SCREEN-SPACE ERROR
        # Экранная ошибка проекции кластера: пропорциональна (R_j / t_pos) * освещенность кластера
        screen_error = err_scale * norm_radiance * (obj_radius.view(1, 1, -1) / (t_pos / mean_d2.sqrt() + 0.3))

        # Уровни дерева полигонов Nanite (4 ступени: 1 -> 2 -> 4 -> 8 векторов):
        # Вектор 0 (LOD 0, точка-силуэт): порог ошибки = 0 (включен всегда)
        # Вектор 1 (LOD 1, 2 грани): включается при screen_error > 0.20
        # Векторы 2..3 (LOD 2, 4 квада): включаются при screen_error > 0.45
        # Векторы 4..7 (LOD 3, 8 микрограней): включаются при screen_error > 0.85
        lod_thresholds = torch.tensor([0.0, 0.20, 0.45, 0.45, 0.85, 0.85, 0.85, 0.85], device=x.device, dtype=x.dtype).view(1, 1, 1, 8)
        soft_lod = torch.sigmoid((screen_error.unsqueeze(-1) - lod_thresholds) * 8.0)
        soft_lod = soft_lod.clone()
        soft_lod[..., 0] = 1.0

        hard_lod = (screen_error.unsqueeze(-1) >= lod_thresholds).to(x.dtype)
        hard_lod[..., 0] = 1.0

        lod_weights = hard_lod if mode == "vram_streaming" else (hard_lod.detach() - soft_lod.detach() + soft_lod)

        # 5. ИЕРАРХИЧЕСКОЕ ПРЕОБРАЗОВАНИЕ ПРИЗНАКА ЧЕРЕЗ БАЗИС ПОЛИГОНОВ NANITE
        # Переводим входной вектор контекста в иерархический базис полигонов (LOD0..LOD3)
        h_raw_groups = h.view(b, t, self.n_groups, self.group_dim)  # (B, T, 8, G)
        # Применяем ортогональное дерево огрубления Хаара-Nanite по оси групп (8x8)
        h_nanite = torch.einsum("km,btmg->btkg", self.nanite_basis, h_raw_groups)

        h_perm = h_nanite.permute(2, 0, 1, 3).reshape(self.n_groups, b * t, self.group_dim)
        in_perm = self.obj_in.permute(1, 2, 0)
        group_dots = torch.bmm(h_perm, in_perm).permute(1, 2, 0).reshape(b, t, self.n_objects, self.n_groups)

        envelope = norm_radiance / (1.0 + norm_radiance)
        w_active = envelope.unsqueeze(-1) * lod_weights
        activated = F.silu(group_dots) * w_active

        # Перенос косвенного света через Surface Cache модулирует базовый силуэт (LOD 0..1)
        gi_boost = (indirect_radiance / (total_radiance + 1e-6)).unsqueeze(-1)
        activated = activated * (1.0 + torch.tanh(self.gi_transfer_gain) * gi_boost * lod_weights)

        act_perm = activated.permute(3, 0, 1, 2).reshape(self.n_groups, b * t, self.n_objects)
        out_perm = self.obj_out.permute(1, 0, 2)
        morphed_nanite = torch.bmm(act_perm, out_perm).permute(1, 0, 2).reshape(b, t, self.n_groups, self.group_dim)

        # Обратное преобразование из дерева полигонов Nanite в пространство признаков
        morphed = torch.einsum("km,btkg->btmg", self.nanite_basis, morphed_nanite).reshape(b, t, d)

        with torch.no_grad():
            near_objects_frac = float((hard_lod[..., -1] > 0.5).float().mean().item())
            detail_volume = float((hard_lod[..., 1:] > 0.5).float().mean().item())
            loaded_volume = float(hard_lod.float().mean().item())
            mean_vectors_per_obj = float(hard_lod.sum(dim=-1).mean().item())
            gi_share = float((indirect_radiance / (total_radiance + 1e-6)).mean().item())
            shadow_occluded = float((1.0 - transmittance).mean().item())

        stats = {
            "near_objects_frac": near_objects_frac,
            "detail_volume": detail_volume,
            "loaded_volume": loaded_volume,
            "mean_dist": float(t_pos.mean().item()),
            "mean_vectors_per_obj": mean_vectors_per_obj,
            "gi_share": gi_share,
            "shadow_occluded": shadow_occluded,
        }
        return x + morphed, stats


class LumenNaniteLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 96, n_objects: int = 96, n_groups: int = 8, n_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([LumenNanite3DLayer(d_model, n_objects, n_groups) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        x = self.embed(tokens)
        keys = (
            "near_objects_frac",
            "detail_volume",
            "loaded_volume",
            "mean_dist",
            "mean_vectors_per_obj",
            "gi_share",
            "shadow_occluded",
        )
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
    model = LumenNaniteLM(d_model=96, n_objects=96, n_groups=8, n_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели v3 (Nanite DAG + Ray Tracing Occlusion + Unreal Lumen GI): {n_params:,}", flush=True)

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
                f"шаг {step:3d} | v3_Lumen_Nanite BPC={bpc:.3f} | детали={st['detail_volume']*100:.1f}% | "
                f"VRAM={st['loaded_volume']*100:.1f}% | векторов/объект={st['mean_vectors_per_obj']:.2f}/8 | "
                f"Lumen_GI={st['gi_share']*100:.1f}% | тени={st['shadow_occluded']*100:.1f}% | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append({"step": step, "bpc": round(bpc, 4), **{k: round(v, 4) for k, v in st.items()}})

    nat_bpc, nat_ppl, nat_st = eval_model_v2(model, valid_data, seq_len, mode="native")
    str_bpc, str_ppl, str_st = eval_model_v2(model, valid_data, seq_len, mode="vram_streaming")
    print("\n=== ИТОГИ V3 (NANITE + RAY TRACING + UNREAL LUMEN GI) ===", flush=True)
    print(
        f"v3 Lumen+Nanite (native):        BPC = {nat_bpc:.3f}, перплексия = {nat_ppl:.2f} | "
        f"деталей: {nat_st['detail_volume']*100:.1f}%, VRAM: {nat_st['loaded_volume']*100:.1f}%, "
        f"векторов на объект: {nat_st['mean_vectors_per_obj']:.2f}/8, вклад Lumen GI: {nat_st['gi_share']*100:.1f}%",
        flush=True,
    )
    print(
        f"v3 Lumen+Nanite (VRAM-стриминг): BPC = {str_bpc:.3f}, перплексия = {str_ppl:.2f} | "
        f"потеря при стриминге: +{str_bpc - nat_bpc:.4f} BPC",
        flush=True,
    )

    torch.manual_seed(11)
    sample = generate_text(model, "The history of science is ", max_new=90, mode="vram_streaming")
    print(f"Генерация v3 (VRAM-стриминг): {sample!r}", flush=True)

    out_path = OUT_DIR / "rezultat_lumen_nanite_v3.json"
    out_path.write_text(
        json.dumps(
            {
                "params": n_params,
                "history": history,
                "validation": {
                    "v3_native": {"bpc": round(nat_bpc, 4), "ppl": round(nat_ppl, 2), "stats": nat_st},
                    "v3_vram": {"bpc": round(str_bpc, 4), "ppl": round(str_ppl, 2), "stats": str_st},
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
