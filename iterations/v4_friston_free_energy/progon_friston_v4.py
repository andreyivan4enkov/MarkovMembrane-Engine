"""Версия 4 (v4): Разделение модели на две части по принципу свободной энергии Карла Фристона
(Free Energy Principle / Predictive Coding / Active Inference) поверх 3D-архитектуры Lumen + Nanite.

Две строго разделенные части системы и 2D-мембрана коллизии (Одеяло Маркова / Markov Blanket):

ЧАСТЬ 1: ВНУТРЕННИЙ МИР СИСТЕМЫ (Generative Model & Internal States mu)
  - Состоит из обученных 3D-объектов (полигональное дерево Nanite + кэш поверхностей Lumen Surface Cache)
    и внутреннего латентного убеждения системы mu(t), которое живет по своим внутренним законам инерции:
      mu_prior(t) = Transition(mu(t-1), Lumen_GI_Cache(t-1)).
  - Внутренний мир САМ (сверху вниз, Top-Down) генерирует ожидаемую проекцию на 2D-мембране:
      s_hat(t) = TopDown_Render(mu_prior(t), 3D_Objects_LOD0).
    Причем базовый прогноз строится дешево — через силуэты объектов (LOD 0) и накопленный свет Lumen!

ЧАСТЬ 2: ВНЕШНЯЯ СРЕДА (External Environment / Context Stream s)
  - Это реальный входящий поток контекста s(t), который существует снаружи и бьет в 2D-мембрану системы.

2D-МЕМБРАНА КОЛЛИЗИИ = ОДЕЯЛО МАРКОВА (Markov Blanket):
  - На мембране встречаются внутреннее ожидание s_hat(t) и внешний контекст s(t).
  - Вычисляется взвешенная по точности (Precision Pi) ошибка предсказания (Prediction Error / Сюрприз):
      eps(t) = Pi(t) * (s(t) - s_hat(t)),   где Pi(t) = exp(-log_var(t)).
  - Вариационная Свободная Энергия Фристона F(t):
      F(t) = 0.5 * || (s(t) - s_hat(t)) * sqrt(Pi(t)) ||^2 - 0.5 * log|Pi(t)| + D_KL(mu_post(t) || mu_prior(t)).

КАК СВОБОДНАЯ ЭНЕРГИЯ УПРАВЛЯЕТ 3D-ДЕТАЛИЗАЦИЕЙ (NANITE LOD) И АКТИВНЫМ ВЫВОДОМ:
  1. Экономия на предсказуемом (Zero-Work on Predicted Context):
     Если внутренний мир угадал внешний сигнал в области объекта j (малая ошибка предсказания |eps_j|),
     то свободная энергия там минимальна -> глубокие полигоны Nanite (LOD 1..3) НЕ РАСПАКОВЫВАЮТСЯ ВООБЩЕ!
     Объект остается точкой ожидания (LOD 0).
  2. Распаковка полигонов только от Сюрприза (Surprise-Driven Nanite Tessellation):
     Глубокие векторные уровни k = 1..7 объекта j открываются ТОЛЬКО пропорционально потоку ошибки предсказания,
     пробившему 2D-мембрану в направлении этого объекта!
  3. Активный вывод (Active Inference — поворот камеры взгляда):
     Луч камеры контекста в 3D поворачивается по градиенту свободной энергии в сторону максимальной невязки,
     чтобы погасить ошибку предсказания и обновить внутреннее состояние:
       mu_post(t) = mu_prior(t) + K_kalman * BottomUp_Error_Correction(eps(t), Nanite_Opened_Polygons).
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
from progon_lumen_nanite_v3 import build_haar_nanite_basis

OUT_DIR = Path(__file__).resolve().parent


class FristonMarkovBlanket3DLayer(nn.Module):
    """
    Слой v4:
    Разделен на:
      1) Внутренний генеративный 3D-мир (Internal World: mu_state, 3D-объекты Nanite, Lumen Surface Cache, Top-Down генератор),
      2) Внешний поток среды (External Context: сенсорный вход s_ext),
      3) 2D-мембрану коллизии (Markov Blanket: невязка eps = s_ext - s_hat, точность Pi, свободная энергия F).
    """

    def __init__(self, d_model: int, n_objects: int, n_groups: int = 8):
        super().__init__()
        assert n_groups == 8 and d_model % 8 == 0
        self.d_model = d_model
        self.n_objects = n_objects
        self.n_groups = n_groups
        self.group_dim = d_model // n_groups

        # --- ГРАНИЦА (ОДЕЯЛО МАРКОВА / СЕНСОРНАЯ 2D-МЕМБРАНА) ---
        self.ext_norm = nn.LayerNorm(d_model)
        self.int_norm = nn.LayerNorm(d_model)
        # Оценка логарифма точности (Precision Pi = exp(log_pi)) на 2D-мембране
        self.to_log_precision = nn.Linear(d_model, n_groups)

        # --- ЧАСТЬ 1: ВНУТРЕННИЙ МИР СИСТЕМЫ (INTERNAL GENERATIVE WORLD) ---
        # Внутренняя инерционная динамика состояния mu(t-1) -> mu_prior(t)
        self.internal_transition = nn.Linear(d_model, d_model)
        # Top-Down управление камерой ожидания из внутреннего состояния mu_prior
        self.mu_to_ray_orig = nn.Linear(d_model, 3)
        self.mu_to_ray_dir = nn.Linear(d_model, 3)
        # Active Inference: коррекция луча камеры от вектора ошибки предсказания eps на 2D-мембране
        self.err_to_ray_shift = nn.Linear(d_model, 3)

        # 3D-объекты внутреннего мира (хранятся в полигональном базисе Nanite: 0=силуэт, 1..7=грани расщепления)
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.3)
        self.obj_log_radius = nn.Parameter(torch.zeros(n_objects))
        self.obj_card_albedo = nn.Parameter(torch.ones(n_objects, 6) * 0.5)

        # Top-Down генеративные веса (строят прогноз s_hat из внутреннего состояния mu)
        # и Bottom-Up веса усвоения ошибки предсказания (обновляют mu_prior -> mu_post только по открытым полигонам!)
        self.obj_in = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, n_groups, self.group_dim) / math.sqrt(d_model))

        # 8 мировых зондов Lumen Radiosity Probes во внутреннем 3D-мире
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

        self.mu_inertia = nn.Parameter(torch.tensor(0.60))
        self.cache_decay = nn.Parameter(torch.tensor(0.65))
        self.kalman_gain = nn.Parameter(torch.tensor(0.85))

    def get_card_weights(self, direction: torch.Tensor) -> torch.Tensor:
        px = F.relu(direction[..., 0])
        nx = F.relu(-direction[..., 0])
        py = F.relu(direction[..., 1])
        ny = F.relu(-direction[..., 1])
        pz = F.relu(direction[..., 2])
        nz = F.relu(-direction[..., 2])
        cards = torch.stack([px, nx, py, ny, pz, nz], dim=-1)
        return cards / (cards.sum(dim=-1, keepdim=True) + 1e-6)

    def forward(
        self, s_ext: torch.Tensor, mu_state: torch.Tensor, mode: str = "native"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Вход:
          s_ext: (B, T, D) — ВНЕШНЯЯ СРЕДА (реальный входящий поток контекста на данном уровне)
          mu_state: (B, T, D) — ВНУТРЕННЕЕ СОСТОЯНИЕ СИСТЕМЫ (убеждения внутреннего мира)
        Выход:
          s_ext_next: (B, T, D) — уточненный сигнал среды для следующего уровня
          mu_post: (B, T, D) — обновленное внутреннее состояние после минимизации свободной энергии
          free_energy_loss: скаляр вариационной свободной энергии Фристона F для обучения генеративной модели
          stats: метрики свободной энергии, ошибки предсказания и числа открытых полигонов Nanite
        """
        b, t, d = s_ext.shape

        # Сенсорное состояние внешней среды на 2D-мембране (Одеяле Маркова)
        s_sensory = self.ext_norm(s_ext)  # (B, T, D)

        # =====================================================================
        # ШАГ 1: ВНУТРЕННИЙ МИР СТРОИТ АПРИОРНОЕ СОСТОЯНИЕ mu_prior(t) И TOP-DOWN ПРОГНОЗ s_hat(t)
        # (ВАЖНО: при предсказании шага t внутренний мир опирается на прошлое состояние mu(t-1),
        #  то есть ЧЕСТНО пытается предсказать новый кадр внешней среды ДО того, как подстроится под него!)
        # =====================================================================
        mu_prev = torch.zeros_like(mu_state)
        mu_prev[:, 1:, :] = mu_state[:, :-1, :]
        mu_prev[:, 0, :] = mu_state[:, 0, :]

        inertia = torch.sigmoid(self.mu_inertia)
        mu_prior = self.int_norm(inertia * mu_prev + (1.0 - inertia) * self.internal_transition(mu_prev))  # (B, T, D)

        # Базовый Top-Down прогноз строится дешево: только через вершины-силуэты LOD 0 (k=0) всех 3D-объектов!
        mu_groups = torch.einsum("km,btmg->btkg", self.nanite_basis, mu_prior.view(b, t, self.n_groups, self.group_dim))
        # Берем только грубый силуэт LOD 0 (k=0): (B, T, G) @ (G, N) -> (B, T, N)
        lod0_in = self.obj_in[:, 0, :]  # (N, G)
        lod0_out = self.obj_out[:, 0, :]  # (N, G)
        prior_dots_lod0 = torch.matmul(mu_groups[:, :, 0, :], lod0_in.T)  # (B, T, N)
        prior_act_lod0 = F.silu(prior_dots_lod0) / math.sqrt(self.n_objects)
        # Прогнозируемый силуэт на 2D-мембране в базисе Nanite (заполняет k=0, а при предсказуемой среде мелкие грани = 0)
        s_hat_lod0 = torch.matmul(prior_act_lod0, lod0_out)  # (B, T, G)
        s_hat_nanite = torch.zeros(b, t, self.n_groups, self.group_dim, device=s_ext.device, dtype=s_ext.dtype)
        s_hat_nanite[:, :, 0, :] = s_hat_lod0
        s_hat = mu_prior + torch.einsum("km,btkg->btmg", self.nanite_basis, s_hat_nanite).reshape(b, t, d)

        # =====================================================================
        # ШАГ 2: 2D-КОЛЛИЗИЯ НА ОДЕЯЛЕ МАРКОВА — ВЫЧИСЛЕНИЕ ОШИБКИ ПРЕДСКАЗАНИЯ И СВОБОДНОЙ ЭНЕРГИИ F
        # =====================================================================
        # Разница между внешней средой s_sensory и внутренним прогнозом s_hat:
        pred_error_raw = s_sensory - s_hat  # (B, T, D) — вектор невязки на 2D-мембране!
        pred_error_nanite = torch.einsum(
            "km,btmg->btkg", self.nanite_basis, pred_error_raw.view(b, t, self.n_groups, self.group_dim)
        )  # (B, T, 8, G)

        # Точность (Precision Pi = exp(log_pi), обратная дисперсия/температура по Фристону)
        log_pi = torch.tanh(self.to_log_precision(mu_prior)) * 1.5  # (B, T, 8) в [-1.5, 1.5]
        precision = torch.exp(log_pi).unsqueeze(-1)  # (B, T, 8, 1)

        # Взвешенная по точности ошибка предсказания: eps = Pi * (s - s_hat)
        weighted_error = precision * pred_error_nanite  # (B, T, 8, G)

        # Локальная плотность сюрприза (Surprise Energy) на каждом такте:
        surprise_per_group = 0.5 * (precision * pred_error_nanite.pow(2)).mean(dim=-1) - 0.5 * log_pi  # (B, T, 8)
        surprise_magnitude = pred_error_nanite.norm(dim=-1) / math.sqrt(self.group_dim)  # (B, T, 8)

        # =====================================================================
        # ШАГ 3: АКТИВНЫЙ ВЫВОД (ACTIVE INFERENCE) — ЛУЧ КАМЕРЫ ПОВОРАЧИВАЕТСЯ В СТОРОНУ ОШИБКИ ПРЕДСКАЗАНИЯ!
        # =====================================================================
        r_orig = self.mu_to_ray_orig(mu_prior)  # точка камеры из внутреннего состояния
        # Направление луча = куда смотрел внутренний мир + доворот в сторону невязки pred_error_raw!
        ray_dir = F.normalize(
            self.mu_to_ray_dir(mu_prior) + 0.65 * self.err_to_ray_shift(pred_error_raw), dim=-1, eps=1e-6
        )

        # Трассировка конуса луча к 3D-объектам внутреннего мира + перекрытие (Z-Occlusion)
        to_obj = self.obj_pos.view(1, 1, self.n_objects, 3) - r_orig.unsqueeze(2)
        dist2 = (to_obj * to_obj).sum(dim=-1) + 1e-6
        dist = torch.sqrt(dist2)

        t_hit = (to_obj * ray_dir.unsqueeze(2)).sum(dim=-1)
        t_pos = F.softplus(t_hit) + 0.25
        in_front = torch.sigmoid(3.5 * t_hit / (dist.mean(dim=-1, keepdim=True).detach() + 1e-4))
        b_perp = torch.sqrt((dist2 - t_hit * t_hit).clamp_min(0.0) + 1e-6)

        raw_r = F.softplus(self.obj_log_radius) + 0.25
        obj_radius = 0.55 * raw_r / (raw_r.mean() + 1e-6)
        cone_radius = 0.28 * t_pos + obj_radius.view(1, 1, -1)
        hit_overlap = torch.exp(-0.5 * (b_perp / cone_radius).pow(2)) * in_front

        # Z-перекрытие (тени вдоль луча)
        sorted_t, sort_idx = torch.sort(t_pos, dim=-1)
        sorted_hit = torch.gather(hit_overlap, -1, sort_idx)
        optical_depth = torch.cumsum(sorted_hit * 0.35, dim=-1) - sorted_hit * 0.35
        trans_sorted = torch.exp(-optical_depth)
        transmittance = torch.zeros_like(trans_sorted).scatter(-1, sort_idx, trans_sorted)

        mean_d2 = dist2.mean(dim=-1, keepdim=True).detach().clamp_min(0.5)
        direct_radiance = hit_overlap * transmittance / (1.0 + (t_pos * t_pos) / mean_d2)

        # Lumen Surface Cache (6 граней) + 8 мировых зондов Radiosity
        inc_dir = -to_obj / dist.unsqueeze(-1)
        card_weights = self.get_card_weights(inc_dir)
        albedo = torch.sigmoid(self.obj_card_albedo).view(1, 1, self.n_objects, 6)
        instant_cards = direct_radiance.unsqueeze(-1) * card_weights * albedo

        obj_to_probe = self.probe_pos.unsqueeze(0) - self.obj_pos.unsqueeze(1)
        op_dist2 = (obj_to_probe * obj_to_probe).sum(dim=-1) + 0.25
        op_dir = obj_to_probe / torch.sqrt(op_dist2).unsqueeze(-1)
        to_probe_flat = (self.get_card_weights(op_dir) / op_dist2.unsqueeze(-1)).permute(0, 2, 1).reshape(self.n_objects * 6, 8)
        to_probe_flat = to_probe_flat / (to_probe_flat.sum(dim=0, keepdim=True) + 1e-6)
        from_probe_flat = (self.get_card_weights(-op_dir) / op_dist2.unsqueeze(-1)).permute(1, 0, 2).reshape(8, self.n_objects * 6)
        from_probe_flat = from_probe_flat / (from_probe_flat.sum(dim=0, keepdim=True) + 1e-6)

        decay = 0.4 + 0.5 * torch.sigmoid(self.cache_decay)
        surface_cache = torch.zeros_like(instant_cards)
        cache_state = torch.zeros(b, self.n_objects, 6, device=s_ext.device, dtype=s_ext.dtype)
        for step_t in range(t):
            probe_irr = torch.matmul(cache_state.reshape(b, self.n_objects * 6), to_probe_flat)
            gi_bounce = torch.matmul(probe_irr, from_probe_flat).reshape(b, self.n_objects, 6) * albedo.squeeze(1)
            cur_cache = decay * cache_state + (1.0 - decay) * (instant_cards[:, step_t] + 0.5 * gi_bounce)
            surface_cache[:, step_t] = cur_cache
            cache_state = cur_cache

        indirect_radiance = (surface_cache * card_weights).sum(dim=-1)
        total_radiance = direct_radiance + 0.5 * indirect_radiance
        mean_rad = total_radiance.mean(dim=-1, keepdim=True).detach().clamp_min(0.12)
        norm_radiance = 0.35 * total_radiance / mean_rad

        # =====================================================================
        # ШАГ 4: РАСЩЕПЛЕНИЕ ПОЛИГОНОВ NANITE УПРАВЛЯЕТСЯ СВОБОДНОЙ ЭНЕРГИЕЙ (СЮРПРИЗОМ)!
        # Если в данном месте контекст предсказан точно (surprise_magnitude мал),
        # полигоны НЕ расщепляются (остается только точка LOD 0).
        # Если возник сюрприз (ошибка на 2D-мембране) и луч упал на объект, открываются уровни LOD 1..3!
        # =====================================================================
        mean_surprise = surprise_magnitude.mean(dim=-1, keepdim=True)  # (B, T, 1)
        norm_surprise = mean_surprise / (mean_surprise.mean(dim=1, keepdim=True).detach() + 1e-6)
        friston_screen_error = norm_radiance * norm_surprise * (obj_radius.view(1, 1, -1) / (t_pos / mean_d2.sqrt() + 0.3)) * 1.4

        lod_thresholds = torch.tensor([0.0, 0.22, 0.48, 0.48, 0.88, 0.88, 0.88, 0.88], device=s_ext.device, dtype=s_ext.dtype).view(1, 1, 1, 8)
        soft_lod = torch.sigmoid((friston_screen_error.unsqueeze(-1) - lod_thresholds) * 8.0)
        soft_lod = soft_lod.clone()
        soft_lod[..., 0] = 1.0

        hard_lod = (friston_screen_error.unsqueeze(-1) >= lod_thresholds).to(s_ext.dtype)
        hard_lod[..., 0] = 1.0
        lod_weights = hard_lod if mode == "vram_streaming" else (hard_lod.detach() - soft_lod.detach() + soft_lod)

        # =====================================================================
        # ШАГ 5: УСВОЕНИЕ ОШИБКИ ПРЕДСКАЗАНИЯ (PERCEPTUAL INFERENCE: mu_prior -> mu_post)
        # Через открытые полигоны Nanite прогоняется именно взвешенная НЕВЯЗКА weighted_error (плюс контекст)!
        # =====================================================================
        # Сочетание сенсорной невязки и внутреннего ожидания на мембране
        membrane_input = 0.65 * weighted_error + 0.35 * mu_groups  # (B, T, 8, G)
        m_perm = membrane_input.permute(2, 0, 1, 3).reshape(self.n_groups, b * t, self.group_dim)
        in_perm = self.obj_in.permute(1, 2, 0)
        group_dots = torch.bmm(m_perm, in_perm).permute(1, 2, 0).reshape(b, t, self.n_objects, self.n_groups)

        envelope = norm_radiance / (1.0 + norm_radiance)
        w_active = envelope.unsqueeze(-1) * lod_weights
        activated = F.silu(group_dots) * w_active

        act_perm = activated.permute(3, 0, 1, 2).reshape(self.n_groups, b * t, self.n_objects)
        out_perm = self.obj_out.permute(1, 0, 2)
        correction_nanite = torch.bmm(act_perm, out_perm).permute(1, 0, 2).reshape(b, t, self.n_groups, self.group_dim)
        correction = torch.einsum("km,btkg->btmg", self.nanite_basis, correction_nanite).reshape(b, t, d)

        # Обновление внутреннего состояния системы по Фристону-Калману:
        # mu_post(t) = mu_prior(t) + K * correction(eps(t))
        k_gain = torch.sigmoid(self.kalman_gain)
        mu_post = mu_prior + k_gain * correction
        s_ext_next = s_ext + correction

        # Вариационная свободная энергия Фристона F(t) = Сюрприз на мембране + KL-сложность обновления состояния
        kl_complexity = 0.5 * (mu_post - mu_prior.detach()).pow(2).mean()
        free_energy = surprise_per_group.mean() + 0.25 * kl_complexity

        with torch.no_grad():
            near_objects_frac = float((hard_lod[..., -1] > 0.5).float().mean().item())
            detail_volume = float((hard_lod[..., 1:] > 0.5).float().mean().item())
            loaded_volume = float(hard_lod.float().mean().item())
            mean_vectors_per_obj = float(hard_lod.sum(dim=-1).mean().item())
            pred_err_norm = float(pred_error_raw.norm(dim=-1).mean().item() / math.sqrt(d))
            fe_val = float(free_energy.item())

        stats = {
            "near_objects_frac": near_objects_frac,
            "detail_volume": detail_volume,
            "loaded_volume": loaded_volume,
            "mean_vectors_per_obj": mean_vectors_per_obj,
            "pred_error_norm": pred_err_norm,
            "free_energy": fe_val,
        }
        return s_ext_next, mu_post, free_energy, stats


class FristonFreeEnergyLM(nn.Module):
    """
    Двухчастная языковая модель по Карлу Фристону:
    - Внешняя среда s_ext входит через сенсорный слой.
    - Внутренний мир системы держит собственное состояние убеждений mu_state и 3D-объекты весов.
    - Внутренний мир предсказывает внешнюю среду (и на уровне каждого 3D-слоя через Одеяло Маркова,
      и на выходе — предсказывая следующий байт внешней среды s(t+1)).
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 96, n_objects: int = 96, n_groups: int = 8, n_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        # Сенсорный вход внешней среды
        self.ext_embed = nn.Embedding(vocab_size, d_model)
        # Начальный априорный якорь внутреннего мира системы (обучаемое базовое ожидание среды)
        self.mu_init_prior = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

        self.layers = nn.ModuleList([FristonMarkovBlanket3DLayer(d_model, n_objects, n_groups) for _ in range(n_layers)])
        self.int_final_norm = nn.LayerNorm(d_model)
        self.ext_final_norm = nn.LayerNorm(d_model)
        # Генеративный декодер внутреннего мира, предсказывающий следующий токен внешней среды s(t+1)
        self.generative_head = nn.Linear(d_model, vocab_size, bias=False)
        self.generative_head.weight = self.ext_embed.weight

    def forward(self, tokens: torch.Tensor, mode: str = "native") -> tuple[torch.Tensor, dict]:
        b, t = tokens.shape
        # Часть 2: Внешняя среда (контекст)
        s_ext = self.ext_embed(tokens)  # (B, T, D)
        # Часть 1: Внутренний мир системы начинает с априорного состояния и выстраивает себя вдоль контекста
        mu_state = self.mu_init_prior.expand(b, t, self.d_model) + 0.5 * s_ext

        keys = (
            "near_objects_frac",
            "detail_volume",
            "loaded_volume",
            "mean_vectors_per_obj",
            "pred_error_norm",
            "free_energy",
        )
        agg = {k: 0.0 for k in keys}
        total_fe = s_ext.new_tensor(0.0)

        for layer in self.layers:
            s_ext, mu_state, fe_layer, st = layer(s_ext, mu_state, mode=mode)
            total_fe = total_fe + fe_layer / len(self.layers)
            for k in keys:
                agg[k] += st[k] / len(self.layers)

        # Выходное предсказание внешней среды строится ансамблем выстроенного внутреннего мира mu_state и границы s_ext
        combined = 0.65 * self.int_final_norm(mu_state) + 0.35 * self.ext_final_norm(s_ext)
        logits = self.generative_head(combined)
        self.last_free_energy = total_fe
        return logits, agg


def eval_friston(model: FristonFreeEnergyLM, data: torch.Tensor, seq_len: int, mode: str) -> tuple[float, float, dict]:
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
    model = FristonFreeEnergyLM(d_model=96, n_objects=96, n_groups=8, n_layers=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров в двухчастной модели v4 (Карл Фристон: Внутренний мир + Одеяло Маркова + Внешняя среда): {n_params:,}", flush=True)

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
        ce_loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
        # Полный функционал обучения Фристона = Предсказание следующего шага внешней среды + Минимизация внутренней Свободной Энергии F
        total_loss = ce_loss + 0.15 * model.last_free_energy

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step == 1 or step % 50 == 0:
            bpc = float(ce_loss.item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | v4_Фристон BPC={bpc:.3f} | F_энергия={st['free_energy']:.3f} | "
                f"невязка_мембраны={st['pred_error_norm']:.3f} | детали={st['detail_volume']*100:.1f}% | "
                f"VRAM={st['loaded_volume']*100:.1f}% ({st['mean_vectors_per_obj']:.2f}/8 вект.) | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append({"step": step, "bpc": round(bpc, 4), **{k: round(v, 4) for k, v in st.items()}})

    nat_bpc, nat_ppl, nat_st = eval_friston(model, valid_data, seq_len, mode="native")
    str_bpc, str_ppl, str_st = eval_friston(model, valid_data, seq_len, mode="vram_streaming")
    print("\n=== ИТОГИ V4 (КАРЛ ФРИСТОН: ВНУТРЕННИЙ МИР vs ВНЕШНЯЯ СРЕДА + LUMEN/NANITE) ===", flush=True)
    print(
        f"v4 Фристон (native):        BPC = {nat_bpc:.3f}, перплексия = {nat_ppl:.2f} | "
        f"невязка мембраны: {nat_st['pred_error_norm']:.3f}, свободная энергия F: {nat_st['free_energy']:.3f} | "
        f"деталей: {nat_st['detail_volume']*100:.1f}%, VRAM: {nat_st['loaded_volume']*100:.1f}% ({nat_st['mean_vectors_per_obj']:.2f}/8 вект.)",
        flush=True,
    )
    print(
        f"v4 Фристон (VRAM-стриминг): BPC = {str_bpc:.3f}, перплексия = {str_ppl:.2f} | "
        f"потеря при стриминге: +{str_bpc - nat_bpc:.4f} BPC",
        flush=True,
    )

    torch.manual_seed(11)
    sample = generate_text(model, "The history of science is ", max_new=90, mode="vram_streaming")
    print(f"Генерация v4 (VRAM-стриминг): {sample!r}", flush=True)

    out_path = OUT_DIR / "rezultat_friston_v4.json"
    out_path.write_text(
        json.dumps(
            {
                "params": n_params,
                "history": history,
                "validation": {
                    "v4_native": {"bpc": round(nat_bpc, 4), "ppl": round(nat_ppl, 2), "stats": nat_st},
                    "v4_vram": {"bpc": round(str_bpc, 4), "ppl": round(str_ppl, 2), "stats": str_st},
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
