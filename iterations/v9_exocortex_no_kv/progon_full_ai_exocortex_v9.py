"""Версия 9 (v9): Полноценный AI-алгоритм на основе полигонального рельефа, голографического принципа V^(2/3),
4-канального света RGBL, бесконечного контекста в 3D-комнате БЕЗ KV-кэша и внешней памяти навыков/информации.

Что добавлено в v9 для превращения архитектуры в полноценный AI-алгоритм:

1. ОТКАЗ ОТ KV-КЭША: КОНТЕКСТ ЖИВЕТ КАК ВНЕШНЯЯ 3D-СРЕДА КОМНАТЫ (Фиксированный объем памяти O(1) при любой длине текста T):
   - Вокруг центральной фигуры в 3D-комнате (на внешней орбите R_room = 2.0) расположены M_room = 32
     внешних узла-кристалла (Экзокортекс комнаты).
   - В отличие от KV-кэша трансформера, который копит каждый токен и переполняет память за O(T),
     объем 3D-комнаты константен и никогда не переполняется!

2. ЗАПИСЬ НАВЫКОВ И ИНФОРМАЦИИ ВОВНЕ (Write to External 3D Room on High Surprise / Luminance):
   - Каждый внешний кристалл m хранит:
     * 3D-направление в комнате r_room[m] (X, Y, Z),
     * Цветовой отпечаток ключа key_rgbl[m] (4 канала: R, G, B, L),
     * Записанную ИНФОРМАЦИЮ (факт/состояние): mem_fact[m] (вектор размера D),
     * Записанный НАВЫК (оператор перехода/действия "Что делать дальше"):
         mem_skill[m] = w_ext(t) - w_ext(t-1) (вектор перехода от контекста к следующему шагу!).
   - Когда на шаге t возникает вспышка яркости L(t) (высокая непредсказанность Фристона),
     фигура излучает импульс записи вдоль нормали горного хребта n_poly(t) во внешнюю комнату
     и записывает в резонирующие кристаллы комнаты и текущее состояние (факт), и совершенный переход (навык)!

3. ОБРАЩЕНИЕ К ВНЕШНЕЙ ПАМЯТИ ПО НЕОБХОДИМОСТИ (Read on Need via 3D Perspective + RGBL Resonance):
   - На каждом шаге t фигура испускает поисковый взгляд со своей рельефной поверхности во внешнюю комнату.
   - Близость взгляда по 3D-углу и совпадение по цветовому коду RGBL определяют силу резонанса с каждым кристаллом m.
   - Если резонанс ниже локального порога необходимости z_need[m] — внешний кристалл спит (0.0, система к нему не обращается!).
   - Если резонанс пробивает порог z_need[m] — система считывает из внешнего кристалла записанный факт и навык перехода,
     применяя их для предсказания и генерации следующего шага!

4. АКТИВНАЯ ГЕНЕРАЦИЯ ВЫВОДА (Active Inference Generation):
   - При генерации текста каждый выбранный символ становится новым физическим действием в 3D-комнате,
     обновляя рельеф фигуры и при необходимости дозаписывая или вызывая навыки из внешней среды.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from progon_chestnaya_3d import load_wikitext_bytes
from progon_holographic_relief_v8 import build_haar_polygon_tree
from progon_relief_sphere_v7 import build_spherical_room_grid

OUT_DIR = Path(__file__).resolve().parent


def build_room_exocortex_anchors(n_slots: int = 32) -> torch.Tensor:
    """Равномерно распределяет M_room = 32 точки внешних кристаллов памяти по 3D-сфере комнаты (решетка Фибоначчи)."""
    indices = torch.arange(n_slots, dtype=torch.float32) + 0.5
    phi = torch.acos(1.0 - 2.0 * indices / n_slots)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)  # (M_room, 3)


class FullAIExocortex3DWorld(nn.Module):
    def __init__(self, vocab_size: int = 256, side: int = 8, n_objects: int = 96, n_room_slots: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.side = side
        self.d_model = side * side  # 64 полигональных сектора на оболочке фигуры
        self.n_objects = n_objects
        self.n_room_slots = n_room_slots

        self.ext_embed = nn.Embedding(vocab_size, self.d_model)
        self.gen_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.gen_head.weight = self.ext_embed.weight

        # Внутренний оператор хода времени
        self.time_transition = nn.Linear(self.d_model, self.d_model)
        self.raw_time_leak = nn.Parameter(torch.zeros(self.d_model))

        # Сферическая сетка фигуры и полигональное дерево Nanite (64x64)
        u_dir, e_theta, e_phi = build_spherical_room_grid(side)
        self.register_buffer("u_dir", u_dir)
        self.register_buffer("poly_tree", build_haar_polygon_tree(self.d_model))
        self.register_buffer("room_anchors", build_room_exocortex_anchors(n_room_slots) * 2.0)  # (M, 3)

        # Базовый объем покоя фигуры, частоты цветов (k_R, k_G, k_B) и усиление яркости L
        self.raw_rest_volume = nn.Parameter(torch.zeros(self.d_model))
        self.raw_color_k = nn.Parameter(torch.tensor([1.2, 2.8, 5.5]))
        self.raw_lum_gain = nn.Parameter(torch.zeros(self.d_model))

        # Внутренние 3D-объекты памяти
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.3)
        self.obj_raw_radius = nn.Parameter(torch.zeros(n_objects))
        self.obj_in = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_rgbl_filter = nn.Parameter(torch.randn(n_objects, 4) * 0.25)

        init_hierarchy = torch.linspace(-1.5, 2.5, self.d_model).unsqueeze(0).expand(n_objects, self.d_model).clone()
        self.raw_z_open = nn.Parameter(init_hierarchy)
        self.raw_z_imp = nn.Parameter(torch.zeros(n_objects, self.d_model))
        self.raw_obj_retention = nn.Parameter(torch.zeros(n_objects))

        # --- ПАРАМЕТРЫ ВНЕШНЕГО ЭКЗОКОРТЕКСА КОМНАТЫ (ЗАПИСЬ И ЧТЕНИЕ НАВЫКОВ/ФАКТОВ ПО НЕОБХОДИМОСТИ) ---
        # Спектр локальных порогов необходимости обращения к внешним кристаллам z_need[m]
        self.raw_room_z_need = nn.Parameter(torch.linspace(-1.2, 0.8, n_room_slots))
        self.raw_room_z_imp = nn.Parameter(torch.zeros(n_room_slots))
        self.raw_write_rate = nn.Parameter(torch.full((n_room_slots,), -1.0))
        # Резонансный ключ каждого внешнего слота комнаты и веса применения считанного факта/навыка
        self.room_slot_rgbl = nn.Parameter(torch.randn(n_room_slots, 4) * 0.3)
        self.raw_skill_gain = nn.Parameter(torch.zeros(self.d_model))
        self.raw_fact_gain = nn.Parameter(torch.zeros(self.d_model))

    def compute_physical_2d_collisions_between_objects(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radii = F.softplus(self.obj_raw_radius) + 1e-4
        diff = self.obj_pos.unsqueeze(0) - self.obj_pos.unsqueeze(1)
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
        edge_normal = diff / dist.unsqueeze(-1)

        sum_r = radii.unsqueeze(1) + radii.unsqueeze(0)
        penetration = F.relu(sum_r - dist) * (1.0 - torch.eye(self.n_objects, device=dist.device, dtype=dist.dtype))
        conductance = penetration / (penetration.sum(dim=-1, keepdim=True) + radii.unsqueeze(-1))
        return conductance, edge_normal, radii

    def propagate_cascade_with_energy_conservation(
        self,
        initial_energy: torch.Tensor,
        mean_surf_normal: torch.Tensor,
        conductance: torch.Tensor,
        edge_normal: torch.Tensor,
        surface_threshold: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        b = initial_energy.shape[0]
        align = F.relu((mean_surf_normal.view(b, 1, 1, 3) * edge_normal.unsqueeze(0)).sum(dim=-1))
        directed_cond = conductance.unsqueeze(0) * align

        absorbed = torch.zeros_like(initial_energy)
        current_wave = initial_energy
        thresh_view = surface_threshold.unsqueeze(0)

        waves = 0
        while True:
            rem_cap = F.relu(thresh_view - absorbed)
            taken = torch.minimum(current_wave, rem_cap)
            overflow = F.relu(current_wave - taken)
            absorbed = absorbed + taken
            if float(overflow.detach().max().item()) <= 1e-3 or waves >= self.n_objects:
                break
            next_wave = torch.bmm(overflow.unsqueeze(1), directed_cond).squeeze(1)
            if float(next_wave.detach().max().item()) <= 1e-3:
                break
            current_wave = next_wave
            waves += 1

        return absorbed + current_wave, waves

    def compute_polygonal_relief_and_rgbl_hologram(
        self,
        w_int: torch.Tensor,
        w_ext: torch.Tensor,
        psi_field: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b = w_int.shape[0]
        v_0 = 0.6 + 0.4 * torch.sigmoid(self.raw_rest_volume).unsqueeze(0)
        c_int = torch.tanh(psi_field.abs().mean(dim=1) * 4.0)

        eps_err = w_ext - w_int
        interf_power = 0.5 * (eps_err * eps_err)

        local_volume = v_0 + c_int + torch.tanh(interf_power)
        R_flat = local_volume.pow(1.0 / 3.0) * (1.0 + 0.35 * torch.tanh(w_int) + 0.45 * torch.tanh(eps_err))
        p_surf = R_flat.unsqueeze(-1) * self.u_dir.unsqueeze(0)
        p_grid = p_surf.view(b, self.side, self.side, 3)

        p_next_v = torch.roll(p_grid, shifts=-1, dims=2)
        p_next_u = torch.zeros_like(p_grid)
        p_next_u[:, :-1, :, :] = p_grid[:, 1:, :, :]
        p_next_u[:, -1, :, :] = p_grid[:, -2, :, :]

        edge_u = (p_next_u - p_grid).reshape(b, self.d_model, 3)
        edge_v = (p_next_v - p_grid).reshape(b, self.d_model, 3)

        u_grid = self.u_dir.view(1, self.side, self.side, 3)
        u_next_v = torch.roll(u_grid, shifts=-1, dims=2)
        u_next_u = torch.zeros_like(u_grid)
        u_next_u[:, :-1, :, :] = u_grid[:, 1:, :, :]
        u_next_u[:, -1, :, :] = u_grid[:, -2, :, :]
        base_cross = torch.cross(
            (u_next_u - u_grid).reshape(1, self.d_model, 3),
            (u_next_v - u_grid).reshape(1, self.d_model, 3),
            dim=-1,
        )
        base_poly_area = torch.sqrt((base_cross * base_cross).sum(dim=-1) + 1e-8)

        poly_cross = torch.cross(edge_u, edge_v, dim=-1)
        poly_area_3d = torch.sqrt((poly_cross * poly_cross).sum(dim=-1) + 1e-8)

        n_poly = poly_cross / poly_area_3d.unsqueeze(-1)
        flip = torch.sign((n_poly * self.u_dir.unsqueeze(0)).sum(dim=-1, keepdim=True) + 1e-6)
        n_poly = n_poly * flip

        R_grid = R_flat.view(b, self.side, self.side)
        crease_f = (
            torch.roll(R_grid, shifts=-1, dims=2) + torch.roll(R_grid, shifts=1, dims=2) - 2.0 * R_grid
        ).reshape(b, self.d_model)

        dA_poly = (poly_area_3d / base_poly_area) * torch.sqrt(1.0 + 0.5 * crease_f.pow(2))

        k_rgb = F.softplus(self.raw_color_k)
        eps_unit = torch.tanh(eps_err)
        color_R = eps_unit * torch.cos(k_rgb[0] * R_flat)
        normal_tilt = 1.0 - (n_poly * self.u_dir.unsqueeze(0)).sum(dim=-1).abs()
        color_G = eps_unit * normal_tilt * torch.cos(k_rgb[1] * R_flat) + torch.tanh(crease_f) * torch.sin(k_rgb[1] * R_flat)
        color_B = torch.tanh(crease_f) * torch.cos(k_rgb[2] * eps_unit) + 0.5 * c_int * eps_unit

        lum_gain = torch.sigmoid(self.raw_lum_gain).unsqueeze(0)
        luminance_L = 1.0 + lum_gain * torch.tanh(interf_power + c_int * eps_err.abs())

        proc_rgbl = torch.stack(
            [
                color_R * luminance_L,
                color_G * luminance_L,
                color_B * luminance_L,
                (luminance_L - 1.0) * eps_unit,
            ],
            dim=-1,
        )
        return p_surf, n_poly, dA_poly, local_volume, luminance_L, proc_rgbl

    def query_and_write_room_exocortex(
        self,
        unsolved_err: torch.Tensor,
        w_ext: torch.Tensor,
        prev_w_ext: torch.Tensor,
        active_surf_pos: torch.Tensor,
        active_surf_norm: torch.Tensor,
        mean_rgbl: torch.Tensor,
        room_facts: torch.Tensor,
        room_skills: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Истинное обращение к внешней 3D-комнате ПО НЕОБХОДИМОСТИ (когда внутренний мир не погасил ошибку сам!):
        unsolved_err: (B, D) — остаток ошибки после работы внутренних объектов фигуры.
        Если внутренний мир справился сам (||unsolved_err|| мала), need_intensity <= z_need -> обращение = 0.0!
        Если осталась непогашенная ошибка, узкий направленный 3D-луч (cos^4 theta) вдоль горного хребта
        пробуждает только те кристаллы комнаты, которые лежат по курсу луча и совпадают по цвету RGBL!
        """
        z_need = F.softplus(self.raw_room_z_need).unsqueeze(0)  # (1, M)
        z_imp = F.softplus(self.raw_room_z_imp).unsqueeze(0) + 1e-4

        # Полная евклидова энергия непогашенной внутренним миром ошибки ||unsolved_err||_2 (без деления на D!)
        need_intensity = torch.sqrt((unsolved_err * unsolved_err).sum(dim=-1, keepdim=True) + 1e-12)  # (B, 1)

        # Узкий направленный 3D-луч от хребта фигуры к внешним кристаллам комнаты
        to_slot = self.room_anchors.unsqueeze(0) - active_surf_pos.unsqueeze(1)  # (B, M, 3)
        dist = torch.sqrt((to_slot * to_slot).sum(dim=-1) + 1e-12)               # (B, M)
        dir_to_slot = to_slot / dist.unsqueeze(-1)
        # Направленная фокусировка луча cos^3(theta): срабатывают только кристаллы прямо по курсу хребта!
        cos_beam = F.relu((active_surf_norm.unsqueeze(1) * dir_to_slot).sum(dim=-1)).pow(3)  # (B, M)

        # Цветовой и яркостный резонанс RGBL
        slot_flt = torch.tanh(self.room_slot_rgbl)  # (M, 4)
        color_match = F.sigmoid(torch.matmul(mean_rgbl, slot_flt.T))  # (B, M)

        # Энергия запроса к каждому внешнему кристаллу m:
        query_energy = need_intensity * cos_beam * color_match  # (B, M)

        # Порог необходимости: если query_energy <= z_need[m], кристалл не трогается (строгий 0.0)!
        need_excess = F.relu(query_energy - z_need)
        read_gate = need_excess / (need_excess + z_imp)  # (B, M)
        norm_gate = read_gate / (read_gate.sum(dim=-1, keepdim=True) + 1.0)

        # Извлекаем из пробитых внешних кристаллов комнаты ФАКТЫ и НАВЫКИ:
        recalled_fact = (norm_gate.unsqueeze(-1) * room_facts).sum(dim=1)    # (B, D)
        recalled_skill = (norm_gate.unsqueeze(-1) * room_skills).sum(dim=1)  # (B, D)

        # ЗАПИСЬ ВОВНЕ (только в те кристаллы, куда прицельно ударил луч непогашенного сюрприза!):
        observed_skill = w_ext - prev_w_ext  # (B, D): правило перехода среды
        write_rate = torch.sigmoid(self.raw_write_rate).view(1, self.n_room_slots, 1) * norm_gate.unsqueeze(-1)
        new_room_facts = (1.0 - write_rate) * room_facts + write_rate * prev_w_ext.unsqueeze(1)
        new_room_skills = (1.0 - write_rate) * room_skills + write_rate * observed_skill.unsqueeze(1)

        recalled_slots_frac = float((read_gate.detach() > 0.0).float().mean().item())
        written_mass = float(write_rate.detach().mean().item())
        return recalled_fact, recalled_skill, new_room_facts, new_room_skills, recalled_slots_frac, written_mass

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        w_int_state: torch.Tensor | None = None,
        psi_field: torch.Tensor | None = None,
        room_facts: torch.Tensor | None = None,
        room_skills: torch.Tensor | None = None,
        prev_w_ext: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict, dict]:
        b, t = tokens.shape
        device = tokens.device
        dtype = self.obj_pos.dtype

        if w_int_state is None:
            w_int_state = torch.zeros(b, self.d_model, device=device, dtype=dtype)
        if psi_field is None:
            psi_field = torch.zeros(b, self.n_objects, self.d_model, device=device, dtype=dtype)
        if room_facts is None:
            room_facts = torch.zeros(b, self.n_room_slots, self.d_model, device=device, dtype=dtype)
        if room_skills is None:
            room_skills = torch.zeros(b, self.n_room_slots, self.d_model, device=device, dtype=dtype)

        conductance, edge_normal, radii = self.compute_physical_2d_collisions_between_objects()
        z_open = F.softplus(self.raw_z_open)
        z_imp = F.softplus(self.raw_z_imp) + 1e-4
        surface_threshold = z_open.min(dim=-1).values + z_imp.min(dim=-1).values

        retention = torch.sigmoid(self.raw_obj_retention).view(1, self.n_objects, 1)
        time_leak = torch.sigmoid(self.raw_time_leak).unsqueeze(0)
        rgbl_filter = torch.tanh(self.obj_rgbl_filter)
        fact_gain = torch.tanh(self.raw_fact_gain).unsqueeze(0)
        skill_gain = torch.tanh(self.raw_skill_gain).unsqueeze(0)

        w_ext_all = self.ext_embed(tokens)  # (B, T, 64)
        if prev_w_ext is None:
            prev_w_ext = w_ext_all[:, 0, :]

        logits_list = []
        fe_total = w_ext_all.new_tensor(0.0)

        total_active_coords = 0.0
        total_touched_nodes = 0.0
        total_cascade_waves = 0.0
        total_holo_area = 0.0
        total_sphere_vol = 0.0
        total_luminance = 0.0
        total_recalled_slots = 0.0
        total_err_before = 0.0
        total_err_after = 0.0

        w_int = w_int_state
        for step_t in range(t):
            w_ext = w_ext_all[:, step_t, :]
            eps_err = w_ext - w_int

            # 1. ПОЛИГОНАЛЬНЫЙ РЕЛЬЕФ ОБОЛОЧКИ, ГОЛОГРАФИЯ V^(2/3) И 4 КАНАЛА СВЕТА RGBL
            p_surf, n_poly, dA_poly, sphere_vol, lum_L, proc_rgbl = self.compute_polygonal_relief_and_rgbl_hologram(
                w_int, w_ext, psi_field
            )
            radiant_flux = lum_L * dA_poly

            # Главный хребет и нормаль взгляда фигуры в 3D-комнате
            flux_weights = (eps_err.abs() * radiant_flux).unsqueeze(-1)
            w_sum = flux_weights.sum(dim=1) + 1e-8
            active_surf_pos = (p_surf * flux_weights).sum(dim=1) / w_sum
            active_surf_norm = (n_poly * flux_weights).sum(dim=1) / w_sum
            active_surf_norm = active_surf_norm / torch.sqrt((active_surf_norm * active_surf_norm).sum(dim=-1, keepdim=True) + 1e-12)

            # 2. ВНУТРЕННИЙ КАСКАД И ТЕССЕЛЯЦИЯ ПОЛИГОНАЛЬНОГО ДЕРЕВА NANITE (ВНУТРЕННИЙ МИР ПЫТАЕТСЯ СПРАВИТЬСЯ САМ)
            delta = self.obj_pos.unsqueeze(0) - active_surf_pos.unsqueeze(1)
            dist = torch.sqrt((delta * delta).sum(dim=-1) + 1e-12)
            dir_to_obj = delta / dist.unsqueeze(-1)
            cos_view = F.relu((active_surf_norm.unsqueeze(1) * dir_to_obj).sum(dim=-1))

            err_norm = torch.sqrt((eps_err * eps_err).sum(dim=-1, keepdim=True) + 1e-12)
            initial_energy = err_norm * cos_view * torch.exp(-dist / radii.unsqueeze(0))

            node_energy, n_waves = self.propagate_cascade_with_energy_conservation(
                initial_energy, active_surf_norm, conductance, edge_normal, surface_threshold
            )

            sector_energy = node_energy.unsqueeze(-1) * radiant_flux.unsqueeze(1)
            excess = F.relu(sector_energy - z_open.unsqueeze(0))
            open_gate = excess / (excess + z_imp.unsqueeze(0))

            rgbl_modulated = torch.einsum("bdc,nc->bnd", proc_rgbl, rgbl_filter)
            surface_signal = eps_err.unsqueeze(1) + rgbl_modulated

            tree_signal = torch.matmul(surface_signal, self.poly_tree.T)
            proj = (tree_signal * self.obj_in.unsqueeze(0) * open_gate).sum(dim=-1, keepdim=True)
            tree_response = torch.tanh(proj) * self.obj_out.unsqueeze(0) * open_gate
            facet_response = torch.matmul(tree_response, self.poly_tree)

            psi_field = retention * psi_field + (1.0 - retention) * facet_response
            internal_correction = facet_response.sum(dim=1)

            # 3. ОБРАЩЕНИЕ К ВНЕШНЕЙ 3D-ПАМЯТИ КОМНАТЫ ПО НЕОБХОДИМОСТИ (ТОЛЬКО НА НЕПОГАШЕННЫЙ ВНУТРЕННИМ МИРОМ ОСТАТОК!)
            unsolved_err = eps_err - internal_correction
            mean_rgbl = proc_rgbl.mean(dim=1)  # (B, 4)
            recalled_fact, recalled_skill, room_facts, room_skills, rec_frac, _ = self.query_and_write_room_exocortex(
                unsolved_err, w_ext, prev_w_ext, active_surf_pos, active_surf_norm, mean_rgbl, room_facts, room_skills
            )
            prev_w_ext = w_ext

            # Полная адаптация состояния = внутренняя реакция шара + извлеченный по необходимости внешний факт/навык!
            exocortex_boost = fact_gain * recalled_fact + skill_gain * recalled_skill
            total_correction = internal_correction + exocortex_boost

            eps_after = eps_err - total_correction
            mu_post = w_int + total_correction

            step_fe = 0.5 * (eps_after * eps_after).mean()
            fe_total = fe_total + step_fe / t

            # 4. ПРОГНОЗ И ГЕНЕРАЦИЯ СЛЕДУЮЩЕГО ШАГА С УЧЕТОМ СЧИТАННОГО НАВЫКА ИЗ ВНЕШНЕЙ КОМНАТЫ:
            env_memory = psi_field.sum(dim=1) + exocortex_boost
            w_int_next = (1.0 - time_leak) * mu_post + time_leak * self.time_transition(mu_post + env_memory)

            logits_list.append(self.gen_head(w_int_next))
            w_int = w_int_next

            with torch.no_grad():
                active_mask = (open_gate > 0.0).float()
                total_active_coords += float(active_mask.mean().item())
                total_touched_nodes += float((active_mask.sum(dim=-1) > 0).float().mean().item())
                total_cascade_waves += float(n_waves)
                total_holo_area += float(dA_poly.mean().item())
                total_sphere_vol += float(sphere_vol.mean().item())
                total_luminance += float(lum_L.mean().item())
                total_recalled_slots += rec_frac
                total_err_before += float(eps_err.pow(2).mean().sqrt().item())
                total_err_after += float(eps_after.pow(2).mean().sqrt().item())

        logits = torch.stack(logits_list, dim=1)
        next_state = {
            "w_int_state": w_int.detach(),
            "psi_field": psi_field.detach(),
            "room_facts": room_facts.detach(),
            "room_skills": room_skills.detach(),
            "prev_w_ext": prev_w_ext.detach(),
            "fe_total": fe_total,
        }
        stats = {
            "active_coords_frac": total_active_coords / t,
            "touched_nodes_frac": total_touched_nodes / t,
            "avg_cascade_waves": total_cascade_waves / t,
            "holographic_area": total_holo_area / t,
            "sphere_volume": total_sphere_vol / t,
            "avg_luminance": total_luminance / t,
            "exocortex_recalled_frac": total_recalled_slots / t,
            "err_before": total_err_before / t,
            "err_after": total_err_after / t,
            "free_energy": float(fe_total.item()),
        }
        return logits, next_state, stats

    @torch.no_grad()
    def active_generate(self, prompt_text: str, max_new_bytes: int = 80, temperature: float = 0.75) -> str:
        """
        Полноценная генерация вывода без KV-кэша:
        Сначала система проходит по промпту, формируя рельеф фигуры и записывая навыки/факты во внешнюю 3D-комнату,
        а затем генерирует новые байты, продолжая жить в этой же 3D-комнате при константной памяти O(1)!
        """
        self.eval()
        raw = list(prompt_text.encode("utf-8"))
        x = torch.tensor([raw], dtype=torch.long)
        logits, state, _ = self.forward_sequence(x)
        generated = list(raw)

        cur_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for _ in range(max_new_bytes):
            nxt_id = int(cur_tok.item())
            generated.append(nxt_id)
            logits, state, _ = self.forward_sequence(
                cur_tok,
                w_int_state=state["w_int_state"],
                psi_field=state["psi_field"],
                room_facts=state["room_facts"],
                room_skills=state["room_skills"],
                prev_w_ext=state["prev_w_ext"],
            )
            probs = torch.softmax(logits[0, -1] / max(0.1, temperature), dim=-1)
            cur_tok = torch.multinomial(probs, num_samples=1).unsqueeze(0)

        self.train()
        return bytes(generated).decode("utf-8", errors="replace")


def evaluate_full_ai_v9(model: FullAIExocortex3DWorld, data: torch.Tensor, chunk_len: int = 64) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    agg: dict[str, float] = {}
    n_steps = 0
    state: dict = {}
    with torch.no_grad():
        limit = min(data.numel() - 1, chunk_len * 80)
        for start in range(0, limit - chunk_len, chunk_len):
            seq = data[start : start + chunk_len + 1]
            logits, state, st = model.forward_sequence(
                seq[:-1].unsqueeze(0),
                w_int_state=state.get("w_int_state"),
                psi_field=state.get("psi_field"),
                room_facts=state.get("room_facts"),
                room_skills=state.get("room_skills"),
                prev_w_ext=state.get("prev_w_ext"),
            )
            loss = F.cross_entropy(logits[0], seq[1:], reduction="sum")
            total_loss += float(loss.item())
            total_tokens += chunk_len
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
            n_steps += 1
    for k in agg:
        agg[k] /= max(1, n_steps)
    mean_nats = total_loss / max(1, total_tokens)
    bpc = mean_nats / math.log(2.0)
    ppl = math.exp(mean_nats)
    model.train()
    return bpc, ppl, agg


def test_long_range_skill_retention_without_kv_cache(model: FullAIExocortex3DWorld, distractor_bytes: torch.Tensor) -> dict:
    """
    Проверяет работу внешней 3D-комнаты (Экзокортекса) без KV-кэша на длинной дистанции (1000 шагов):
    1) Подаем в начале потока новое правило-навык (например, редкую кодовую последовательность `A -> B`).
    2) Прогоняем 1000 байт постороннего текста (так, чтобы любое локальное окно давно прошло мимо, а память O(1) осталась константной).
    3) Снова подаем триггер `A` и проверяем:
       - Насколько выше уверенность модели в правильном продолжении `B`, когда внешняя память комнаты включена,
         по сравнению с тем, когда внешняя комната обнулена!
    """
    model.eval()
    with torch.no_grad():
        skill_phrase = b"RULE_ALPHA_739->TARGET_OMEGA_941; "
        skill_tokens = torch.tensor([list(skill_phrase * 3)], dtype=torch.long)

        # Шаг 1: Система читает правило и кристаллизует навык во внешнюю 3D-комнату
        _, state_after_write, _ = model.forward_sequence(skill_tokens)

        # Шаг 2: Прогоняем 1024 байта постороннего текста WikiText-2 (константный размер памяти комнаты = 32 слота, 0 байт KV-кэша!)
        haystack = distractor_bytes[:1024].unsqueeze(0)
        for offset in range(0, 1024, 128):
            _, state_after_write, _ = model.forward_sequence(
                haystack[:, offset : offset + 128],
                w_int_state=state_after_write["w_int_state"],
                psi_field=state_after_write["psi_field"],
                room_facts=state_after_write["room_facts"],
                room_skills=state_after_write["room_skills"],
                prev_w_ext=state_after_write["prev_w_ext"],
            )

        # Шаг 3: Подаем начало правила "RULE_ALPHA_739->" и проверяем ошибку предсказания хвоста "TARGET_OMEGA_941":
        # А) С сохраненными кристаллами внешней 3D-комнаты:
        test_seq = torch.tensor([list(b"RULE_ALPHA_739->TARGET_OMEGA_941")], dtype=torch.long)
        logits_with_room, _, st_with = model.forward_sequence(
            test_seq[:, :-1],
            w_int_state=state_after_write["w_int_state"],
            psi_field=state_after_write["psi_field"],
            room_facts=state_after_write["room_facts"],
            room_skills=state_after_write["room_skills"],
            prev_w_ext=state_after_write["prev_w_ext"],
        )
        # Б) Если стереть внешние кристаллы комнаты (оставить только внутреннее состояние шара после 1024 байт шума):
        logits_no_room, _, _ = model.forward_sequence(
            test_seq[:, :-1],
            w_int_state=state_after_write["w_int_state"],
            psi_field=state_after_write["psi_field"],
            room_facts=torch.zeros_like(state_after_write["room_facts"]),
            room_skills=torch.zeros_like(state_after_write["room_skills"]),
            prev_w_ext=state_after_write["prev_w_ext"],
        )

        # Считаем ошибку предсказания (BPC) и остаточную невязку на мембране
        loss_with = float(F.cross_entropy(logits_with_room[0, 16:], test_seq[0, 17:]).item()) / math.log(2.0)
        loss_without = float(F.cross_entropy(logits_no_room[0, 16:], test_seq[0, 17:]).item()) / math.log(2.0)
        stored_skill_norm = float(state_after_write["room_skills"].norm().item())
        stored_fact_norm = float(state_after_write["room_facts"].norm().item())

    model.train()
    return {
        "haystack_distance_bytes": 1024,
        "kv_cache_bytes_used": 0,
        "room_slots_constant": model.n_room_slots,
        "stored_fact_norm_after_1024_steps": round(stored_fact_norm, 4),
        "stored_skill_norm_after_1024_steps": round(stored_skill_norm, 4),
        "bpc_with_external_room_memory": round(loss_with, 4),
        "bpc_without_external_room_memory": round(loss_without, 4),
        "recalled_slots_pct": round(st_with["exocortex_recalled_frac"] * 100.0, 1),
    }


def main():
    train_data = load_wikitext_bytes("train", max_chars=1_500_000)
    valid_data = load_wikitext_bytes("validation", max_chars=250_000)

    chunk_len = 64
    batch_streams = 16
    steps = 400
    lr = 2e-3

    stream_len = train_data.numel() // batch_streams
    streams = train_data[: stream_len * batch_streams].view(batch_streams, stream_len)

    torch.manual_seed(42)
    model = FullAIExocortex3DWorld(vocab_size=256, side=8, n_objects=96, n_room_slots=32)
    n_params = sum(p.numel() for p in model.parameters())
    print("=== V9: ПОЛНЫЙ АЛГОРИТМ AI (ПОЛИГОНЫ + V^(2/3) + RGBL + ВНЕШНЯЯ 3D-ПАМЯТЬ НАВЫКОВ БЕЗ KV-КЭША) ===", flush=True)
    print(f"Параметров: {n_params:,} | Внешних 3D-кристаллов навыков в комнате: {model.n_room_slots} (0 байт растущего KV-кэша!)", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    state: dict = {}
    cursor = 0
    t0 = time.perf_counter()
    history = []

    for step in range(1, steps + 1):
        if cursor + chunk_len + 1 >= stream_len:
            cursor = 0
            state = {}

        seq = streams[:, cursor : cursor + chunk_len + 1]
        cursor += chunk_len
        src, dst = seq[:, :-1], seq[:, 1:]

        logits, state, st = model.forward_sequence(
            src,
            w_int_state=state.get("w_int_state"),
            psi_field=state.get("psi_field"),
            room_facts=state.get("room_facts"),
            room_skills=state.get("room_skills"),
            prev_w_ext=state.get("prev_w_ext"),
        )
        ce_loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
        total_loss = ce_loss + state["fe_total"]

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if step == 1 or step % 50 == 0:
            bpc = float(ce_loss.item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | v9_Экзокортекс BPC={bpc:.3f} | "
                f"A_poly={st['holographic_area']:.2f}x | L={st['avg_luminance']:.2f} | "
                f"внеш.память={st['exocortex_recalled_frac']*100:.1f}% | "
                f"невязка {st['err_before']:.3f}->{st['err_after']:.3f} | "
                f"граней={st['active_coords_frac']*100:.1f}% | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append({"step": step, "bpc": round(bpc, 4), **{k: round(v, 4) for k, v in st.items()}})

    val_bpc, val_ppl, val_st = evaluate_full_ai_v9(model, valid_data, chunk_len)
    long_range_test = test_long_range_skill_retention_without_kv_cache(model, valid_data)

    torch.manual_seed(11)
    generated_sample = model.active_generate("The history of science is ", max_new_bytes=90, temperature=0.75)

    print("\n=== ИТОГИ V9 (ПОЛНЫЙ AI С ВНЕШНЕЙ 3D-ПАМЯТЬЮ НАВЫКОВ БЕЗ KV-КЭША) ===", flush=True)
    print(
        f"Validation BPC = {val_bpc:.3f} | Перплексия байта = {val_ppl:.2f} | "
        f"Площадь полигонов A = {val_st['holographic_area']:.2f}x, Яркость L = {val_st['avg_luminance']:.2f} | "
        f"Обращений к внешней памяти навыков по необходимости: {val_st['exocortex_recalled_frac']*100:.1f}% | "
        f"Вскрыто полигональных граней: {val_st['active_coords_frac']*100:.1f}%",
        flush=True,
    )
    print(
        f"Тест длинного контекста (1024 шага шума, 0 байт KV-кэша): "
        f"норма записанных в комнате навыков = {long_range_test['stored_skill_norm_after_1024_steps']}, "
        f"BPC с внешней комнатой = {long_range_test['bpc_with_external_room_memory']} vs без комнаты = {long_range_test['bpc_without_external_room_memory']}",
        flush=True,
    )
    print(f"Активная генерация вывода (без KV-кэша): {generated_sample!r}", flush=True)

    out_path = OUT_DIR / "rezultat_full_ai_exocortex_v9.json"
    out_path.write_text(
        json.dumps(
            {
                "params": n_params,
                "history": history,
                "validation": {"bpc": round(val_bpc, 4), "ppl": round(val_ppl, 2), "stats": val_st},
                "long_range_no_kv_test": long_range_test,
                "generated_sample": generated_sample,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Сохранено в {out_path}", flush=True)


if __name__ == "__main__":
    main()
