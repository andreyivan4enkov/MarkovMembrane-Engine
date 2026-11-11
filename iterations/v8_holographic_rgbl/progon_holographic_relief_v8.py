"""Версия 8 (v8): Единый синтез:
1. Полигональная 2D-оболочка центральной фигуры в 3D-комнате (X, Y, Z):
   - Оболочка состоит из замкнутой 3D-сетки полигонов (треугольников/квадов) с вершинами p(u, v, t) = R(u, v, t) * u_dir(u, v).
   - Точная 3D-площадь dA(u, v, t) и точная 3D-нормаль n(u, v, t) каждого полигона рельефа вычисляются
     напрямую через векторное произведение 3D-ребер полигона:
       edge_u = p(u+1, v) - p(u, v),   edge_v = p(u, v+1) - p(u, v),
       cross_uv = edge_u x edge_v,     dA_poly = ||cross_uv||,     n_poly = cross_uv / dA_poly!
     Когда на поверхности вырастает гора или впадина, ребра полигона удлиняются в 3D, и площадь полигона dA_poly
     физически увеличивается ("3 км по горе вместо 1 км по равнине"), а нормаль грани поворачивается!

2. Голографический принцип (A ~ V^(2/3)) от активной внутренней сложности и двусторонней интерференции:
   - Внутренний объем фигуры V(u, v, t) определяется плотностью активно задействованных внутренних признаков
     и процессов C_int(u, v, t) + энергией встречной интерференции между внутренним миром (изнутри шара)
     и внешней средой (снаружи шара).
   - Радиус вершин полигона масштабируется как R ~ V^(1/3), поэтому площадь полигонального рельефа
     автоматически масштабируется как dA_poly ~ R^2 ~ V^(2/3) * крутизна_склона!

3. Иерархическое огрубление полигонов (Nanite DAG: 1 -> 2 -> 4 -> 8 граней):
   - Векторы каждого 3D-узла памяти уложены в ортогональное полигональное дерево Хаара-Nanite:
     при малом объеме/дали полигоны сшиваются в крупный родительский силуэт без потери суммы граней,
     а при росте площади полигона dA_poly и яркости L — тесселируются (расщепляются) в детальные микрополигоны!

4. Цвета (R, G, B) + Яркость (L / Luminance), натянутые прямо на полигоны рельефа:
   - На каждом полигоне рельефа из двусторонней интерференции (внутреннее <-> внешнее) и геометрии его ребер
     детерминированно генерируются 4 параллельных световых канала (R, G, B, L):
     * R (Красный): длинноволновая фазовая интерференция на вершинах полигона;
     * G (Зеленый): средневолновой перелив вдоль ребер и наклона граней полигона (edge_u, edge_v);
     * B (Синий): коротковолновая текстура излома между соседними полигонами (кривизна/шероховатость);
     * L (Яркость / Luminance): энергетическая светимость канала (сила излучения полигона), которая
       умножается на 3D-площадь полигона dA_poly, определяя дальность пробития луча в 3D-комнате
       и глубину тесселяции полигонов!
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
from progon_relief_sphere_v7 import build_spherical_room_grid

OUT_DIR = Path(__file__).resolve().parent


def build_haar_polygon_tree(d_model: int = 64) -> torch.Tensor:
    """
    Строит полное ортонормированное бинарное дерево огрубления полигонов (Haar Wavelet / Nanite DAG)
    размерности d_model x d_model (для d_model = 64: 6 уровней иерархии 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 полигона):
    - Ось 0: глобальный силуэт всей фигуры (1 вершина)
    - Ось 1: диполь полушарий (2 крупных полигона)
    - Оси 2..3: квадранты (4 полигона)
    - Оси 4..7: октанты (8 полигонов)
    - Оси 8..15: 16 полигональных граней
    - Оси 16..31: 32 грани
    - Оси 32..63: 64 микрополигона рельефа!
    """
    mat = torch.zeros(d_model, d_model, dtype=torch.float32)
    mat[0, :] = 1.0 / math.sqrt(d_model)
    row = 1
    step = d_model
    while step >= 2:
        half = step // 2
        n_blocks = d_model // step
        scale = 1.0 / math.sqrt(step)
        for b in range(n_blocks):
            start = b * step
            mat[row, start : start + half] = scale
            mat[row, start + half : start + step] = -scale
            row += 1
        step = half
    return mat  # (64, 64), строго ортонормированная матрица: mat @ mat.T == I


class HolographicPolygonRGBL3DWorld(nn.Module):
    def __init__(self, vocab_size: int = 256, side: int = 8, n_objects: int = 96):
        super().__init__()
        self.vocab_size = vocab_size
        self.side = side
        self.d_model = side * side  # 64 полигональных сектора на 2D-оболочке фигуры
        self.n_objects = n_objects

        self.ext_embed = nn.Embedding(vocab_size, self.d_model)
        self.gen_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.gen_head.weight = self.ext_embed.weight

        # Внутренний оператор хода времени
        self.time_transition = nn.Linear(self.d_model, self.d_model)
        self.raw_time_leak = nn.Parameter(torch.zeros(self.d_model))

        # Сферическая сетка 3D-комнаты и ортонормированное полигональное дерево Nanite (64x64)
        u_dir, e_theta, e_phi = build_spherical_room_grid(side)
        self.register_buffer("u_dir", u_dir)  # (64, 3)
        self.register_buffer("poly_tree", build_haar_polygon_tree(self.d_model))  # (64, 64)

        # Базовый объем покоя фигуры и волновые числа 3 цветовых каналов (k_R, k_G, k_B) + коэффициент яркости L
        self.raw_rest_volume = nn.Parameter(torch.zeros(self.d_model))
        self.raw_color_k = nn.Parameter(torch.tensor([1.2, 2.8, 5.5]))
        self.raw_lum_gain = nn.Parameter(torch.zeros(self.d_model))

        # 3D-объекты в комнате (X, Y, Z)
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.3)
        self.obj_raw_radius = nn.Parameter(torch.zeros(n_objects))

        # Векторные грани каждого 3D-объекта в базисе полигонального дерева (от крупных граней 0..7 к мелким 32..63)
        # + 4-канальный световой фильтр объекта (R, G, B, L - цвет + яркость!)
        self.obj_in = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_rgbl_filter = nn.Parameter(torch.randn(n_objects, 4) * 0.25)

        # Локальные пороги вскрытия уровней полигонального дерева z_open[j, d] и импедансы z_imp[j, d]
        init_hierarchy = torch.linspace(-1.5, 2.5, self.d_model).unsqueeze(0).expand(n_objects, self.d_model).clone()
        self.raw_z_open = nn.Parameter(init_hierarchy)
        self.raw_z_imp = nn.Parameter(torch.zeros(n_objects, self.d_model))

        # Память 3D-среды Psi(t)
        self.raw_obj_retention = nn.Parameter(torch.zeros(n_objects))

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
        """
        Строит 3D-полигональный рельеф центральной фигуры и натягивает на полигоны процедурный свет RGBL:
        1) Внутренняя сложность C_int(u, v, t) от активных процессов внутри шара задает объем V(u, v, t).
        2) Вершины полигональной сетки в 3D-комнате:
             p(u, v, t) = R(u, v, t) * u_dir(u, v),   где R ~ V^(1/3) + рельеф от двусторонней интерференции.
        3) Для каждого полигона (u, v) строятся два его 3D-ребра к соседним вершинам сетки:
             edge_u = p(u+1, v) - p(u, v),
             edge_v = p(u, v+1) - p(u, v).
           Векторное произведение ребер cross_uv = edge_u x edge_v дает:
             - точную 3D-нормаль грани полигона n_poly = normalize(cross_uv),
             - точную 3D-площадь полигона dA_poly = ||cross_uv|| / ||cross_uv_base|| * V^(2/3)!
        4) На каждом полигоне из двусторонней интерференции (w_int <-> w_ext) и геометрии ребер
           детерминированно генерируются 4 канала света RGBL = (R, G, B, Luminance):
             - R, G, B — 3 спектральных цвета текстуры полигона,
             - L (Яркость) — энергетическая светимость полигона, усиливающая его площадь и дальность луча!
        """
        b = w_int.shape[0]
        v_0 = 0.6 + 0.4 * torch.sigmoid(self.raw_rest_volume).unsqueeze(0)  # (1, 64)

        # 1. Активная внутренняя сложность C_int(u, v, t) от задействованных внутренних граней
        c_int = torch.tanh(psi_field.abs().mean(dim=1) * 4.0)  # (B, 64)

        # 2. Двусторонняя интерференция на 2D-границе (внутренняя волна w_int <-> внешняя волна w_ext)
        eps_err = w_ext - w_int  # (B, 64)
        interf_power = 0.5 * (eps_err * eps_err)

        # Объем сектора V(u, v, t) и голографический масштаб площади V^(2/3)
        local_volume = v_0 + c_int + torch.tanh(interf_power)  # (B, 64)
        holo_area_2_3 = local_volume.pow(2.0 / 3.0)            # (B, 64) — теорема о 2/3!

        # Радиус 3D-вершин полигональной оболочки R(u, v, t) с упругим натяжением мембраны Лапласа:
        R_flat = local_volume.pow(1.0 / 3.0) * (1.0 + 0.35 * torch.tanh(w_int) + 0.45 * torch.tanh(eps_err))  # (B, 64) > 0
        # 3D-координаты вершин полигонов в комнате (X, Y, Z): (B, 8, 8, 3)
        p_surf = R_flat.unsqueeze(-1) * self.u_dir.unsqueeze(0)  # (B, 64, 3)
        p_grid = p_surf.view(b, self.side, self.side, 3)

        # 3. НАСТОЯЩАЯ ГЕОМЕТРИЯ 3D-ПОЛИГОНОВ ЧЕРЕЗ ВЕКТОРНОЕ ПРОИЗВЕДЕНИЕ 3D-РЕБЕР:
        # Соседние вершины полигона по широте (u) и долготе (v)
        p_next_v = torch.roll(p_grid, shifts=-1, dims=2)
        p_next_u = torch.zeros_like(p_grid)
        p_next_u[:, :-1, :, :] = p_grid[:, 1:, :, :]
        p_next_u[:, -1, :, :] = p_grid[:, -2, :, :]

        # 3D-векторы ребер полигона в пространстве (X, Y, Z)
        edge_u = (p_next_u - p_grid).reshape(b, self.d_model, 3)  # (B, 64, 3)
        edge_v = (p_next_v - p_grid).reshape(b, self.d_model, 3)  # (B, 64, 3)

        # Базовые ребра на единичной гладкой сфере (для точного отношения площади рельефного полигона к гладкому!)
        u_grid = self.u_dir.view(1, self.side, self.side, 3)
        u_next_v = torch.roll(u_grid, shifts=-1, dims=2)
        u_next_u = torch.zeros_like(u_grid)
        u_next_u[:, :-1, :, :] = u_grid[:, 1:, :, :]
        u_next_u[:, -1, :, :] = u_grid[:, -2, :, :]
        base_cross = torch.cross((u_next_u - u_grid).reshape(1, self.d_model, 3), (u_next_v - u_grid).reshape(1, self.d_model, 3), dim=-1)
        base_poly_area = torch.sqrt((base_cross * base_cross).sum(dim=-1) + 1e-8)  # (1, 64)

        # Векторное произведение реальных 3D-ребер рельефного полигона:
        poly_cross = torch.cross(edge_u, edge_v, dim=-1)  # (B, 64, 3)
        poly_area_3d = torch.sqrt((poly_cross * poly_cross).sum(dim=-1) + 1e-8)  # (B, 64)

        # Ориентируем нормаль полигона наружу (сонаправленно с u_dir)
        n_poly = poly_cross / poly_area_3d.unsqueeze(-1)
        flip = torch.sign((n_poly * self.u_dir.unsqueeze(0)).sum(dim=-1, keepdim=True) + 1e-6)
        n_poly = n_poly * flip

        # Излом между соседними полигонами (дискретный лапласиан радиуса по сетке)
        R_grid = R_flat.view(b, self.side, self.side)
        crease_f = (
            torch.roll(R_grid, shifts=-1, dims=2) + torch.roll(R_grid, shifts=1, dims=2) - 2.0 * R_grid
        ).reshape(b, self.d_model)

        # Относительная 3D-площадь рельефного полигона (точнoе отношение ||edge_u x edge_v|| / ||base_u x base_v|| ~ V^(2/3) * крутизна горы!):
        dA_poly = (poly_area_3d / base_poly_area) * torch.sqrt(1.0 + 0.5 * crease_f.pow(2))  # (B, 64)

        # 4. ПРОЦЕДУРНЫЕ ЦВЕТА (R, G, B) + ЯРКОСТЬ КАНАЛА (L / LUMINANCE) НА КАЖДОМ ПОЛИГОНЕ:
        k_rgb = F.softplus(self.raw_color_k)
        eps_unit = torch.tanh(eps_err)
        # Канал R (Красный): длинноволновая стоячая фаза на вершине полигона
        color_R = eps_unit * torch.cos(k_rgb[0] * R_flat)
        # Канал G (Зеленый): цвет наклона 3D-грани полигона (normal_tilt = 1 - |<n_poly, u_dir>|)
        normal_tilt = 1.0 - (n_poly * self.u_dir.unsqueeze(0)).sum(dim=-1).abs()
        color_G = eps_unit * normal_tilt * torch.cos(k_rgb[1] * R_flat) + torch.tanh(crease_f) * torch.sin(k_rgb[1] * R_flat)
        # Канал B (Синий): высокочастотный цвет излома граней (crease_f и внутренняя сложность c_int)
        color_B = torch.tanh(crease_f) * torch.cos(k_rgb[2] * eps_unit) + 0.5 * c_int * eps_unit

        # Канал L (ЯРКОСТЬ / LUMINANCE): энергетическая светимость полигона от мощности двустороннего резонанса
        # и активной внутренней сложности!
        lum_gain = torch.sigmoid(self.raw_lum_gain).unsqueeze(0)
        luminance_L = 1.0 + lum_gain * torch.tanh(interf_power + c_int * eps_err.abs())  # (B, 64) в [1, 2]

        # Вектор цвета и яркости RGBL на каждом полигоне: (B, 64, 4)
        proc_rgbl = torch.stack(
            [
                color_R * luminance_L,
                color_G * luminance_L,
                color_B * luminance_L,
                (luminance_L - 1.0) * eps_unit,
            ],
            dim=-1,
        )  # (B, 64, 4)

        return p_surf, n_poly, dA_poly, local_volume, luminance_L, proc_rgbl

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        w_int_state: torch.Tensor | None = None,
        psi_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        b, t = tokens.shape
        device = tokens.device
        dtype = self.obj_pos.dtype

        if w_int_state is None:
            w_int_state = torch.zeros(b, self.d_model, device=device, dtype=dtype)
        if psi_field is None:
            psi_field = torch.zeros(b, self.n_objects, self.d_model, device=device, dtype=dtype)

        conductance, edge_normal, radii = self.compute_physical_2d_collisions_between_objects()
        z_open = F.softplus(self.raw_z_open)  # (N, 64)
        z_imp = F.softplus(self.raw_z_imp) + 1e-4
        surface_threshold = z_open.min(dim=-1).values + z_imp.min(dim=-1).values

        retention = torch.sigmoid(self.raw_obj_retention).view(1, self.n_objects, 1)
        time_leak = torch.sigmoid(self.raw_time_leak).unsqueeze(0)
        rgbl_filter = torch.tanh(self.obj_rgbl_filter)  # (N, 4) — фильтр (R, G, B, L) каждого 3D-объекта

        w_ext_all = self.ext_embed(tokens)  # (B, T, 64)
        logits_list = []
        fe_total = w_ext_all.new_tensor(0.0)

        total_active_coords = 0.0
        total_touched_nodes = 0.0
        total_cascade_waves = 0.0
        total_holo_area = 0.0
        total_sphere_vol = 0.0
        total_luminance = 0.0
        total_err_before = 0.0
        total_err_after = 0.0

        w_int = w_int_state
        for step_t in range(t):
            w_ext = w_ext_all[:, step_t, :]
            eps_err = w_ext - w_int

            # 1. ПОЛИГОНАЛЬНЫЙ РЕЛЬЕФ ОБОЛОЧКИ, ГОЛОГРАФИЧЕСКАЯ ПЛОЩАДЬ dA_poly (~V^(2/3)) И ЦВЕТА + ЯРКОСТЬ (RGBL)
            p_surf, n_poly, dA_poly, sphere_vol, lum_L, proc_rgbl = self.compute_polygonal_relief_and_rgbl_hologram(
                w_int, w_ext, psi_field
            )

            # Полный световой поток каждого полигона (Фотометрия: Сила света = Яркость L * Площадь полигона dA_poly)!
            radiant_flux = lum_L * dA_poly  # (B, 64)
            holo_err = eps_err

            # 2. ПЕРСПЕКТИВА И ДАЛЬНОСТЬ ОТ ЯРКИХ ПОЛИГОНОВ РЕЛЬЕФА К 3D-УЗЛАМ КОМНАТЫ
            flux_weights = (eps_err.abs() * radiant_flux).unsqueeze(-1)  # (B, 64, 1)
            w_sum = flux_weights.sum(dim=1) + 1e-8
            active_surf_pos = (p_surf * flux_weights).sum(dim=1) / w_sum
            active_surf_norm = (n_poly * flux_weights).sum(dim=1) / w_sum
            active_surf_norm = active_surf_norm / torch.sqrt((active_surf_norm * active_surf_norm).sum(dim=-1, keepdim=True) + 1e-12)

            delta = self.obj_pos.unsqueeze(0) - active_surf_pos.unsqueeze(1)
            dist = torch.sqrt((delta * delta).sum(dim=-1) + 1e-12)
            dir_to_obj = delta / dist.unsqueeze(-1)

            cos_view = F.relu((active_surf_norm.unsqueeze(1) * dir_to_obj).sum(dim=-1))
            r_view = radii.unsqueeze(0)

            holo_energy_norm = torch.sqrt((holo_err * holo_err).sum(dim=-1, keepdim=True) + 1e-12)
            initial_energy = holo_energy_norm * cos_view * torch.exp(-dist / r_view)

            node_energy, n_waves = self.propagate_cascade_with_energy_conservation(
                initial_energy, active_surf_norm, conductance, edge_normal, surface_threshold
            )

            # 3. ТЕССЕЛЯЦИЯ (РАСЩЕПЛЕНИЕ) ПОЛИГОНАЛЬНОГО ДЕРЕВА NANITE ОТ ПОТОКА (ЯРКОСТЬ L * ПЛОЩАДЬ dA_poly):
            # Переводим сигнал рельефа и цветов RGBL в иерархический базис дерева полигонов (от крупных граней к мелким)
            sector_energy = node_energy.unsqueeze(-1) * radiant_flux.unsqueeze(1)  # (B, N, 64)
            excess = F.relu(sector_energy - z_open.unsqueeze(0))
            open_gate = excess / (excess + z_imp.unsqueeze(0))  # (B, N, 64), строгий 0.0 ниже порога!

            # 4. ПАРАЛЛЕЛЬНОЕ ВЫЧИСЛЕНИЕ ЧЕРЕЗ ПОЛИГОНАЛЬНОЕ ДЕРЕВО + ЦВЕТА И ЯРКОСТЬ (RGBL):
            # Сворачиваем 4 канала света RGBL каждого полигона с фильтром (R, G, B, L) каждого 3D-объекта:
            rgbl_modulated = torch.einsum("bdc,nc->bnd", proc_rgbl, rgbl_filter)  # (B, N, 64)
            surface_signal = holo_err.unsqueeze(1) + rgbl_modulated               # (B, N, 64)

            # Проекция в иерархическое дерево полигонов poly_tree (от LOD 0 до микрограней)
            tree_signal = torch.matmul(surface_signal, self.poly_tree.T)  # (B, N, 64)
            proj = (tree_signal * self.obj_in.unsqueeze(0) * open_gate).sum(dim=-1, keepdim=True)  # (B, N, 1)
            tree_response = torch.tanh(proj) * self.obj_out.unsqueeze(0) * open_gate               # (B, N, 64)

            # Возврат из дерева полигонов на 2D-оболочку рельефа
            facet_response = torch.matmul(tree_response, self.poly_tree)  # (B, N, 64)

            psi_field = retention * psi_field + (1.0 - retention) * facet_response
            correction = facet_response.sum(dim=1)  # (B, 64)

            eps_after = eps_err - correction
            mu_post = w_int + correction

            step_fe = 0.5 * (eps_after * eps_after).mean()
            fe_total = fe_total + step_fe / t

            env_memory = psi_field.sum(dim=1)
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
                total_err_before += float(eps_err.pow(2).mean().sqrt().item())
                total_err_after += float(eps_after.pow(2).mean().sqrt().item())

        logits = torch.stack(logits_list, dim=1)
        stats = {
            "active_coords_frac": total_active_coords / t,
            "touched_nodes_frac": total_touched_nodes / t,
            "avg_cascade_waves": total_cascade_waves / t,
            "holographic_area": total_holo_area / t,
            "sphere_volume": total_sphere_vol / t,
            "avg_luminance": total_luminance / t,
            "err_before": total_err_before / t,
            "err_after": total_err_after / t,
            "free_energy": float(fe_total.item()),
        }
        return logits, w_int.detach(), psi_field.detach(), fe_total, stats


def evaluate_holographic_world(
    model: HolographicPolygonRGBL3DWorld, data: torch.Tensor, chunk_len: int = 64
) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    agg: dict[str, float] = {}
    n_steps = 0
    w_int = None
    psi_field = None
    with torch.no_grad():
        limit = min(data.numel() - 1, chunk_len * 80)
        for start in range(0, limit - chunk_len, chunk_len):
            seq = data[start : start + chunk_len + 1]
            logits, w_int, psi_field, _, st = model.forward_sequence(seq[:-1].unsqueeze(0), w_int, psi_field)
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
    model = HolographicPolygonRGBL3DWorld(vocab_size=256, side=8, n_objects=96)
    n_params = sum(p.numel() for p in model.parameters())
    print("=== V8: ПОЛИГОНАЛЬНЫЙ 2D-РЕЛЬЕФ + ГОЛОГРАФИЧЕСКИЙ ПРИНЦИП (V^(2/3)) + ЦВЕТА (R,G,B) + ЯРКОСТЬ (L) ===", flush=True)
    print(f"Параметров: {n_params:,} | Полигоны рельефа + иерархия Nanite + 4 канала света RGBL!", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    w_int, psi_field = None, None
    cursor = 0
    t0 = time.perf_counter()
    history = []

    for step in range(1, steps + 1):
        if cursor + chunk_len + 1 >= stream_len:
            cursor = 0
            w_int, psi_field = None, None

        seq = streams[:, cursor : cursor + chunk_len + 1]
        cursor += chunk_len
        src, dst = seq[:, :-1], seq[:, 1:]

        logits, w_int, psi_field, fe_loss, st = model.forward_sequence(src, w_int, psi_field)
        ce_loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
        total_loss = ce_loss + fe_loss

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if step == 1 or step % 50 == 0:
            bpc = float(ce_loss.item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | v8_Полигоны_RGBL BPC={bpc:.3f} | "
                f"V={st['sphere_volume']:.2f} -> A_poly={st['holographic_area']:.2f}x | "
                f"Яркость L={st['avg_luminance']:.2f} | "
                f"невязка {st['err_before']:.3f}->{st['err_after']:.3f} | "
                f"волн={st['avg_cascade_waves']:.1f} | узлов={st['touched_nodes_frac']*100:.1f}% | "
                f"граней={st['active_coords_frac']*100:.1f}% | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append({"step": step, "bpc": round(bpc, 4), **{k: round(v, 4) for k, v in st.items()}})

    val_bpc, val_ppl, val_st = evaluate_holographic_world(model, valid_data, chunk_len)
    print("\n=== ИТОГИ V8 (ПОЛИГОНАЛЬНЫЙ РЕЛЬЕФ + ГОЛОГРАФИЯ V^(2/3) + ЦВЕТА RGB + ЯРКОСТЬ L) ===", flush=True)
    print(
        f"Validation BPC = {val_bpc:.3f} | Перплексия байта = {val_ppl:.2f} | "
        f"Объем шара V = {val_st['sphere_volume']:.2f}, Площадь полигонов A = {val_st['holographic_area']:.2f}x, "
        f"Яркость L = {val_st['avg_luminance']:.2f} | "
        f"Гашение невязки: {val_st['err_before']:.3f} -> {val_st['err_after']:.3f} | "
        f"Вскрыто полигональных граней: {val_st['active_coords_frac']*100:.1f}% (узлов: {val_st['touched_nodes_frac']*100:.1f}%)",
        flush=True,
    )

    out_path = OUT_DIR / "rezultat_holographic_relief_v8.json"
    out_path.write_text(
        json.dumps(
            {
                "params": n_params,
                "history": history,
                "validation": {"bpc": round(val_bpc, 4), "ppl": round(val_ppl, 2), "stats": val_st},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Сохранено в {out_path}", flush=True)


if __name__ == "__main__":
    main()
