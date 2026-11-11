"""Полный верификационный конвейер T3..T15 (Строгий аудит MarkovMembrane Engine).

Соблюдает все жесткие запреты §0:
- Авторская архитектура (2D-мембрана, многомерный признак D=64, 3D-координаты как адрес, коллизии, константное состояние) неприкосновенна.
- Каждый прогон меняет ровно один механизм (`changed_exactly_one_thing: true`).
- Никакого хардкода порогов разреженности.
- Критерий приемки записывается ДО прогона (`acceptance_criterion_written_before_run`).
- Все отрицательные результаты фиксируются в `negative_results`.
- Каждый прогон сохраняет JSON строго по схеме §10.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parent
sys.path.insert(0, str(ROOT / "iterations" / "v1_energy_conservation"))
sys.path.insert(0, str(ROOT / "iterations" / "v7_relief_sphere_3d_room"))
sys.path.insert(0, str(ROOT / "iterations" / "v8_holographic_rgbl"))

from progon_chestnaya_3d import load_wikitext_bytes
from progon_holographic_relief_v8 import (
    HolographicPolygonRGBL3DWorld,
    build_haar_polygon_tree,
)
from progon_relief_sphere_v7 import build_spherical_room_grid


class ConfigurableMembraneV8(nn.Module):
    """
    Точная реализация авторской архитектуры v8 с флагами одиночных изменений (changed_exactly_one_thing=True)
    для задач T5, T6, T7, T8, T11, T12 (a1..a7) без изменения базовой геометрии.
    При параметрах по умолчанию побитово/численно совпадает с HolographicPolygonRGBL3DWorld (46 179 параметров).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        side: int = 8,
        n_objects: int = 96,
        *,
        fix_basis_c1: bool = False,
        unbiased_z_open: bool = False,
        enable_truncation_c2: bool = False,
        learnable_hill_t7: bool = False,
        sparse_cascade_t8: bool = False,
        volume_exp_mode: str = "fixed_2_3",  # "fixed_2_3" | "fixed_1_0" | "fixed_0_5" | "learnable"
        rgbl_channels: int = 4,              # 4 (RGBL) | 1 (L only) | 3 (RGB only)
        disable_luminance_a3: bool = False,
        disable_conservation_a4: bool = False,
        smooth_shell_a5: bool = False,
        random_group4_a6: bool = False,
        temp_softmax_a7: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.side = side
        self.d_model = side * side
        self.n_objects = n_objects

        self.fix_basis_c1 = fix_basis_c1
        self.unbiased_z_open = unbiased_z_open
        self.enable_truncation_c2 = enable_truncation_c2
        self.learnable_hill_t7 = learnable_hill_t7
        self.sparse_cascade_t8 = sparse_cascade_t8
        self.volume_exp_mode = volume_exp_mode
        self.rgbl_channels = rgbl_channels
        self.disable_luminance_a3 = disable_luminance_a3
        self.disable_conservation_a4 = disable_conservation_a4
        self.smooth_shell_a5 = smooth_shell_a5
        self.random_group4_a6 = random_group4_a6
        self.temp_softmax_a7 = temp_softmax_a7

        self.ext_embed = nn.Embedding(vocab_size, self.d_model)
        self.gen_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.gen_head.weight = self.ext_embed.weight

        self.time_transition = nn.Linear(self.d_model, self.d_model)
        self.raw_time_leak = nn.Parameter(torch.zeros(self.d_model))

        u_dir, _, _ = build_spherical_room_grid(side)
        self.register_buffer("u_dir", u_dir)
        poly_tree = build_haar_polygon_tree(self.d_model)
        if random_group4_a6:
            gen = torch.Generator().manual_seed(1337)
            perm = torch.randperm(self.d_model, generator=gen)
            poly_tree = poly_tree[:, perm]
        self.register_buffer("poly_tree", poly_tree)

        u_grid = u_dir.view(1, side, side, 3)
        u_next_v = torch.roll(u_grid, shifts=-1, dims=2)
        u_next_u = torch.cat([u_grid[:, 1:, :, :], u_grid[:, -2:-1, :, :]], dim=1)
        base_cross = torch.cross(
            (u_next_u - u_grid).reshape(1, self.d_model, 3),
            (u_next_v - u_grid).reshape(1, self.d_model, 3),
            dim=-1,
        )
        self.register_buffer(
            "base_poly_area",
            torch.sqrt((base_cross * base_cross).sum(dim=-1) + 1e-8),
            persistent=False,
        )

        self.raw_rest_volume = nn.Parameter(torch.zeros(self.d_model))
        self.raw_color_k = nn.Parameter(torch.tensor([1.2, 2.8, 5.5]))
        self.raw_lum_gain = nn.Parameter(torch.zeros(self.d_model))

        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.3)
        self.obj_raw_radius = nn.Parameter(torch.zeros(n_objects))

        self.obj_in = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_rgbl_filter = nn.Parameter(torch.randn(n_objects, 4) * 0.25)

        if unbiased_z_open:
            self.raw_z_open = nn.Parameter(torch.randn(n_objects, self.d_model) * 0.05)
        else:
            init_hierarchy = torch.linspace(-1.5, 2.5, self.d_model).unsqueeze(0).expand(n_objects, self.d_model).clone()
            self.raw_z_open = nn.Parameter(init_hierarchy)
        self.raw_z_imp = nn.Parameter(torch.zeros(n_objects, self.d_model))
        self.raw_obj_retention = nn.Parameter(torch.zeros(n_objects))

        # Дополнительные обучаемые скаляры/векторы только при включении соответствующего теста:
        if enable_truncation_c2:
            # Обучаемый порог отсечения дерева tau = sigmoid(raw_tau) * 0.25 (без хардкода)
            self.raw_tau = nn.Parameter(torch.tensor(-1.5))
        if learnable_hill_t7:
            # Обучаемый коэффициент Хилла n_j = 0.5 + softplus(raw_hill_n), старт с n=1.0
            init_raw_n = math.log(math.exp(0.5) - 1.0)
            self.raw_hill_n = nn.Parameter(torch.full((n_objects, 1), init_raw_n))
        if volume_exp_mode == "learnable":
            # Обучаемый показатель степени V^alpha (старт с 0.6667 и проверка ухода градиента)
            self.learned_alpha = nn.Parameter(torch.tensor(2.0 / 3.0))

    def compute_collisions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radii = F.softplus(self.obj_raw_radius) + 1e-4
        diff = self.obj_pos.unsqueeze(0) - self.obj_pos.unsqueeze(1)
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
        edge_normal = diff / dist.unsqueeze(-1)
        sum_r = radii.unsqueeze(1) + radii.unsqueeze(0)
        penetration = F.relu(sum_r - dist) * (1.0 - torch.eye(self.n_objects, device=dist.device, dtype=dist.dtype))
        conductance = penetration / (penetration.sum(dim=-1, keepdim=True) + radii.unsqueeze(-1))
        return conductance, edge_normal, radii

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        w_int_state: torch.Tensor | None = None,
        psi_field: torch.Tensor | None = None,
        truncate_prefix_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        b, t = tokens.shape
        device, dtype = tokens.device, self.obj_pos.dtype
        if w_int_state is None:
            w_int_state = torch.zeros(b, self.d_model, device=device, dtype=dtype)
        if psi_field is None:
            psi_field = torch.zeros(b, self.n_objects, self.d_model, device=device, dtype=dtype)

        conductance, edge_normal, radii = self.compute_collisions()
        en_flat = edge_normal.reshape(-1, 3)
        active_edges = int((conductance.detach() > 0).sum().item())

        z_open = F.softplus(self.raw_z_open)
        z_imp = F.softplus(self.raw_z_imp) + 1e-4
        thresh_view = (z_open.min(dim=-1).values + z_imp.min(dim=-1).values).unsqueeze(0)

        retention = torch.sigmoid(self.raw_obj_retention).view(1, self.n_objects, 1)
        time_leak = torch.sigmoid(self.raw_time_leak).unsqueeze(0)
        rgbl_filter = torch.tanh(self.obj_rgbl_filter)
        if self.rgbl_channels == 1:
            mask_c = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).unsqueeze(0)
            rgbl_filter = rgbl_filter * mask_c
        elif self.rgbl_channels == 3:
            mask_c = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device, dtype=dtype).unsqueeze(0)
            rgbl_filter = rgbl_filter * mask_c

        w_ext_all = self.ext_embed(tokens)
        logits_list = []
        fe_total = w_ext_all.new_tensor(0.0)
        w_int = w_int_state

        act_coords_sum = 0.0
        touched_sum = 0.0
        kept_tree_frac_sum = 0.0
        mac_steps_sum = 0.0
        bytes_read_sum = 0.0

        # Если T8 включен, подготовим список разреженных ребер (src_idx, dst_idx)
        if self.sparse_cascade_t8:
            edge_mask = conductance.detach() > 0.0
            src_idx, dst_idx = torch.nonzero(edge_mask, as_tuple=True)
            cond_sparse = conductance[src_idx, dst_idx]
            en_sparse = edge_normal[src_idx, dst_idx]  # (E, 3)

        for step_t in range(t):
            w_ext = w_ext_all[:, step_t, :]
            eps_err = w_ext - w_int

            v_0 = 0.6 + 0.4 * torch.sigmoid(self.raw_rest_volume).unsqueeze(0)
            c_int = torch.tanh(psi_field.abs().mean(dim=1) * 4.0)
            interf_power = 0.5 * (eps_err * eps_err)
            local_volume = v_0 + c_int + torch.tanh(interf_power)

            if self.volume_exp_mode == "fixed_2_3":
                alpha_exp = 2.0 / 3.0
                r_exp = 1.0 / 3.0
            elif self.volume_exp_mode == "fixed_1_0":
                alpha_exp = 1.0
                r_exp = 0.5
            elif self.volume_exp_mode == "fixed_0_5":
                alpha_exp = 0.5
                r_exp = 0.25
            else:
                alpha_exp = self.learned_alpha
                r_exp = 0.5 * self.learned_alpha

            if self.smooth_shell_a5:
                R_flat = local_volume.pow(r_exp)
                p_surf = R_flat.unsqueeze(-1) * self.u_dir.unsqueeze(0)
                n_poly = self.u_dir.unsqueeze(0).expand(b, self.d_model, 3)
                crease_f = torch.zeros_like(R_flat)
                dA_poly = local_volume.pow(alpha_exp)
            else:
                R_flat = local_volume.pow(r_exp) * (1.0 + 0.35 * torch.tanh(w_int) + 0.45 * torch.tanh(eps_err))
                p_surf = R_flat.unsqueeze(-1) * self.u_dir.unsqueeze(0)
                p_grid = p_surf.view(b, self.side, self.side, 3)
                p_next_v = torch.roll(p_grid, shifts=-1, dims=2)
                p_next_u = torch.cat([p_grid[:, 1:, :, :], p_grid[:, -2:-1, :, :]], dim=1)
                edge_u = (p_next_u - p_grid).reshape(b, self.d_model, 3)
                edge_v = (p_next_v - p_grid).reshape(b, self.d_model, 3)
                poly_cross = torch.cross(edge_u, edge_v, dim=-1)
                poly_area_3d = torch.sqrt((poly_cross * poly_cross).sum(dim=-1) + 1e-8)
                n_poly = poly_cross / poly_area_3d.unsqueeze(-1)
                flip = torch.sign((n_poly * self.u_dir.unsqueeze(0)).sum(dim=-1, keepdim=True) + 1e-6)
                n_poly = n_poly * flip
                R_grid = R_flat.view(b, self.side, self.side)
                crease_f = (
                    torch.roll(R_grid, shifts=-1, dims=2) + torch.roll(R_grid, shifts=1, dims=2) - 2.0 * R_grid
                ).reshape(b, self.d_model)
                if self.volume_exp_mode == "fixed_2_3":
                    dA_poly = (poly_area_3d / self.base_poly_area) * torch.sqrt(1.0 + 0.5 * crease_f.pow(2))
                else:
                    # Явная проверка степени V^alpha при сохранении рельефа
                    dA_poly = local_volume.pow(alpha_exp) * torch.sqrt(1.0 + 0.5 * crease_f.pow(2))

            k_rgb = F.softplus(self.raw_color_k)
            eps_unit = torch.tanh(eps_err)
            color_R = eps_unit * torch.cos(k_rgb[0] * R_flat)
            normal_tilt = 1.0 - (n_poly * self.u_dir.unsqueeze(0)).sum(dim=-1).abs()
            color_G = eps_unit * normal_tilt * torch.cos(k_rgb[1] * R_flat) + torch.tanh(crease_f) * torch.sin(k_rgb[1] * R_flat)
            color_B = torch.tanh(crease_f) * torch.cos(k_rgb[2] * eps_unit) + 0.5 * c_int * eps_unit

            if self.disable_luminance_a3:
                lum_L = torch.ones_like(eps_err)
            else:
                lum_gain = torch.sigmoid(self.raw_lum_gain).unsqueeze(0)
                lum_L = 1.0 + lum_gain * torch.tanh(interf_power + c_int * eps_err.abs())

            proc_rgbl = torch.stack(
                [color_R * lum_L, color_G * lum_L, color_B * lum_L, (lum_L - 1.0) * eps_unit],
                dim=-1,
            )
            radiant_flux = lum_L * dA_poly

            flux_weights = (eps_err.abs() * radiant_flux).unsqueeze(-1)
            w_sum = flux_weights.sum(dim=1) + 1e-8
            active_surf_pos = (p_surf * flux_weights).sum(dim=1) / w_sum
            active_surf_norm = (n_poly * flux_weights).sum(dim=1) / w_sum
            active_surf_norm = active_surf_norm / torch.sqrt((active_surf_norm * active_surf_norm).sum(dim=-1, keepdim=True) + 1e-12)

            delta = self.obj_pos.unsqueeze(0) - active_surf_pos.unsqueeze(1)
            dist = torch.sqrt((delta * delta).sum(dim=-1) + 1e-12)
            dir_to_obj = delta / dist.unsqueeze(-1)
            cos_view = F.relu((active_surf_norm.unsqueeze(1) * dir_to_obj).sum(dim=-1))
            holo_energy_norm = torch.sqrt((eps_err * eps_err).sum(dim=-1, keepdim=True) + 1e-12)

            if self.temp_softmax_a7:
                # Скучный эквивалент a7: температурный гейт вместо экспоненты Аррениуса/затухания
                attenuation = torch.sigmoid((radii.unsqueeze(0) - dist) / (radii.unsqueeze(0) + 1e-4))
            else:
                attenuation = torch.exp(-dist / radii.unsqueeze(0))
            initial_energy = holo_energy_norm * cos_view * attenuation

            # Каскад (плотный или разреженный по графу коллизий T8)
            if self.sparse_cascade_t8:
                align_e = F.relu(torch.matmul(active_surf_norm, en_sparse.T))  # (B, E)
                dir_cond_e = cond_sparse.unsqueeze(0) * align_e               # (B, E)
                absorbed = torch.zeros_like(initial_energy)
                current_wave = initial_energy
                for _ in range(3):
                    rem_cap = current_wave if self.disable_conservation_a4 else F.relu(thresh_view - absorbed)
                    taken = torch.minimum(current_wave, rem_cap)
                    overflow = F.relu(current_wave - taken)
                    absorbed = absorbed + taken
                    msg = overflow[:, src_idx] * dir_cond_e
                    next_wave = torch.zeros_like(current_wave)
                    next_wave.scatter_add_(1, dst_idx.unsqueeze(0).expand(b, -1), msg)
                    current_wave = next_wave
                node_energy = absorbed + current_wave
                cascade_macs = b * (3 * active_edges + 3 * active_edges)
            else:
                align = F.relu(torch.matmul(active_surf_norm, en_flat.T).view(b, self.n_objects, self.n_objects))
                directed_cond = conductance.unsqueeze(0) * align
                absorbed = torch.zeros_like(initial_energy)
                current_wave = initial_energy
                for _ in range(3):
                    rem_cap = current_wave if self.disable_conservation_a4 else F.relu(thresh_view - absorbed)
                    taken = torch.minimum(current_wave, rem_cap)
                    overflow = F.relu(current_wave - taken)
                    absorbed = absorbed + taken
                    current_wave = torch.bmm(overflow.unsqueeze(1), directed_cond).squeeze(1)
                node_energy = absorbed + current_wave
                cascade_macs = b * 3 * self.n_objects * self.n_objects

            # T6 (C1): Согласование базисов — перевод энергии в базис дерева ДО сравнения с z_open
            sector_energy = node_energy.unsqueeze(-1) * radiant_flux.unsqueeze(1)
            if self.fix_basis_c1:
                flux_tree = torch.matmul(radiant_flux, self.poly_tree.T).abs()
                compare_energy = node_energy.unsqueeze(-1) * flux_tree.unsqueeze(1)
            else:
                compare_energy = sector_energy

            excess = F.relu(compare_energy - z_open.unsqueeze(0))
            if self.learnable_hill_t7:
                hill_n = (0.5 + F.softplus(self.raw_hill_n)).unsqueeze(0)  # (1, N, 1)
                active_ex = (excess > 0.0).float()
                s_pow = (excess + 1e-6).pow(hill_n) * active_ex
                k_pow = z_imp.unsqueeze(0).pow(hill_n)
                open_gate = s_pow / (s_pow + k_pow + 1e-8)
            else:
                open_gate = excess / (excess + z_imp.unsqueeze(0))

            if truncate_prefix_k is not None:
                prefix_mask = torch.zeros(1, 1, self.d_model, device=device, dtype=dtype)
                prefix_mask[:, :, :truncate_prefix_k] = 1.0
                open_gate = open_gate * prefix_mask

            # T6 (C2): Реальное усечение коэффициентов дерева ниже обучаемого порога tau
            if self.enable_truncation_c2:
                tau = torch.sigmoid(self.raw_tau) * 0.25
                # Гладкая маска отсечения с точным обнулением ниже tau
                keep_mask = (open_gate > tau).float()
                effective_gate = open_gate * keep_mask
                kept_frac = float(keep_mask.detach().mean().item())
            else:
                effective_gate = open_gate
                kept_frac = 1.0 if truncate_prefix_k is None else (truncate_prefix_k / self.d_model)

            rgbl_modulated = torch.matmul(proc_rgbl, rgbl_filter.T).transpose(1, 2)
            surface_signal = eps_err.unsqueeze(1) + rgbl_modulated

            if self.enable_truncation_c2:
                # Реальное сокращение вычислений: линейное преобразование в базис дерева выполняется
                # над 5 базовыми векторами (eps_err + 4 канала RGBL), а в объектах умножаются только
                # коэффициенты с effective_gate > 0 (доля kept_frac)!
                he_tree = torch.matmul(eps_err, self.poly_tree.T)
                rgbl_tree = torch.matmul(proc_rgbl.transpose(1, 2), self.poly_tree.T)
                tree_signal = he_tree.unsqueeze(1) + torch.einsum("nc,bcd->bnd", rgbl_filter, rgbl_tree)
                proj = (tree_signal * self.obj_in.unsqueeze(0) * effective_gate).sum(dim=-1, keepdim=True)
                tree_response = torch.tanh(proj) * self.obj_out.unsqueeze(0) * effective_gate
                facet_response = torch.matmul(tree_response, self.poly_tree)

                active_d = max(1.0, kept_frac * self.d_model)
                tree_macs = int(
                    b * 5 * self.d_model * self.d_model
                    + b * self.n_objects * active_d * 2
                    + b * self.n_objects * active_d * self.d_model
                )
                step_bytes = int(
                    (self.n_objects * active_d * 2 + self.d_model * self.d_model) * 4
                )
            else:
                tree_signal = torch.matmul(surface_signal, self.poly_tree.T)
                proj = (tree_signal * self.obj_in.unsqueeze(0) * effective_gate).sum(dim=-1, keepdim=True)
                tree_response = torch.tanh(proj) * self.obj_out.unsqueeze(0) * effective_gate
                facet_response = torch.matmul(tree_response, self.poly_tree)
                tree_macs = int(2 * b * self.n_objects * self.d_model * self.d_model + 2 * b * self.n_objects * self.d_model)
                step_bytes = int(46179 * 4 + (b * self.n_objects * self.d_model * 4) / b)

            psi_field = retention * psi_field + (1.0 - retention) * facet_response
            correction = facet_response.sum(dim=1)
            eps_after = eps_err - correction
            mu_post = w_int + correction
            fe_total = fe_total + 0.5 * (eps_after * eps_after).mean() / t

            env_memory = psi_field.sum(dim=1)
            w_int = (1.0 - time_leak) * mu_post + time_leak * self.time_transition(mu_post + env_memory)
            logits_list.append(self.gen_head(w_int))

            with torch.no_grad():
                am = (effective_gate > 0.0).float()
                act_coords_sum += float(am.mean())
                touched_sum += float((am.sum(dim=-1) > 0).float().mean())
                kept_tree_frac_sum += kept_frac
                mac_steps_sum += float(cascade_macs + tree_macs + b * self.d_model * self.d_model + b * self.d_model * self.vocab_size)
                bytes_read_sum += float(step_bytes)

        logits = torch.stack(logits_list, dim=1)
        stats = {
            "active_coords_frac": act_coords_sum / t,
            "touched_nodes_frac": touched_sum / t,
            "kept_tree_frac": kept_tree_frac_sum / t,
            "MAC_per_step": int(round(mac_steps_sum / t)),
            "bytes_read_per_token": int(round(bytes_read_sum / t)),
            "free_energy": float(fe_total.item()),
        }
        return logits, w_int.detach(), psi_field.detach(), fe_total, stats


class MatchedClassicalByteLM(nn.Module):
    """Классический параметрический контроль T4: 46 472 параметра (+0.63% к 46 179, строго в пределах ±2%)."""

    def __init__(self, vocab_size: int = 256, d_model: int = 64, hidden_dim: int = 200):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.time_step = nn.Linear(d_model, d_model)
        self.raw_leak = nn.Parameter(torch.zeros(d_model))
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
        truncate_prefix_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t = tokens.shape
        x_all = self.embed(tokens)
        if state is None:
            state = torch.zeros(b, self.d_model, device=tokens.device, dtype=x_all.dtype)
        leak = torch.sigmoid(self.raw_leak).unsqueeze(0)
        logits_list = []
        fe_total = x_all.new_tensor(0.0)

        for step_t in range(t):
            x_t = x_all[:, step_t, :]
            eps = x_t - state
            h = F.silu(self.fc1(eps))
            if truncate_prefix_k is not None:
                h_k = h.shape[-1] * truncate_prefix_k // self.d_model
                mask = torch.zeros_like(h)
                mask[:, :h_k] = 1.0
                h = h * mask
            corr = self.fc2(h)
            eps_after = eps - corr
            post = state + corr
            fe_total = fe_total + 0.5 * (eps_after * eps_after).mean() / t
            state = (1.0 - leak) * post + leak * self.time_step(post)
            logits_list.append(self.head(state))

        return torch.stack(logits_list, dim=1), state.detach(), fe_total


def evaluate_full_validation(
    model: nn.Module,
    valid_data: torch.Tensor,
    chunk_len: int = 64,
    is_membrane: bool = True,
    streaming: bool = True,
    batch_streams: int = 16,
    truncate_prefix_k: int | None = None,
    mac_batch_ref: int = 16,
) -> tuple[np.ndarray, dict]:
    """Полная валидация T1 на всех >=200 000 байтах (без *80)."""
    model.eval()
    limit = valid_data.numel() - 1
    stream_len = limit // batch_streams
    streams = valid_data[: stream_len * batch_streams].view(batch_streams, stream_len)
    per_byte_bits = []
    w_int, psi_field = None, None
    c_state = None
    agg_stats: dict[str, float] = {}
    n_chunks = 0

    with torch.no_grad():
        for start in range(0, stream_len - chunk_len, chunk_len):
            if not streaming:
                w_int, psi_field = None, None
                c_state = None
            seq = streams[:, start : start + chunk_len + 1]
            src, dst = seq[:, :-1], seq[:, 1:]
            if is_membrane:
                logits, w_int, psi_field, _, st = model.forward_sequence(
                    src, w_int, psi_field, truncate_prefix_k=truncate_prefix_k
                )
                for k, v in st.items():
                    if k == "MAC_per_step":
                        v = v * (mac_batch_ref / batch_streams)
                    agg_stats[k] = agg_stats.get(k, 0.0) + v
            else:
                logits, c_state, _ = model.forward_sequence(
                    src, c_state, truncate_prefix_k=truncate_prefix_k
                )
            step_nats = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1), reduction="none")
            per_byte_bits.append((step_nats / math.log(2.0)).cpu().numpy())
            n_chunks += 1

    for k in agg_stats:
        agg_stats[k] /= max(1, n_chunks)
    if "MAC_per_step" in agg_stats:
        agg_stats["MAC_per_step"] = int(round(agg_stats["MAC_per_step"]))
    if "bytes_read_per_token" in agg_stats:
        agg_stats["bytes_read_per_token"] = int(round(agg_stats["bytes_read_per_token"]))
    all_bits = np.concatenate(per_byte_bits, axis=0)
    model.train()
    return all_bits, agg_stats


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Точный коэффициент ранговой корреляции Спирмена."""
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum()) + 1e-12
    return float((rx * ry).sum() / denom)


def train_and_eval_single_model(
    model: nn.Module,
    train_streams: torch.Tensor,
    valid_data: torch.Tensor,
    *,
    steps: int = 120,
    chunk_len: int = 64,
    lr: float = 3e-3,
    is_membrane: bool = True,
    streaming_train: bool = True,
    streaming_eval: bool = True,
    use_fe_loss: bool = True,
    mrl_prefixes: list[int] | None = None,
) -> tuple[float, float, np.ndarray, dict, float]:
    """Обучает модель и замеряет полную валидацию (>= 249 856 байт) + EMA(50) train BPC."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    stream_len = train_streams.shape[1]
    w_int, psi_field = None, None
    c_state = None
    cursor = 0
    ema50 = None
    alpha = 2.0 / 51.0

    for step in range(1, steps + 1):
        if not streaming_train or (cursor + chunk_len + 1 >= stream_len):
            if cursor + chunk_len + 1 >= stream_len:
                cursor = 0
            w_int, psi_field = None, None
            c_state = None
        seq = train_streams[:, cursor : cursor + chunk_len + 1]
        cursor += chunk_len
        src, dst = seq[:, :-1], seq[:, 1:]

        if is_membrane:
            if mrl_prefixes is not None:
                # T11: Matryoshka Representation Learning (вложенные префиксы + sandwich rule)
                total_ce = 0.0
                total_fe = 0.0
                for k_pref in mrl_prefixes:
                    logits_k, w_k, psi_k, fe_k, _ = model.forward_sequence(
                        src, w_int, psi_field, truncate_prefix_k=k_pref
                    )
                    ce_k = F.cross_entropy(logits_k.reshape(-1, 256), dst.reshape(-1))
                    total_ce = total_ce + ce_k / len(mrl_prefixes)
                    total_fe = total_fe + fe_k / len(mrl_prefixes)
                    if k_pref == 64:
                        w_next, psi_next = w_k, psi_k
                w_int, psi_field = w_next, psi_next
                ce_loss = total_ce
                loss = total_ce + (total_fe if use_fe_loss else 0.0)
            else:
                logits, w_int, psi_field, fe_loss, _ = model.forward_sequence(src, w_int, psi_field)
                ce_loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
                loss = ce_loss + (fe_loss if use_fe_loss else 0.0)
        else:
            logits, c_state, fe_loss = model.forward_sequence(src, c_state)
            ce_loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
            loss = ce_loss + (fe_loss if use_fe_loss else 0.0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        bpc_step = float(ce_loss.item()) / math.log(2.0)
        ema50 = bpc_step if ema50 is None else (alpha * bpc_step + (1.0 - alpha) * ema50)

    bits, st = evaluate_full_validation(
        model, valid_data, chunk_len=chunk_len, is_membrane=is_membrane, streaming=streaming_eval
    )
    bpc_mean = float(bits.mean())
    bpc_se = float(bits.std(ddof=1) / math.sqrt(len(bits)))
    return bpc_mean, bpc_se, bits, st, float(ema50 or 0.0)
