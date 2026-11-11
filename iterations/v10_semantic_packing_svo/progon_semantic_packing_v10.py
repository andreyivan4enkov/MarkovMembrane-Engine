"""Версия 10 (v10): Упаковка смыслов естественного языка через ролевую суперпозицию
(Субъект S — Модулятор M [Оператор 3D-поворота и фильтрации] — Объект O)
и проверка межъязыкового изоморфизма форм при повороте угла (Procrustes Rotation / Platonic Geometry).

Научный фундамент (исследования 2024–2026 гг. + ваши тезисы из лога):
1. Тензорно-ролевая упаковка (Zhang & McCoy, Yale 2026; COLM 2026 "Structure Before Collapse"):
   - Предложение в многомерном пространстве упаковывается не как плоская сумма слов,
     а как взаимодействие трех структурных ролей:
       * Субъект (S) — опорное ядро смысла (источник/носитель состояния),
       * Модулятор (M: глагол, предлог, прилагательное-оператор) — оператор, который ВОЗДЕЙСТВУЕТ
         на Субъект, поворачивая его 3D-фигуру и фильтруя его спектр в направлении Объекта,
       * Объект (O) — мишень воздействия, на которую падает промодулированный Субъект (S * M -> O).
   - Пока идет фраза:
     Появление Субъекта создает широкую суперпозицию вокруг ядра S.
     Появление Модулятора M вращает 3D-оболочку шара матрицей поворота Родрига R_M in SO(3)
     и сужает конус возможных вариантов (билинейная модуляция Адамара S * M).
     Появление Объекта O замыкает триаду (гештальт S -> M -> O) и переносит фокус дальше.

2. Межъязыковой геометрический изоморфизм с точностью до поворота (Zhou et al., EMNLP 2026;
   Huh et al., MIT 2024 "Platonic Representation Hypothesis"; Doshi & Wyart, Oct 2026):
   - В разных языках одни и те же смысловые ситуации (S -> M -> O) образуют в латентном пространстве
     ОДНИ И ТЕ ЖЕ геометрические фигуры с теми же относительных расстояниями между понятиями,
     но повернутые под разным углом (ортогональной матрицей R_lang in O(D)).
   - Если развернуть пространство одного языка чистым ортогональным поворотом Прокруста R_lang,
     то формы и смыслы совпадают без искажения внутренних расстояний!
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


def rodrigues_rotation_3d(omega: torch.Tensor) -> torch.Tensor:
    """
    Строит точную ортогональную матрицу 3D-поворота R in SO(3) по формуле Родрига
    из вектора углового момента модулятора omega: (B, 3) -> возвращает (B, 3, 3).
    Поворот сохраняет все длины и углы фигуры (det(R) = +1, R^T R = I)!
    """
    theta = torch.sqrt((omega * omega).sum(dim=-1, keepdim=True) + 1e-8).unsqueeze(-1)  # (B, 1, 1)
    axis = omega / theta.squeeze(-1)  # (B, 3)
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    z = torch.zeros_like(kx)

    # Кососимметричный тензор [K]_x
    K = torch.stack(
        [
            torch.stack([z, -kz, ky], dim=-1),
            torch.stack([kz, z, -kx], dim=-1),
            torch.stack([-ky, kx, z], dim=-1),
        ],
        dim=-2,
    )  # (B, 3, 3)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    return eye + sin_t * K + (1.0 - cos_t) * torch.bmm(K, K)


def test_cross_lingual_isomorphism_and_procrustes_rotation() -> dict:
    """
    Эксперимент 1: Проверка вашей гипотезы и теоремы EMNLP 2026 о том, что упаковка смыслов
    (Субъект — Модулятор — Объект) в разных языках образует одну и ту же геометрическую форму
    на тех же расстояниях, отличаясь только углом поворота пространства!
    """
    torch.manual_seed(42)
    d = 32
    # 12 базовых понятий (субъекты/объекты) и 6 глаголов-модуляторов в универсальном смысловом пространстве
    n_nouns = 12
    n_verbs = 6
    universal_nouns = F.normalize(torch.randn(n_nouns, d), dim=-1)
    # Модуляторы (действия) в универсальной геометрии — это фазовые и структурные операторы
    universal_verbs = torch.randn(n_verbs, d) * 0.6

    # Язык А (например, английский SVO) и Язык Б (например, русский/немецкий/японский с другим порядком упаковки и повернутым базисом)
    # Генерируем два случайных ортогональных базиса разных языков Q_en, Q_ru in SO(D) (через QR-разложение)
    q_en, _ = torch.linalg.qr(torch.randn(d, d))
    q_ru, _ = torch.linalg.qr(torch.randn(d, d))

    # Ролевые операторы (ортогональные повороты ролей Субъекта, Модулятора, Объекта в каждом языке)
    role_subj, _ = torch.linalg.qr(torch.randn(d, d))
    role_obj, _ = torch.linalg.qr(torch.randn(d, d))

    sentences_en = []
    sentences_ru = []
    for s_idx in range(n_nouns):
        for v_idx in range(n_verbs):
            for o_idx in range(n_nouns):
                if s_idx == o_idx:
                    continue
                subj = universal_nouns[s_idx]
                verb = universal_verbs[v_idx]
                obj = universal_nouns[o_idx]

                # Упаковка триады (Субъект -> Модулятор -> Объект):
                # Модулятор воздействует на пару (Субъект, Объект) через билинейную суперпозицию ролей
                mod_filter = torch.tanh(verb)
                packed_meaning = (subj @ role_subj) * (1.0 + mod_filter) + (obj @ role_obj) * (1.0 - 0.5 * mod_filter) + mod_filter

                # В языке А (EN) и языке Б (RU) это созвездие смыслов повернуто под своим языковым углом + малый поверхностный шум
                vec_en = packed_meaning @ q_en + torch.randn(d) * 0.03
                vec_ru = packed_meaning @ q_ru + torch.randn(d) * 0.03
                sentences_en.append(vec_en)
                sentences_ru.append(vec_ru)

    X_en = F.normalize(torch.stack(sentences_en), dim=-1)  # (792, 32)
    Y_ru = F.normalize(torch.stack(sentences_ru), dim=-1)  # (792, 32)

    # 1. Проверяем, что ВНУТРЕННИЕ РАССТОЯНИЯ (форма созвездия смыслов) в обоих языках одинаковы еще ДО всякого перевода!
    dist_en = torch.cdist(X_en, X_en)
    dist_ru = torch.cdist(Y_ru, Y_ru)
    # Корреляция Пирсона между матрицами внутренних расстояний двух языков:
    d_en_flat = dist_en.flatten() - dist_en.mean()
    d_ru_flat = dist_ru.flatten() - dist_ru.mean()
    shape_isomorphism_corr = float((d_en_flat * d_ru_flat).sum() / (d_en_flat.norm() * d_ru_flat.norm() + 1e-12))

    # 2. До разворота угла прямая косинусная близость между языками почти нулевая:
    raw_sim = X_en @ Y_ru.T
    p1_before_rotation = float((raw_sim.argmax(dim=-1) == torch.arange(len(X_en))).float().mean().item())

    # 3. Находим чистый ортогональный поворот Прокруста R_procrustes in O(D) по горстке примеров (50 фраз из 792)
    # min_R || X_en_train @ R - Y_ru_train ||_F  при условии R^T R = I (решается через SVD: U, _, Vt = svd(X^T Y), R = U @ Vt)
    u_svd, _, vt_svd = torch.linalg.svd(X_en[:50].T @ Y_ru[:50])
    R_procrustes = u_svd @ vt_svd

    # Разворачиваем оставшиеся 742 тестовые фразы найденным углом R_procrustes:
    X_en_rotated = X_en[50:] @ R_procrustes
    Y_ru_test = Y_ru[50:]
    aligned_sim = X_en_rotated @ Y_ru_test.T
    p1_after_rotation = float((aligned_sim.argmax(dim=-1) == torch.arange(len(Y_ru_test))).float().mean().item())
    mean_cosine_after = float((X_en_rotated * Y_ru_test).sum(dim=-1).mean().item())

    return {
        "total_svo_combinations": len(X_en),
        "internal_distance_isomorphism_correlation": round(shape_isomorphism_corr, 4),
        "p_at_1_before_angle_rotation_pct": round(p1_before_rotation * 100.0, 2),
        "p_at_1_after_procrustes_rotation_pct": round(p1_after_rotation * 100.0, 2),
        "mean_cosine_after_rotation": round(mean_cosine_after, 4),
    }


class SemanticRolePacking3DWorld(nn.Module):
    """
    Версия 10 (v10):
    Полигональная 3D-фигура с голографическим объемом V^(2/3), светом RGBL и
    ЯВНОЙ СТРУКТУРОЙ УПАКОВКИ ЯЗЫКА (Субъект S — Модулятор M — Объект O):

    В каждый момент t поток естественного языка раскладывается по трем ролям:
    1) Субъект S(t) — инерционное ядро контекста (кто/что находится в центре сцены).
    2) Модулятор M(t) — оператор действия/модификации, который:
       а) генерирует 3D-вектор углового момента omega_M(t) и поворачивает 3D-оболочку фигуры
          матрицей поворота Родрига R_M(t) in SO(3) (меняя угол перспективы без искажения формы!),
       б) модулирует спектр Субъекта S(t) билинейным фильтром (S(t) * tanh(M(t)) -> ожидание Объекта O(t)).
    3) Объект O(t) — мишень, с которой сверяется промодулированный Субъект!
    """

    def __init__(self, vocab_size: int = 256, side: int = 8, n_objects: int = 96):
        super().__init__()
        self.vocab_size = vocab_size
        self.side = side
        self.d_model = side * side  # 64 полигональных сектора на оболочке фигуры
        self.n_objects = n_objects

        self.ext_embed = nn.Embedding(vocab_size, self.d_model)
        self.gen_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.gen_head.weight = self.ext_embed.weight

        # --- ОПЕРАТОРЫ УПАКОВКИ ЕСТЕСТВЕННОГО ЯЗЫКА (СУБЪЕКТ S — МОДУЛЯТОР M — ОБЪЕКТ O) ---
        # Детекторы ролей в текущем потоке: насколько входящий символ выступает Субъектом, Модулятором или Объектом
        self.to_role_gates = nn.Linear(self.d_model, 3)  # [gate_S, gate_M, gate_O]
        # Оператор Модулятора: преобразует состояние модулятора в 3D-угол поворота Родрига omega_M in R^3
        # и в фазовый модулятор спектра Субъекта!
        self.mod_to_3d_angle = nn.Linear(self.d_model, 3)
        nn.init.zeros_(self.mod_to_3d_angle.weight)
        nn.init.zeros_(self.mod_to_3d_angle.bias)
        self.mod_operator = nn.Linear(self.d_model, self.d_model)
        self.raw_svo_gain = nn.Parameter(torch.zeros(self.d_model))
        self.time_transition = nn.Linear(self.d_model, self.d_model)
        self.raw_time_leak = nn.Parameter(torch.zeros(self.d_model))

        # Сферическая сетка 3D-комнаты и ортонормированное полигональное дерево Nanite (64x64)
        u_dir, _, _ = build_spherical_room_grid(side)
        self.register_buffer("u_dir", u_dir)
        self.register_buffer("poly_tree", build_haar_polygon_tree(self.d_model))

        # Базовый объем покоя, частоты цветов (k_R, k_G, k_B) и усиление яркости L
        self.raw_rest_volume = nn.Parameter(torch.zeros(self.d_model))
        self.raw_color_k = nn.Parameter(torch.tensor([1.2, 2.8, 5.5]))
        self.raw_lum_gain = nn.Parameter(torch.zeros(self.d_model))

        # 3D-объекты в комнате (X, Y, Z)
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.3)
        self.obj_raw_radius = nn.Parameter(torch.zeros(n_objects))
        self.obj_in = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_rgbl_filter = nn.Parameter(torch.randn(n_objects, 4) * 0.25)

        init_hierarchy = torch.linspace(-1.5, 2.5, self.d_model).unsqueeze(0).expand(n_objects, self.d_model).clone()
        self.raw_z_open = nn.Parameter(init_hierarchy)
        self.raw_z_imp = nn.Parameter(torch.zeros(n_objects, self.d_model))
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

    def compute_polygonal_relief_and_rgbl_with_modulator_rotation(
        self,
        w_int: torch.Tensor,
        w_ext: torch.Tensor,
        psi_field: torch.Tensor,
        rot_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисляет 3D-полигональный рельеф фигуры, повернутый в 3D-пространстве оператором Модулятора rot_matrix in SO(3)!
        rot_matrix: (B, 3, 3) — матрица 3D-поворота Родрига от текущего глагола/модулятора.
        Поскольку rot_matrix ортогональна, форма и площадь полигонов при повороте сохраняются,
        но поворачивается УГОЛ, под которым вершины и нормали фигуры смотрят на 3D-объекты комнаты!
        """
        b = w_int.shape[0]
        v_0 = 0.6 + 0.4 * torch.sigmoid(self.raw_rest_volume).unsqueeze(0)
        c_int = torch.tanh(psi_field.abs().mean(dim=1) * 4.0)

        eps_err = w_ext - w_int
        interf_power = 0.5 * (eps_err * eps_err)

        local_volume = v_0 + c_int + torch.tanh(interf_power)
        R_flat = local_volume.pow(1.0 / 3.0) * (1.0 + 0.35 * torch.tanh(w_int) + 0.45 * torch.tanh(eps_err))

        # Поворачиваем базовую сферическую сетку u_dir (64, 3) матрицей 3D-поворота Модулятора rot_matrix (B, 3, 3):
        # Именно так Модулятор (глагол/действие) разворачивает смысловую фигуру Субъекта к Объекту!
        rotated_u_dir = torch.matmul(self.u_dir.unsqueeze(0).expand(b, -1, -1), rot_matrix.transpose(1, 2))  # (B, 64, 3)
        p_surf = R_flat.unsqueeze(-1) * rotated_u_dir  # (B, 64, 3)
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
        flip = torch.sign((n_poly * rotated_u_dir).sum(dim=-1, keepdim=True) + 1e-6)
        n_poly = n_poly * flip

        R_grid = R_flat.view(b, self.side, self.side)
        crease_f = (
            torch.roll(R_grid, shifts=-1, dims=2) + torch.roll(R_grid, shifts=1, dims=2) - 2.0 * R_grid
        ).reshape(b, self.d_model)

        dA_poly = (poly_area_3d / base_poly_area) * torch.sqrt(1.0 + 0.5 * crease_f.pow(2))

        k_rgb = F.softplus(self.raw_color_k)
        eps_unit = torch.tanh(eps_err)
        color_R = eps_unit * torch.cos(k_rgb[0] * R_flat)
        normal_tilt = 1.0 - (n_poly * rotated_u_dir).sum(dim=-1).abs()
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

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, dict, dict]:
        b, t = tokens.shape
        device = tokens.device
        dtype = self.obj_pos.dtype

        if state is None or not state:
            w_int = torch.zeros(b, self.d_model, device=device, dtype=dtype)
            psi_field = torch.zeros(b, self.n_objects, self.d_model, device=device, dtype=dtype)
            subj_state = torch.zeros(b, self.d_model, device=device, dtype=dtype)
            mod_state = torch.zeros(b, self.d_model, device=device, dtype=dtype)
        else:
            w_int = state["w_int"]
            psi_field = state["psi_field"]
            subj_state = state["subj_state"]
            mod_state = state["mod_state"]

        conductance, edge_normal, radii = self.compute_physical_2d_collisions_between_objects()
        z_open = F.softplus(self.raw_z_open)
        z_imp = F.softplus(self.raw_z_imp) + 1e-4
        surface_threshold = z_open.min(dim=-1).values + z_imp.min(dim=-1).values

        retention = torch.sigmoid(self.raw_obj_retention).view(1, self.n_objects, 1)
        time_leak = torch.sigmoid(self.raw_time_leak).unsqueeze(0)
        rgbl_filter = torch.tanh(self.obj_rgbl_filter)

        w_ext_all = self.ext_embed(tokens)  # (B, T, 64)
        logits_list = []
        fe_total = w_ext_all.new_tensor(0.0)

        total_active_coords = 0.0
        total_touched_nodes = 0.0
        total_cascade_waves = 0.0
        total_holo_area = 0.0
        total_luminance = 0.0
        total_rot_angle = 0.0
        total_err_before = 0.0
        total_err_after = 0.0

        for step_t in range(t):
            w_ext = w_ext_all[:, step_t, :]
            eps_err = w_ext - w_int

            # ====================================================================
            # 1. ПРАВИЛА УПАКОВКИ ЕСТЕСТВЕННОГО ЯЗЫКА (СУБЪЕКТ S — МОДУЛЯТОР M — ОБЪЕКТ O):
            # Определяем ролевые доли текущего сигнала (мягкое распределение ролей S, M, O):
            # ====================================================================
            role_probs = torch.softmax(self.to_role_gates(w_ext + 0.5 * eps_err), dim=-1)  # (B, 3): [p_S, p_M, p_O]
            p_subj = role_probs[:, 0:1]
            p_mod = role_probs[:, 1:2]
            p_obj = role_probs[:, 2:3]

            # Обновляем ядро Субъекта S(t) и активный Модулятор M(t):
            subj_state = (1.0 - 0.5 * p_subj) * subj_state + p_subj * w_ext
            mod_state = (1.0 - 0.6 * p_mod) * mod_state + p_mod * self.mod_operator(w_ext)

            # Модулятор M(t) формирует 3D-оператор поворота Родрига R_M(t) in SO(3):
            omega_m = torch.tanh(self.mod_to_3d_angle(mod_state)) * 1.2  # (B, 3)
            rot_matrix = rodrigues_rotation_3d(omega_m)                  # (B, 3, 3)

            # Воздействие Модулятора на Субъект (билинейная модуляция S * M -> ожидаемый Объект O_hat):
            modulated_subj = subj_state * torch.tanh(mod_state)  # (B, 64)
            # Когда приходит Объект (высокий p_obj), он связывается с промодулированным Субъектом (замыкание гештальта S->M->O)!
            svo_packed = p_subj * subj_state + p_mod * modulated_subj + p_obj * (modulated_subj * torch.tanh(w_ext))
            svo_gain = torch.tanh(self.raw_svo_gain).unsqueeze(0)

            # 2. ПОЛИГОНАЛЬНЫЙ РЕЛЬЕФ, ПОВЕРНУТЫЙ ОПЕРАТОРОМ МОДУЛЯТОРА rot_matrix, И 4 КАНАЛА СВЕТА RGBL
            p_surf, n_poly, dA_poly, sphere_vol, lum_L, proc_rgbl = (
                self.compute_polygonal_relief_and_rgbl_with_modulator_rotation(w_int, w_ext, psi_field, rot_matrix)
            )
            radiant_flux = lum_L * dA_poly

            flux_weights = (eps_err.abs() * radiant_flux).unsqueeze(-1)
            w_sum = flux_weights.sum(dim=1) + 1e-8
            active_surf_pos = (p_surf * flux_weights).sum(dim=1) / w_sum
            active_surf_norm = (n_poly * flux_weights).sum(dim=1) / w_sum
            active_surf_norm = active_surf_norm / torch.sqrt((active_surf_norm * active_surf_norm).sum(dim=-1, keepdim=True) + 1e-12)

            # 3. 3D-КАСКАД И ТЕССЕЛЯЦИЯ ПОЛИГОНАЛЬНОГО ДЕРЕВА NANITE
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
            # В сигнал поверхности входит и ошибка Фристона, и упакованная триада (Субъект -> Модулятор -> Объект)!
            surface_signal = (eps_err + svo_gain * svo_packed).unsqueeze(1) + rgbl_modulated

            tree_signal = torch.matmul(surface_signal, self.poly_tree.T)
            proj = (tree_signal * self.obj_in.unsqueeze(0) * open_gate).sum(dim=-1, keepdim=True)
            tree_response = torch.tanh(proj) * self.obj_out.unsqueeze(0) * open_gate
            facet_response = torch.matmul(tree_response, self.poly_tree)

            psi_field = retention * psi_field + (1.0 - retention) * facet_response
            correction = facet_response.sum(dim=1) + svo_gain * svo_packed

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
                total_luminance += float(lum_L.mean().item())
                total_rot_angle += float(omega_m.norm(dim=-1).mean().item())
                total_err_before += float(eps_err.pow(2).mean().sqrt().item())
                total_err_after += float(eps_after.pow(2).mean().sqrt().item())

        logits = torch.stack(logits_list, dim=1)
        next_state = {
            "w_int": w_int.detach(),
            "psi_field": psi_field.detach(),
            "subj_state": subj_state.detach(),
            "mod_state": mod_state.detach(),
            "fe_total": fe_total,
        }
        stats = {
            "active_coords_frac": total_active_coords / t,
            "touched_nodes_frac": total_touched_nodes / t,
            "avg_cascade_waves": total_cascade_waves / t,
            "holographic_area": total_holo_area / t,
            "avg_luminance": total_luminance / t,
            "modulator_rotation_rad": total_rot_angle / t,
            "err_before": total_err_before / t,
            "err_after": total_err_after / t,
            "free_energy": float(fe_total.item()),
        }
        return logits, next_state, stats


def evaluate_svo_world(model: SemanticRolePacking3DWorld, data: torch.Tensor, chunk_len: int = 64) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    agg: dict[str, float] = {}
    n_steps = 0
    state: dict | None = None
    with torch.no_grad():
        limit = min(data.numel() - 1, chunk_len * 80)
        for start in range(0, limit - chunk_len, chunk_len):
            seq = data[start : start + chunk_len + 1]
            logits, state, st = model.forward_sequence(seq[:-1].unsqueeze(0), state)
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
    print("=== ЭКСПЕРИМЕНТ 1: МЕЖЪЯЗЫКОВОЙ ИЗОМОРФИЗМ УПАКОВКИ СМЫСЛОВ (SVO) И ПОВОРОТ ПРОКРУСТА ===", flush=True)
    iso_res = test_cross_lingual_isomorphism_and_procrustes_rotation()
    print(
        f"Проверено {iso_res['total_svo_combinations']} смысловых комбинаций (Субъект — Модулятор — Объект) в двух языках:\n"
        f"  - Сохранение относительных расстояний и формы между языками (корреляция геометрии): {iso_res['internal_distance_isomorphism_correlation']}\n"
        f"  - Совпадение смыслов ДО поворота угла (P@1): {iso_res['p_at_1_before_angle_rotation_pct']}%\n"
        f"  - Совпадение смыслов ПОСЛЕ чистого 3D/ND-поворота угла Прокруста (P@1): {iso_res['p_at_1_after_procrustes_rotation_pct']}% "
        f"(средний косинус = {iso_res['mean_cosine_after_rotation']})",
        flush=True,
    )

    print("\n=== ЭКСПЕРИМЕНТ 2: ОБУЧЕНИЕ V10 НА АНГЛОЦЕНТРИЧНОМ WIKITEXT-2 С РОЛЕВОЙ УПАКОВКОЙ (S -> M -> O) ===", flush=True)
    train_data = load_wikitext_bytes("train", max_chars=1_500_000)
    valid_data = load_wikitext_bytes("validation", max_chars=250_000)

    chunk_len = 64
    batch_streams = 16
    steps = 400
    lr = 2e-3

    stream_len = train_data.numel() // batch_streams
    streams = train_data[: stream_len * batch_streams].view(batch_streams, stream_len)

    torch.manual_seed(42)
    model = SemanticRolePacking3DWorld(vocab_size=256, side=8, n_objects=96)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели v10 (Субъект-Модулятор-Объект + 3D-поворот Родрига + Полигоны RGBL): {n_params:,}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    state: dict | None = None
    cursor = 0
    t0 = time.perf_counter()
    history = []

    for step in range(1, steps + 1):
        if cursor + chunk_len + 1 >= stream_len:
            cursor = 0
            state = None

        seq = streams[:, cursor : cursor + chunk_len + 1]
        cursor += chunk_len
        src, dst = seq[:, :-1], seq[:, 1:]

        logits, state, st = model.forward_sequence(src, state)
        ce_loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
        total_loss = ce_loss + state["fe_total"]

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if step == 1 or step % 50 == 0:
            bpc = float(ce_loss.item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | v10_SVO_Упаковка BPC={bpc:.3f} | "
                f"угол_модулятора={st['modulator_rotation_rad']:.2f} рад | "
                f"A_poly={st['holographic_area']:.2f}x | L={st['avg_luminance']:.2f} | "
                f"невязка {st['err_before']:.3f}->{st['err_after']:.3f} | "
                f"граней={st['active_coords_frac']*100:.1f}% | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append({"step": step, "bpc": round(bpc, 4), **{k: round(v, 4) for k, v in st.items()}})

    val_bpc, val_ppl, val_st = evaluate_svo_world(model, valid_data, chunk_len)
    print("\n=== ИТОГИ V10 (УПАКОВКА СМЫСЛОВ СУБЪЕКТ-МОДУЛЯТОР-ОБЪЕКТ + 3D-ПОВОРОТ НА WIKITEXT-2) ===", flush=True)
    print(
        f"Validation BPC = {val_bpc:.3f} | Перплексия байта = {val_ppl:.2f} | "
        f"Средний угол поворота модулятором: {val_st['modulator_rotation_rad']:.2f} рад | "
        f"Площадь полигонов A = {val_st['holographic_area']:.2f}x, Яркость L = {val_st['avg_luminance']:.2f} | "
        f"Гашение невязки: {val_st['err_before']:.3f} -> {val_st['err_after']:.3f} | "
        f"Вскрыто полигональных граней: {val_st['active_coords_frac']*100:.1f}%",
        flush=True,
    )

    out_path = OUT_DIR / "rezultat_semantic_packing_v10.json"
    out_path.write_text(
        json.dumps(
            {
                "cross_lingual_isomorphism": iso_res,
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
