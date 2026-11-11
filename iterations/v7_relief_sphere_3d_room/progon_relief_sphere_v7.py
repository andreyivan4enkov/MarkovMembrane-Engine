"""Версия 7 (v7): 3D-комната (X, Y, Z) с динамической рельефной фигурой в центре
(Внутренний мир внутри фигуры <-> 2D-граница с горами/впадинами/шероховатостью <-> Внешняя среда снаружи)
+ Перспектива и дальность вокруг и внутри фигуры.

Физика и геометрия модели v7:
1. 3D-комната (X, Y, Z) и деформируемая фигура в центре:
   - Поверхность центральной фигуры параметризована двумерной сферической сеткой (theta_u, phi_v)
     размера H x W = 8 x 8 = 64 сектора с базовыми единичными направлениями u_dir[u, v] in R^3.
   - В каждый момент t радиус фигуры в направлении (u, v) равен R(u, v, t) > 0,
     а реальные 3D-координаты точки поверхности в комнате (X, Y, Z):
       p_surf(u, v, t) = R(u, v, t) * u_dir[u, v].

2. Внутреннее состояние (внутри шара) vs Внешняя среда (снаружи шара) по Карлу Фристону:
   - Внутренний мир изнутри давит на оболочку своим ожиданием формы: R_int(u, v, t).
   - Внешняя среда (контекст) давит снаружи: P_ext(u, v, t).
   - От столкновения внутреннего и внешнего на 2D-границе возникают неровности, впадины и выпуклости:
       eps(u, v, t) = P_ext(u, v, t) - R_int(u, v, t),
       R(u, v, t) = R_base(u, v) + delta_R(eps(u, v, t)).

3. Эффект "1 км по равнине vs 3 км по горе" (Развертка площади рельефа и наклон нормалей):
   - Локальные градиенты высоты рельефа по широте и долготе (крутизна склонов гор и впадин):
       dR_du = (R(u+1, v) - R(u-1, v)) / 2,
       dR_dv = (R(u, v+1) - R(u, v-1)) / 2.
   - Коэффициент увеличения реальной 2D-площади поверхности за счет гор и шероховатости:
       gamma_relief(u, v, t) = sqrt(1 + dR_du^2 + dR_dv^2 + roughness^2) >= 1.0.
     На ровной поверхности gamma = 1 ("1 километр"), на крутой горе gamma -> 2..4 ("3 километра площади"
     в том же самом секторе!).
   - Кроме того, склон горы меняет 3D-нормаль поверхности n_surf(u, v, t):
       n_surf = normalize(u_dir - dR_du * e_theta - dR_dv * e_phi).
     За счет наклона нормалей на склонах гор соседние и противоположные участки поверхности
     начинают видеть боковые 3D-объекты под новыми углами!

4. Перспектива и дальность в 3D-комнате вокруг и внутри фигуры:
   - В 3D-комнате (X, Y, Z) живут 3D-объекты памяти с координатами r_j = (x_j, y_j, z_j).
   - Когда участок поверхности фигуры p_surf(u, v, t) вытягивается горой навстречу объекту r_j,
     расстояние d_j = ||r_j - p_surf|| сокращается, а площадь горного склона gamma_relief растет ->
     по закону перспективы (gamma_relief / d_j) объект приближается и открывает больше внутренних векторов
     сразу по всем 64 граням!
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

OUT_DIR = Path(__file__).resolve().parent


def build_spherical_room_grid(side: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Строит 3D-направления u_dir (X, Y, Z) и касательные орты (e_theta, e_phi) для 2D-поверхности фигуры в центре комнаты.
    Возвращает тензоры формы (side*side, 3).
    """
    theta = torch.linspace(0.15 * math.pi, 0.85 * math.pi, side)
    phi = torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / side), side)
    th, ph = torch.meshgrid(theta, phi, indexing="ij")

    # Единичный радиус-вектор в 3D-комнате (X, Y, Z)
    ux = torch.sin(th) * torch.cos(ph)
    uy = torch.sin(th) * torch.sin(ph)
    uz = torch.cos(th)
    u_dir = torch.stack([ux, uy, uz], dim=-1).reshape(side * side, 3)

    # Касательный орт вдоль широты theta (для расчета наклона склона горы по u)
    tx = torch.cos(th) * torch.cos(ph)
    ty = torch.cos(th) * torch.sin(ph)
    tz = -torch.sin(th)
    e_theta = torch.stack([tx, ty, tz], dim=-1).reshape(side * side, 3)

    # Касательный орт вдоль долготы phi (для расчета наклона склона горы по v)
    px = -torch.sin(ph)
    py = torch.cos(ph)
    pz = torch.zeros_like(ph)
    e_phi = torch.stack([px, py, pz], dim=-1).reshape(side * side, 3)

    return u_dir, e_theta, e_phi


class ReliefSphereFriston3DWorld(nn.Module):
    """
    Модель v7:
    В центре 3D-комнаты (X, Y, Z) находится деформируемая фигура с 2D-рельефной границей R(u, v, t).
    Позволяет сравнить:
      - enable_relief=False: гладкий шар без гор и шероховатостей (gamma_relief = 1, нормали не наклоняются).
      - enable_relief=True: полная рельефная фигура (горы, впадины, шероховатость, рост площади gamma_relief > 1,
        поворот нормалей на склонах гор и перспективная дальность в 3D-комнате).
    """

    def __init__(self, vocab_size: int = 256, side: int = 8, n_objects: int = 96, enable_relief: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.side = side
        self.d_model = side * side  # 64 сектора поверхности фигуры
        self.n_objects = n_objects
        self.enable_relief = enable_relief

        self.ext_embed = nn.Embedding(vocab_size, self.d_model)
        self.gen_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.gen_head.weight = self.ext_embed.weight

        # Внутренняя динамика состояния внутри шара: переводит усвоенное состояние mu_post(t) -> в ожидание R_int(t+1)
        self.time_transition = nn.Linear(self.d_model, self.d_model)
        self.raw_time_leak = nn.Parameter(torch.zeros(self.d_model))

        # Сетка 3D-направлений и касательных векторов поверхности фигуры в центре комнаты (X, Y, Z)
        u_dir, e_theta, e_phi = build_spherical_room_grid(side)
        self.register_buffer("u_dir", u_dir)      # (64, 3)
        self.register_buffer("e_theta", e_theta)  # (64, 3)
        self.register_buffer("e_phi", e_phi)      # (64, 3)

        # Базовый радиус покоя фигуры в каждом из 64 секторов и упругость коры (шероховатости)
        self.raw_base_radius = nn.Parameter(torch.zeros(self.d_model))
        self.raw_roughness_gain = nn.Parameter(torch.zeros(self.d_model))

        # 3D-объекты в комнате (X, Y, Z) — как внутри фигуры, так и вокруг нее:
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.3)
        self.obj_raw_radius = nn.Parameter(torch.zeros(n_objects))

        # Полные векторные грани каждого 3D-объекта (все 64 грани доступны при раскрытии перспективы!)
        self.obj_in = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, self.d_model) / math.sqrt(self.d_model))

        # Собственные локальные пороги вскрытия z_open[j, d] и локальные импедансы z_imp[j, d] каждой грани
        init_hierarchy = torch.linspace(-1.5, 2.5, self.d_model).unsqueeze(0).expand(n_objects, self.d_model).clone()
        self.raw_z_open = nn.Parameter(init_hierarchy)
        self.raw_z_imp = nn.Parameter(torch.zeros(n_objects, self.d_model))

        # Память 3D-среды Psi(t)
        self.raw_obj_retention = nn.Parameter(torch.zeros(n_objects))

    def compute_surface_relief_and_normals(
        self, r_int: torch.Tensor, eps_err: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисляет 3D-форму деформированной фигуры в центре комнаты:
        - p_surf: (B, 64, 3) — реальные 3D-координаты (X, Y, Z) каждого участка поверхности фигуры
        - n_surf: (B, 64, 3) — локальные 3D-нормали с учетом наклона склонов гор и впадин
        - gamma_area: (B, 64) — коэффициент развертки площади ("1 км по равнине vs 3 км по горе")
        """
        b = r_int.shape[0]
        base_r = F.softplus(self.raw_base_radius).unsqueeze(0) + 0.5  # (1, 64)

        if not self.enable_relief:
            # Гладкий шар без рельефа: радиус постоянный, нормали строго по u_dir, площадь = 1.0
            p_surf = base_r.unsqueeze(-1) * self.u_dir.unsqueeze(0).expand(b, -1, -1)
            n_surf = self.u_dir.unsqueeze(0).expand(b, -1, -1)
            gamma_area = torch.ones(b, self.d_model, device=r_int.device, dtype=r_int.dtype)
            return p_surf, n_surf, gamma_area

        # Высота рельефа R(u, v, t) формируется столкновением внутреннего состояния r_int и внешней невязки eps_err:
        # выпуклости (горы) там, где внешний сигнал прогибает мембрану наружу, впадины — где внутрь
        height_field = base_r + 0.45 * torch.tanh(r_int) + 0.55 * eps_err  # (B, 64)
        R_grid = F.softplus(height_field).view(b, self.side, self.side)     # (B, 8, 8) > 0

        # Считаем крутизну склонов гор по широте (u) и долготе (v) на замкнутой сфере:
        # По широте u — конечная разность с соседними поясами, по долготе v — циклическая разность вокруг оси шара (torch.roll)
        dR_dv = 0.5 * (torch.roll(R_grid, shifts=-1, dims=2) - torch.roll(R_grid, shifts=1, dims=2))
        dR_du = torch.zeros_like(R_grid)
        dR_du[:, 1:-1, :] = 0.5 * (R_grid[:, 2:, :] - R_grid[:, :-2, :])
        dR_du[:, 0, :] = R_grid[:, 1, :] - R_grid[:, 0, :]
        dR_du[:, -1, :] = R_grid[:, -1, :] - R_grid[:, -2, :]

        # Лапласиан рельефа = высокочастотная шероховатость (мелкие зубцы и трещины на склонах горы)
        laplacian = (
            torch.roll(R_grid, shifts=-1, dims=2)
            + torch.roll(R_grid, shifts=1, dims=2)
            - 2.0 * R_grid
        ).reshape(b, self.d_model)
        roughness_gain = F.softplus(self.raw_roughness_gain).unsqueeze(0)
        roughness = roughness_gain * laplacian.abs()

        dR_du_flat = dR_du.reshape(b, self.d_model)
        dR_dv_flat = dR_dv.reshape(b, self.d_model)
        R_flat = R_grid.reshape(b, self.d_model)

        # Коэффициент увеличения площади поверхности за счет гор и шероховатости:
        # gamma_area = sqrt(1 + (dR/du)^2 + (dR/dv)^2 + roughness^2) >= 1.0 ("эффект 3 км по горе вместо 1 км по равнине"!)
        gamma_area = torch.sqrt(1.0 + dR_du_flat.pow(2) + dR_dv_flat.pow(2) + roughness.pow(2))  # (B, 64)

        # Реальные 3D-координаты (X, Y, Z) деформированной поверхности в комнате:
        p_surf = R_flat.unsqueeze(-1) * self.u_dir.unsqueeze(0)  # (B, 64, 3)

        # Наклон 3D-нормали на склоне горы:
        # склон поворачивает вектор нормали от радиального луча u_dir в сторону градиента высоты!
        raw_normal = (
            self.u_dir.unsqueeze(0)
            - dR_du_flat.unsqueeze(-1) * self.e_theta.unsqueeze(0)
            - dR_dv_flat.unsqueeze(-1) * self.e_phi.unsqueeze(0)
        )
        n_surf = raw_normal / torch.sqrt((raw_normal * raw_normal).sum(dim=-1, keepdim=True) + 1e-12)
        return p_surf, n_surf, gamma_area

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

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        r_int_state: torch.Tensor | None = None,
        psi_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        b, t = tokens.shape
        device = tokens.device
        dtype = self.obj_pos.dtype

        if r_int_state is None:
            r_int_state = torch.zeros(b, self.d_model, device=device, dtype=dtype)
        if psi_field is None:
            psi_field = torch.zeros(b, self.n_objects, self.d_model, device=device, dtype=dtype)

        conductance, edge_normal, radii = self.compute_physical_2d_collisions_between_objects()
        z_open = F.softplus(self.raw_z_open)  # (N, 64)
        z_imp = F.softplus(self.raw_z_imp) + 1e-4
        surface_threshold = z_open.min(dim=-1).values + z_imp.min(dim=-1).values  # (N,)

        retention = torch.sigmoid(self.raw_obj_retention).view(1, self.n_objects, 1)
        time_leak = torch.sigmoid(self.raw_time_leak).unsqueeze(0)

        p_ext_all = self.ext_embed(tokens)  # (B, T, 64) — внешнее поле вокруг фигуры
        logits_list = []
        fe_total = p_ext_all.new_tensor(0.0)

        total_active_coords = 0.0
        total_touched_nodes = 0.0
        total_cascade_waves = 0.0
        total_area_factor = 0.0
        total_err_before = 0.0
        total_err_after = 0.0

        r_int = r_int_state
        for step_t in range(t):
            p_ext = p_ext_all[:, step_t, :]  # (B, 64) — давление внешней среды снаружи фигуры

            # 1. КОЛЛИЗИЯ ВНУТРЕННЕГО И ВНЕШНЕГО НА 2D-ОБОЛОЧКЕ ФИГУРЫ
            eps_err = p_ext - r_int  # (B, 64) — деформирующая сила невязки

            # 2. ВЫЧИСЛЕНИЕ РЕЛЬЕФА ФИГУРЫ (ГОРЫ, ВПАДИНЫ, ШЕРОХОВАТОСТЬ) И ПЛОЩАДИ gamma_area >= 1
            p_surf, n_surf, gamma_area = self.compute_surface_relief_and_normals(r_int, eps_err)
            # p_surf: (B, 64, 3), n_surf: (B, 64, 3), gamma_area: (B, 64)

            # Эффективная емкость сигнала на рельефной границе:
            # там, где выросла гора/шероховатость (gamma_area > 1), площадь контакта увеличивается пропорционально gamma_area!
            relief_amplified_err = eps_err * gamma_area  # (B, 64)

            # 3. ПЕРСПЕКТИВА И ДАЛЬНОСТЬ ОТ РЕЛЬЕФНОЙ ФИГУРЫ К 3D-ОБЪЕКТАМ В КОМНАТЕ (X, Y, Z)
            # Центр тяжести деформации фигуры и главный вектор нормали горного хребта:
            err_weights = eps_err.abs().unsqueeze(-1)  # (B, 64, 1)
            weight_sum = err_weights.sum(dim=1) + 1e-8  # (B, 1)
            active_surf_pos = (p_surf * err_weights).sum(dim=1) / weight_sum  # (B, 3) — точка главного хребта в (X,Y,Z)
            active_surf_norm = (n_surf * err_weights).sum(dim=1) / weight_sum
            active_surf_norm = active_surf_norm / torch.sqrt((active_surf_norm * active_surf_norm).sum(dim=-1, keepdim=True) + 1e-12)

            # Дальность от активного хребта поверхности фигуры до каждого 3D-объекта в комнате:
            delta = self.obj_pos.unsqueeze(0) - active_surf_pos.unsqueeze(1)  # (B, N, 3)
            dist = torch.sqrt((delta * delta).sum(dim=-1) + 1e-12)  # (B, N)
            dir_to_obj = delta / dist.unsqueeze(-1)

            # Перспективная видимость вдоль наклоненной нормали горного склона:
            cos_view = F.relu((active_surf_norm.unsqueeze(1) * dir_to_obj).sum(dim=-1))  # (B, N)
            r_view = radii.unsqueeze(0)

            # Полная энергия удара с учетом прироста площади горного рельефа gamma_area:
            relief_energy_norm = torch.sqrt((relief_amplified_err * relief_amplified_err).sum(dim=-1, keepdim=True) + 1e-12)
            initial_energy = relief_energy_norm * cos_view * torch.exp(-dist / r_view)  # (B, N)

            # Каскад в 3D-комнате по соприкасающимся объектам (с законом сохранения энергии)
            node_energy, n_waves = self.propagate_cascade_with_energy_conservation(
                initial_energy, active_surf_norm, conductance, edge_normal, surface_threshold
            )  # (B, N)

            # 4. ПЕРСПЕКТИВНОЕ ВСКРЫТИЕ ГРАНЕЙ С УЧЕТОМ РЕЛЬЕФА ПОВЕРХНОСТИ gamma_area(u, v)
            # Заметьте: каждая грань d in 0..63 соответствует сектору (u, v) рельефной поверхности!
            # Если в секторе d выросла гора (gamma_area[:, d] > 1), она локально приближает поверхность к объектам
            # и пропорционально увеличивает поток энергии именно через эту грань d!
            sector_energy = node_energy.unsqueeze(-1) * gamma_area.unsqueeze(1)  # (B, N, 64)
            excess = F.relu(sector_energy - z_open.unsqueeze(0))  # (B, N, 64)
            open_gate = excess / (excess + z_imp.unsqueeze(0))    # (B, N, 64), строгий 0.0 при sector_energy <= z_open!

            # 5. АДАПТАЦИЯ ВНУТРЕННЕГО СОСТОЯНИЯ ШАРА ПОД ВНЕШНЮЮ СРЕДУ (КАРЛ ФРИСТОН)
            proj = (relief_amplified_err.unsqueeze(1) * self.obj_in.unsqueeze(0) * open_gate).sum(dim=-1, keepdim=True)
            facet_response = torch.tanh(proj) * self.obj_out.unsqueeze(0) * open_gate  # (B, N, 64)

            psi_field = retention * psi_field + (1.0 - retention) * facet_response
            correction = facet_response.sum(dim=1)  # (B, 64)

            # Остаточная деформация на поверхности фигуры после адаптации внутреннего состояния:
            eps_after = eps_err - correction
            mu_post = r_int + correction

            step_fe = 0.5 * (eps_after * eps_after).mean()
            fe_total = fe_total + step_fe / t

            # Переход внутреннего состояния шара к предсказанию формы следующего кадра внешней среды R_int(t+1):
            env_memory = psi_field.sum(dim=1)
            r_int_next = (1.0 - time_leak) * mu_post + time_leak * self.time_transition(mu_post + env_memory)

            logits_list.append(self.gen_head(r_int_next))
            r_int = r_int_next

            with torch.no_grad():
                active_mask = (open_gate > 0.0).float()
                total_active_coords += float(active_mask.mean().item())
                total_touched_nodes += float((active_mask.sum(dim=-1) > 0).float().mean().item())
                total_cascade_waves += float(n_waves)
                total_area_factor += float(gamma_area.mean().item())
                total_err_before += float(eps_err.pow(2).mean().sqrt().item())
                total_err_after += float(eps_after.pow(2).mean().sqrt().item())

        logits = torch.stack(logits_list, dim=1)
        stats = {
            "active_coords_frac": total_active_coords / t,
            "touched_nodes_frac": total_touched_nodes / t,
            "avg_cascade_waves": total_cascade_waves / t,
            "relief_area_multiplier": total_area_factor / t,
            "err_before": total_err_before / t,
            "err_after": total_err_after / t,
            "free_energy": float(fe_total.item()),
        }
        return logits, r_int.detach(), psi_field.detach(), fe_total, stats


def evaluate_relief_world(model: ReliefSphereFriston3DWorld, data: torch.Tensor, chunk_len: int = 64) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    agg: dict[str, float] = {}
    n_steps = 0
    r_int = None
    psi_field = None
    with torch.no_grad():
        limit = min(data.numel() - 1, chunk_len * 80)
        for start in range(0, limit - chunk_len, chunk_len):
            seq = data[start : start + chunk_len + 1]
            logits, r_int, psi_field, _, st = model.forward_sequence(seq[:-1].unsqueeze(0), r_int, psi_field)
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
    smooth_sphere = ReliefSphereFriston3DWorld(vocab_size=256, side=8, n_objects=96, enable_relief=False)
    torch.manual_seed(42)
    relief_figure = ReliefSphereFriston3DWorld(vocab_size=256, side=8, n_objects=96, enable_relief=True)

    n_params = sum(p.numel() for p in relief_figure.parameters())
    print("=== V7: 3D-КОМНАТА (X,Y,Z) С ДИНАМИЧЕСКОЙ РЕЛЬЕФНОЙ ФИГУРОЙ В ЦЕНТРЕ (ГОРЫ/ВПАДИНЫ/ПЕРСПЕКТИВА) ===", flush=True)
    print(f"Параметров: {n_params:,} | Сравнение: Гладкий шар (1.0x площадь) vs Рельефная фигура (горы/шероховатость)", flush=True)

    opt_smooth = torch.optim.AdamW(smooth_sphere.parameters(), lr=lr)
    opt_relief = torch.optim.AdamW(relief_figure.parameters(), lr=lr)

    r_sm, psi_sm = None, None
    r_rl, psi_rl = None, None
    cursor = 0
    t0 = time.perf_counter()
    history = []

    for step in range(1, steps + 1):
        if cursor + chunk_len + 1 >= stream_len:
            cursor = 0
            r_sm, psi_sm = None, None
            r_rl, psi_rl = None, None

        seq = streams[:, cursor : cursor + chunk_len + 1]
        cursor += chunk_len
        src, dst = seq[:, :-1], seq[:, 1:]

        log_s, r_sm, psi_sm, fe_s, st_s = smooth_sphere.forward_sequence(src, r_sm, psi_sm)
        loss_s = F.cross_entropy(log_s.reshape(-1, 256), dst.reshape(-1)) + fe_s
        opt_smooth.zero_grad()
        loss_s.backward()
        opt_smooth.step()

        log_r, r_rl, psi_rl, fe_r, st_r = relief_figure.forward_sequence(src, r_rl, psi_rl)
        loss_r = F.cross_entropy(log_r.reshape(-1, 256), dst.reshape(-1)) + fe_r
        opt_relief.zero_grad()
        loss_r.backward()
        opt_relief.step()

        if step == 1 or step % 50 == 0:
            bpc_s = float((loss_s - fe_s).item()) / math.log(2.0)
            bpc_r = float((loss_r - fe_r).item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | гладкий_шар BPC={bpc_s:.3f} (граней={st_s['active_coords_frac']*100:.1f}%) | "
                f"рельефная_фигура BPC={bpc_r:.3f} (площадь={st_r['relief_area_multiplier']:.2f}x, "
                f"невязка {st_r['err_before']:.3f}->{st_r['err_after']:.3f}, "
                f"волн={st_r['avg_cascade_waves']:.1f}, граней={st_r['active_coords_frac']*100:.1f}%) | "
                f"{time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append(
                {
                    "step": step,
                    "smooth_bpc": round(bpc_s, 4),
                    "relief_bpc": round(bpc_r, 4),
                    "smooth_stats": {k: round(v, 4) for k, v in st_s.items()},
                    "relief_stats": {k: round(v, 4) for k, v in st_r.items()},
                }
            )

    sm_val_bpc, sm_val_ppl, sm_val_st = evaluate_relief_world(smooth_sphere, valid_data, chunk_len)
    rl_val_bpc, rl_val_ppl, rl_val_st = evaluate_relief_world(relief_figure, valid_data, chunk_len)

    print("\n=== ИТОГИ V7 НА ОТЛОЖЕННЫХ СТАТЬЯХ WIKITEXT-2 (ГЛАДКИЙ ШАР vs РЕЛЬЕФНАЯ ФИГУРА С ГОРАМИ) ===", flush=True)
    print(
        f"Гладкий шар (площадь 1.00x):       Validation BPC = {sm_val_bpc:.3f} | Перплексия = {sm_val_ppl:.2f} | "
        f"гашение невязки: {sm_val_st['err_before']:.3f} -> {sm_val_st['err_after']:.3f} | "
        f"вскрыто граней: {sm_val_st['active_coords_frac']*100:.1f}%",
        flush=True,
    )
    print(
        f"Рельефная фигура (площадь {rl_val_st['relief_area_multiplier']:.2f}x): Validation BPC = {rl_val_bpc:.3f} | Перплексия = {rl_val_ppl:.2f} | "
        f"гашение невязки: {rl_val_st['err_before']:.3f} -> {rl_val_st['err_after']:.3f} | "
        f"вскрыто граней: {rl_val_st['active_coords_frac']*100:.1f}% (узлов: {rl_val_st['touched_nodes_frac']*100:.1f}%)",
        flush=True,
    )

    out_path = OUT_DIR / "rezultat_relief_sphere_v7.json"
    out_path.write_text(
        json.dumps(
            {
                "params": n_params,
                "history": history,
                "validation": {
                    "smooth_sphere": {"bpc": round(sm_val_bpc, 4), "ppl": round(sm_val_ppl, 2), "stats": sm_val_st},
                    "relief_figure": {"bpc": round(rl_val_bpc, 4), "ppl": round(rl_val_ppl, 2), "stats": rl_val_st},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Сохранено в {out_path}", flush=True)


if __name__ == "__main__":
    main()
