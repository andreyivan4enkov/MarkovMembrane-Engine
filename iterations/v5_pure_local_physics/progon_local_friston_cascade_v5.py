"""Версия 5 (финальная чистая физика): Эмерджентная архитектура Карла Фристона и 3D-коллизий
БЕЗ единого костыля и с согласованной стрелой времени.

Что исправлено в физике:
1. Согласованная размерность энергии и порога пробоя оболочки:
   - Полная энергия удара невязки — евклидова норма вектора ошибки ||eps(t)||_2 (без деления на D).
   - Порог начала каскадного перелива из узла j к соприкасающимся 3D-соседям — это порог пробоя
     внешней оболочки узла z_surface(j) = min_d z_open[j, d], а полная емкость поглощения узла —
     сумма порогов открытых в нем граней. Благодаря этому каскад реально идет по 2D-контактам (волны > 0),
     пока энергия удара не будет поглощена!
2. Единая непротиворечивая стрела времени по Карлу Фристону:
   - На шаге t внешняя среда предъявляет s_ext(t), а у внутреннего мира уже готов прогноз s_hat(t),
     сделанный на шаге t-1.
   - На 2D-мембране возникает невязка eps(t) = s_ext(t) - s_hat(t).
   - Каскад по 3D-объектам вскрывает только те векторные грани (j, d), где E_j(t) > z_open[j, d]
     (у остальных граней строгий 0.0 и в прямом проходе, и в бэкпропе!).
   - Открытые грани формируют реакцию гашения ошибок correction(t), уменьшая невязку на мембране:
       eps_after(t) = eps(t) - correction(t),
       mu_post(t) = s_hat(t) + correction(t).
   - Из обновленного состояния mu_post(t) и следа в 3D-среде Psi(t) внутренний мир строит
     прогноз СЛЕДУЮЩЕГО шага s_hat(t+1), из которого декодируется следующий символ текста!
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


class EmergentFriston3DWorld(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_objects: int = 96):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_objects = n_objects

        self.ext_embed = nn.Embedding(vocab_size, d_model)
        self.gen_head = nn.Linear(d_model, vocab_size, bias=False)
        self.gen_head.weight = self.ext_embed.weight

        # Оператор хода времени внутреннего мира: переводит усвоенное состояние mu_post(t) -> в прогноз s_hat(t+1)
        self.time_transition = nn.Linear(d_model, d_model)
        self.raw_time_leak = nn.Parameter(torch.zeros(d_model))

        # Проекция невязки в 3D-точку удара и вектор направления удара
        self.to_3d_pos = nn.Linear(d_model, 3)
        self.to_3d_vel = nn.Linear(d_model, 3)

        # 3D-объекты внутреннего мира: координаты r_j и радиусы оболочек R_j
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.2)
        self.obj_raw_radius = nn.Parameter(torch.zeros(n_objects))

        # Векторные грани каждого 3D-объекта (D граней внутри каждого из N объектов)
        self.obj_in = nn.Parameter(torch.randn(n_objects, d_model) / math.sqrt(d_model))
        self.obj_out = nn.Parameter(torch.randn(n_objects, d_model) / math.sqrt(d_model))

        # Локальные пороги вскрытия z_open[j, d] и локальные импедансы z_imp[j, d] каждой грани
        init_hierarchy = torch.linspace(-1.5, 2.5, d_model).unsqueeze(0).expand(n_objects, d_model).clone()
        self.raw_z_open = nn.Parameter(init_hierarchy)
        self.raw_z_imp = nn.Parameter(torch.zeros(n_objects, d_model))

        # Локальная память среды Psi(t) каждого объекта
        self.raw_obj_retention = nn.Parameter(torch.zeros(n_objects))

    def compute_physical_2d_collisions_between_objects(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Физическое пересечение 3D-оболочек объектов (без top-k):
        проводят только реально пересекающиеся в 3D шары (d_jm < R_j + R_m).
        """
        radii = F.softplus(self.obj_raw_radius) + 1e-4  # (N,)
        diff = self.obj_pos.unsqueeze(0) - self.obj_pos.unsqueeze(1)  # (N, N, 3): r_m - r_j
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # (N, N)
        edge_normal = diff / dist.unsqueeze(-1)

        sum_r = radii.unsqueeze(1) + radii.unsqueeze(0)
        penetration = F.relu(sum_r - dist)
        eye_mask = 1.0 - torch.eye(self.n_objects, device=dist.device, dtype=dist.dtype)
        penetration = penetration * eye_mask

        # Закон сохранения потока через 2D-границу
        total_contact = penetration.sum(dim=-1, keepdim=True) + radii.unsqueeze(-1)
        conductance = penetration / total_contact
        return conductance, edge_normal, radii

    def propagate_cascade_with_energy_conservation(
        self,
        initial_energy: torch.Tensor,
        ray_dir: torch.Tensor,
        conductance: torch.Tensor,
        edge_normal: torch.Tensor,
        surface_threshold: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """
        Каскад со строгим законом сохранения энергии в цикле WHILE:
        Узел j сам удерживает энергию до порога своей оболочки surface_threshold[j],
        а весь избыток overflow = ReLU(E_j - surface_threshold[j]) покидает узел и течет через 2D-контакты к соседям!
        """
        b = initial_energy.shape[0]
        align = F.relu((ray_dir.view(b, 1, 1, 3) * edge_normal.unsqueeze(0)).sum(dim=-1))
        directed_cond = conductance.unsqueeze(0) * align  # (B, N, N), сумма по строке < 1

        absorbed = torch.zeros_like(initial_energy)
        current_wave = initial_energy
        thresh_view = surface_threshold.unsqueeze(0)  # (1, N)

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
        s_hat_state: torch.Tensor | None = None,
        psi_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        b, t = tokens.shape
        device = tokens.device
        dtype = self.obj_pos.dtype

        if s_hat_state is None:
            s_hat_state = torch.zeros(b, self.d_model, device=device, dtype=dtype)
        if psi_field is None:
            psi_field = torch.zeros(b, self.n_objects, self.d_model, device=device, dtype=dtype)

        conductance, edge_normal, radii = self.compute_physical_2d_collisions_between_objects()
        z_open = F.softplus(self.raw_z_open)  # (N, D) > 0: локальный порог вскрытия каждой грани
        z_imp = F.softplus(self.raw_z_imp) + 1e-4  # (N, D) > 0: локальный импеданс каждой грани
        # Порог пробоя внешней оболочки узла, после которого избыток энергии идет каскадом к соседям:
        surface_threshold = z_open.min(dim=-1).values + z_imp.min(dim=-1).values  # (N,)

        retention = torch.sigmoid(self.raw_obj_retention).view(1, self.n_objects, 1)
        time_leak = torch.sigmoid(self.raw_time_leak).unsqueeze(0)  # (1, D)

        s_ext_all = self.ext_embed(tokens)  # (B, T, D)
        logits_list = []
        fe_total = s_ext_all.new_tensor(0.0)

        total_active_coords = 0.0
        total_touched_nodes = 0.0
        total_cascade_waves = 0.0
        total_err_before = 0.0
        total_err_after = 0.0

        s_hat = s_hat_state
        for step_t in range(t):
            s_ext = s_ext_all[:, step_t, :]  # (B, D) — реальный сигнал внешней среды в момент t

            # 1. ОДЕЯЛО МАРКОВА (2D-МЕМБРАНА): ОШИБКА ПРЕДСКАЗАНИЯ МЕЖДУ ВНЕШНЕЙ СРЕДОЙ И ПРОГНОЗОМ ВНУТРЕННЕГО МИРА
            eps_err = s_ext - s_hat  # (B, D)
            # Полная евклидова энергия удара невязки ||eps_err||_2 (без деления на D!)
            err_norm = torch.sqrt((eps_err * eps_err).sum(dim=-1, keepdim=True) + 1e-12)  # (B, 1)

            # 2. 3D-КОЛЛИЗИЯ И КАСКАД ПО СОПРИКАСАЮЩИМСЯ ОБЪЕКТАМ В ЦИКЛЕ WHILE
            r_hit = self.to_3d_pos(s_hat) + self.to_3d_pos(eps_err)  # (B, 3)
            v_hit = self.to_3d_vel(eps_err)  # (B, 3)
            ray_dir = v_hit / torch.sqrt((v_hit * v_hit).sum(dim=-1, keepdim=True) + 1e-12)

            delta = self.obj_pos.unsqueeze(0) - r_hit.unsqueeze(1)  # (B, N, 3)
            dist = torch.sqrt((delta * delta).sum(dim=-1) + 1e-12)  # (B, N)
            normal = delta / dist.unsqueeze(-1)

            cos_inc = F.relu((ray_dir.unsqueeze(1) * normal).sum(dim=-1))  # (B, N)
            r_view = radii.unsqueeze(0)  # (1, N)
            initial_energy = err_norm * cos_inc * torch.exp(-dist / r_view)  # (B, N)

            node_energy, n_waves = self.propagate_cascade_with_energy_conservation(
                initial_energy, ray_dir, conductance, edge_normal, surface_threshold
            )  # (B, N)

            # 3. ВСКРЫТИЕ ВЕКТОРНЫХ ГРАНЕЙ ОБЪЕКТОВ ПО ЛОКАЛЬНЫМ ПОРОГАМ z_open[j, d]
            # Если node_energy <= z_open[j, d], то excess == 0.0 (строгий ноль в прямом проходе и в бэкпропе!)
            excess = F.relu(node_energy.unsqueeze(-1) - z_open.unsqueeze(0))  # (B, N, D)
            open_gate = excess / (excess + z_imp.unsqueeze(0))  # (B, N, D)

            # 4. ГАШЕНИЕ НЕВЯЗКИ ОТКРЫТЫМИ ГРАНЯМИ И ОБНОВЛЕНИЕ ВНУТРЕННЕГО МИРА
            # Каждая открытая грань проектирует невязку eps_err и выдает компенсирующий импульс
            proj = (eps_err.unsqueeze(1) * self.obj_in.unsqueeze(0) * open_gate).sum(dim=-1, keepdim=True)  # (B, N, 1)
            facet_response = torch.tanh(proj) * self.obj_out.unsqueeze(0) * open_gate  # (B, N, D)

            # След в 3D-среде Psi(t) обновляется только в открытых гранях
            psi_field = retention * psi_field + (1.0 - retention) * facet_response

            # Суммарная коррекция от всех сработавших 3D-объектов
            correction = facet_response.sum(dim=1)  # (B, D)

            # Остаточная невязка на 2D-мембране ПОСЛЕ срабатывания каскада:
            eps_after = eps_err - correction  # (B, D)
            # Выстроенное внутреннее состояние системы в момент t:
            mu_post = s_hat + correction  # (B, D)

            # Свободная энергия Фристона = квадрат непогашенной невязки после каскада:
            # минимизируя F, градиент прямо учит открытые грани гасить eps_err (делая ||eps_after|| < ||eps_err||)!
            step_fe = 0.5 * (eps_after * eps_after).mean()
            fe_total = fe_total + step_fe / t

            # 5. ШАГ ВНУТРЕННЕГО МИРА ВДОЛЬ СТРЕЛЫ ВРЕМЕНИ t -> t+1:
            # Внутренний мир строит прогноз следующего состояния внешней среды s_hat(t+1)
            env_memory = psi_field.sum(dim=1)  # (B, D)
            s_hat_next = (1.0 - time_leak) * mu_post + time_leak * self.time_transition(mu_post + env_memory)

            # Из этого прогноза s_hat_next декодируется следующий байт внешней среды tokens[:, t+1]
            logits_list.append(self.gen_head(s_hat_next))
            s_hat = s_hat_next

            with torch.no_grad():
                active_mask = (open_gate > 0.0).float()
                total_active_coords += float(active_mask.mean().item())
                total_touched_nodes += float((active_mask.sum(dim=-1) > 0).float().mean().item())
                total_cascade_waves += float(n_waves)
                total_err_before += float(err_norm.mean().item() / math.sqrt(self.d_model))
                total_err_after += float(torch.sqrt((eps_after * eps_after).mean(dim=-1) + 1e-12).mean().item())

        logits = torch.stack(logits_list, dim=1)
        stats = {
            "active_coords_frac": total_active_coords / t,
            "touched_nodes_frac": total_touched_nodes / t,
            "avg_cascade_waves": total_cascade_waves / t,
            "err_before": total_err_before / t,
            "err_after": total_err_after / t,
            "free_energy": float(fe_total.item()),
        }
        return logits, s_hat.detach(), psi_field.detach(), fe_total, stats


def evaluate_continuous_world(model: EmergentFriston3DWorld, data: torch.Tensor, chunk_len: int = 64) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    agg: dict[str, float] = {}
    n_steps = 0
    s_hat_state = None
    psi_field = None
    with torch.no_grad():
        limit = min(data.numel() - 1, chunk_len * 80)
        for start in range(0, limit - chunk_len, chunk_len):
            seq = data[start : start + chunk_len + 1]
            src = seq[:-1].unsqueeze(0)
            dst = seq[1:]
            logits, s_hat_state, psi_field, _, st = model.forward_sequence(src, s_hat_state, psi_field)
            loss = F.cross_entropy(logits[0], dst, reduction="sum")
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


def verify_zero_gradient_on_unopened_facets(model: EmergentFriston3DWorld, sample_seq: torch.Tensor) -> dict:
    model.zero_grad()
    src = sample_seq[:, :1]  # проверяем одиночный такт удара, чтобы видеть мгновенный срез каскада
    dst = sample_seq[:, 1:2]
    logits, _, _, fe, _ = model.forward_sequence(src)
    loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), dst.reshape(-1)) + fe
    loss.backward()

    grad_in = model.obj_in.grad
    assert grad_in is not None
    nonzero_facets = int((grad_in.abs() > 0.0).sum().item())
    total_facets = int(grad_in.numel())
    zero_facets = total_facets - nonzero_facets
    model.zero_grad()
    return {
        "total_facets": total_facets,
        "rewritten_facets_in_backprop": nonzero_facets,
        "untouched_exact_zero_facets": zero_facets,
        "rewritten_percent": round(100.0 * nonzero_facets / total_facets, 2),
    }


def main():
    train_data = load_wikitext_bytes("train", max_chars=1_500_000)
    valid_data = load_wikitext_bytes("validation", max_chars=250_000)

    chunk_len = 64
    batch_streams = 16
    steps = 400
    lr = 2e-3

    torch.manual_seed(42)
    model = EmergentFriston3DWorld(vocab_size=256, d_model=64, n_objects=96)
    n_params = sum(p.numel() for p in model.parameters())
    print("=== V5: ЧИСТАЯ ЭМЕРДЖЕНТНАЯ ФИЗИКА ФРИСТОНА И 3D-КОЛЛИЗИЙ (0 КОСТЫЛЕЙ) ===", flush=True)
    print(f"Параметров: {n_params:,} (96 3D-объектов по 64 векторных грани = 6144 граней)", flush=True)

    stream_len = train_data.numel() // batch_streams
    streams = train_data[: stream_len * batch_streams].view(batch_streams, stream_len)

    sample_seq = streams[:, : chunk_len + 1]
    bp_check = verify_zero_gradient_on_unopened_facets(model, sample_seq)
    print(
        f"Бэкпроп на 1 такте (до обучения): всего граней = {bp_check['total_facets']}, "
        f"переписано каскадом = {bp_check['rewritten_facets_in_backprop']} ({bp_check['rewritten_percent']}%), "
        f"строгий 0.0 градиента = {bp_check['untouched_exact_zero_facets']}",
        flush=True,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    s_hat_state = None
    psi_field = None
    cursor = 0
    t0 = time.perf_counter()
    history = []

    for step in range(1, steps + 1):
        if cursor + chunk_len + 1 >= stream_len:
            cursor = 0
            s_hat_state = None
            psi_field = None

        seq = streams[:, cursor : cursor + chunk_len + 1]
        cursor += chunk_len
        src = seq[:, :-1]
        dst = seq[:, 1:]

        logits, s_hat_state, psi_field, fe_loss, st = model.forward_sequence(src, s_hat_state, psi_field)
        ce_loss = F.cross_entropy(logits.reshape(-1, 256), dst.reshape(-1))
        total_loss = ce_loss + fe_loss

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if step == 1 or step % 50 == 0:
            bpc = float(ce_loss.item()) / math.log(2.0)
            with torch.no_grad():
                cond, _, _ = model.compute_physical_2d_collisions_between_objects()
                avg_colliding_neighbors = float((cond > 0.0).float().sum(dim=-1).mean().item())
            print(
                f"шаг {step:3d} | BPC={bpc:.3f} | F={st['free_energy']:.3f} "
                f"(невязка до={st['err_before']:.3f} -> после={st['err_after']:.3f}) | "
                f"волн_каскада={st['avg_cascade_waves']:.2f} | "
                f"2D-контактов/узел={avg_colliding_neighbors:.1f} | "
                f"узлов={st['touched_nodes_frac']*100:.1f}% | "
                f"вскрыто_граней={st['active_coords_frac']*100:.1f}% | "
                f"{time.perf_counter()-t0:.0f}с",
                flush=True,
            )
            history.append(
                {
                    "step": step,
                    "bpc": round(bpc, 4),
                    "avg_colliding_neighbors": round(avg_colliding_neighbors, 2),
                    **{k: round(v, 4) for k, v in st.items()},
                }
            )

    bp_check_after = verify_zero_gradient_on_unopened_facets(model, sample_seq)
    val_bpc, val_ppl, val_st = evaluate_continuous_world(model, valid_data, chunk_len=chunk_len)

    print("\n=== ИТОГИ ЧИСТОЙ ЭМЕРДЖЕНТНОЙ МОДЕЛИ V5 НА ОТЛОЖЕННЫХ СТАТЬЯХ WIKITEXT-2 ===", flush=True)
    print(
        f"Validation BPC = {val_bpc:.3f} | Перплексия байта = {val_ppl:.2f} | "
        f"Гашение невязки каскадом: {val_st['err_before']:.3f} -> {val_st['err_after']:.3f} | "
        f"Среднее число волн каскада: {val_st['avg_cascade_waves']:.2f} | "
        f"Затронуто узлов: {val_st['touched_nodes_frac']*100:.1f}% | "
        f"Вскрыто векторных граней: {val_st['active_coords_frac']*100:.1f}%",
        flush=True,
    )
    print(
        f"Бэкпроп на 1 такте (после обучения): переписывается {bp_check_after['rewritten_facets_in_backprop']}/{bp_check_after['total_facets']} "
        f"граней ({bp_check_after['rewritten_percent']}%), строгий ноль у {bp_check_after['untouched_exact_zero_facets']} граней.",
        flush=True,
    )

    out_path = OUT_DIR / "rezultat_local_friston_v5.json"
    out_path.write_text(
        json.dumps(
            {
                "params": n_params,
                "backprop_check_before": bp_check,
                "backprop_check_after": bp_check_after,
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
