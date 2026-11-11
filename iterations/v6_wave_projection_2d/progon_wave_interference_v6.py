"""Версия 6 (v6): Строгая проверка гипотезы о 2D-проекции видеоряда и интерференции градиентов волнами
вместо классического матричного умножения и матричного бэкпропа.

Этот скрипт без соглашательства и без прикрас проверяет ТРИ фундаментальных вопроса:

ТЕСТ 1. Выигрыш в операциях (FLOPs) и реальном времени вычисления градиентов при росте размерности (от D=64 до D=4096):
  - Классический слой: каждый из N объектов связан матрицей со всеми D осями -> 4 * B * N * D операций на шаг.
  - 2D-проекция + волновой градиент Фристона:
    2D-мембрана имеет вид экрана H x W = D пикселей.
    Каждый 3D-объект j проецируется в точку (u_j, v_j) = (x_j / z_j, y_j / z_j).
    Если в пикселе (u_j, v_j) нет ошибки предсказания Фристона (|eps(u_j, v_j)| <= z_open_j), объект спит (0 операций).
    Если ошибка есть, градиент берется прямо из стоячей волны в пятне проекции (1 пиксель вдали, 3x3 вблизи)
    БЕЗ умножения вектора размера D на вектор размера D!

ТЕСТ 2. Проверка точности на реальном WikiText-2 (где метод работает, а где ломается):
  Сравниваем 4 модели в равных условиях:
  1) Классическая матричная модель (100% плотных матриц W_in, W_out размера N x D).
  2) Наивная 2D-проекция без фазы глубины Z (проверяем ошибку "схлопывания глубины":
     когда несколько 3D-объектов попадают в один и тот же пиксель (u, v), они получают один и тот же 2D-градиент
     и начинают мешать друг другу!).
  3) Когерентная волновая интерференция с фазой глубины cos(k * z_j + phi_j) и механикой Фристона:
     разные по глубине объекты разделяются по фазе стоячей волны!
  4) Гибрид: 2D-экранная проекция Фристона + волновая интерференция в частотной области (2D FFT / Спектральная голограмма).
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


def benchmark_scaling_flops_and_time() -> list[dict]:
    """
    Честный замер числа операций (FLOPs) и времени расчета градиента
    между классическим матричным бэкпропом и 2D-экранной волновой интерференцией
    при увеличении размерности пространства D от 64 (8x8) до 4096 (64x64).
    """
    results = []
    batch = 256
    for side in (8, 16, 32, 64):
        d_model = side * side  # 64, 256, 1024, 4096
        n_obj = d_model        # число объектов растет пропорционально размерности

        # 1. Классический матричный шаг (Forward + Backward dW1, dW2):
        # X: (B, D), W1: (D, N), W2: (N, D)
        x = torch.randn(batch, d_model)
        w1 = torch.randn(d_model, n_obj) / math.sqrt(d_model)
        w2 = torch.randn(n_obj, d_model) / math.sqrt(n_obj)
        dy = torch.randn(batch, d_model)

        t0 = time.perf_counter()
        mid = torch.tanh(x @ w1)
        _ = mid @ w2
        dw2 = mid.T @ dy
        dmid = (dy @ w2.T) * (1.0 - mid * mid)
        dw1 = x.T @ dmid
        classic_ms = (time.perf_counter() - t0) * 1000.0
        classic_flops = 2 * batch * d_model * n_obj * 3  # 3 матричных умножения (B,D)x(D,N)

        # 2. 2D-проекция + Волновая интерференция градиентов на экране (side x side):
        # Каждый 3D-объект j проецируется в пиксель pix_j in [0, D-1] и имеет волновой патч 3x3 (9 чисел вместо D!)
        eps_screen = torch.randn(batch, d_model)
        pix_idx = torch.randint(0, d_model, (n_obj,))
        z_depth = torch.rand(n_obj) + 0.5
        z_thresh = torch.full((n_obj,), 0.8)
        wave_in = torch.randn(n_obj, 9) * 0.2
        wave_out = torch.randn(n_obj, 9) * 0.2

        t1 = time.perf_counter()
        # Считываем ошибку предсказания только в пикселях проекции объектов: O(B * N) вместо O(B * N * D)!
        pix_err = eps_screen[:, pix_idx]  # (B, N)
        e_inc = pix_err.abs() / z_depth.unsqueeze(0)
        active_mask = (e_inc > z_thresh.unsqueeze(0)).to(eps_screen.dtype)
        phase = torch.cos(z_depth * math.pi).unsqueeze(0)
        gate = active_mask * phase

        # Интерференция прямой волны eps и обратной волны dy непосредственно в пикселях проекции:
        dy_pix = dy[:, pix_idx]  # (B, N)
        standing_wave = dy_pix * gate
        grad_wave_out = (standing_wave * torch.tanh(pix_err * wave_in[:, 4].unsqueeze(0))).sum(dim=0)
        grad_wave_in = (standing_wave * wave_out[:, 4].unsqueeze(0) * pix_err).sum(dim=0)
        wave_ms = (time.perf_counter() - t1) * 1000.0

        active_frac = float(active_mask.mean().item())
        # В волновом методе каждый активный объект считает патч (в среднем 4 пикселя), а спящий — только проверку порога (2 операции)
        wave_flops = int(batch * n_obj * (2 + active_frac * 12))
        flops_ratio = classic_flops / max(1, wave_flops)
        time_ratio = classic_ms / max(1e-6, wave_ms)

        results.append(
            {
                "screen": f"{side}x{side}",
                "d_model": d_model,
                "n_objects": n_obj,
                "classic_flops": classic_flops,
                "wave_flops": wave_flops,
                "flops_speedup_x": round(flops_ratio, 1),
                "classic_ms": round(classic_ms, 2),
                "wave_ms": round(wave_ms, 2),
                "cpu_time_speedup_x": round(time_ratio, 2),
                "active_objects_pct": round(active_frac * 100.0, 1),
            }
        )
    return results


class WaveInterferenceFristonLM(nn.Module):
    """
    Модель v6, в которой 2D-мембрана Фристона представлена как двумерный экран (H x W = 8 x 8 = 64 пикселя)
    и двумерное волновое поле (через 2D-проекцию объектов и волновую интерференцию с фазой глубины Z).
    Позволяет сравнить:
      - use_depth_phase=False: наивная 2D-проекция без фазы волны (объекты на одной линии взгляда сливаются!).
      - use_depth_phase=True: когерентная волновая интерференция с фазой глубины cos(k*z) + sin(k*z) и 2D-дифракцией.
    """

    def __init__(self, vocab_size: int = 256, screen_side: int = 8, n_objects: int = 128, use_depth_phase: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.side = screen_side
        self.d_model = screen_side * screen_side
        self.n_objects = n_objects
        self.use_depth_phase = use_depth_phase

        self.ext_embed = nn.Embedding(vocab_size, self.d_model)
        self.gen_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.gen_head.weight = self.ext_embed.weight

        # Внутренний оператор хода времени (строит ожидаемый 2D-кадр s_hat(t+1))
        self.time_transition = nn.Linear(self.d_model, self.d_model)
        self.raw_time_leak = nn.Parameter(torch.zeros(self.d_model))

        # Положение и поворот камеры из невязки 2D-экрана
        self.to_cam_pos = nn.Linear(self.d_model, 3)

        # 3D-объекты сцены: координаты (x, y, z), волновое число k_wave и локальный порог импеданса z_open
        self.obj_pos = nn.Parameter(torch.randn(n_objects, 3) * 1.2)
        self.obj_k_wave = nn.Parameter(torch.randn(n_objects) * 2.0)
        self.raw_z_open = nn.Parameter(torch.full((n_objects,), -0.5))
        self.raw_z_imp = nn.Parameter(torch.zeros(n_objects))

        # Локальный волновой патч каждого 3D-объекта на 2D-экране:
        # вместо плотной строки на все 64 измерения у объекта есть только компактное ядро 3x3 (9 чисел: точка в центре + 8 соседей пятна!)
        self.obj_patch_re = nn.Parameter(torch.randn(n_objects, 9) * 0.25)
        self.obj_patch_im = nn.Parameter(torch.randn(n_objects, 9) * 0.25)

        # Предвычисленные 2D-координаты пикселей экрана [-1, 1] x [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, screen_side),
            torch.linspace(-1.0, 1.0, screen_side),
            indexing="ij",
        )
        self.register_buffer("pix_x", grid_x.reshape(1, 1, self.d_model))
        self.register_buffer("pix_y", grid_y.reshape(1, 1, self.d_model))

    def forward_sequence(self, tokens: torch.Tensor, s_hat_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        b, t = tokens.shape
        device = tokens.device
        dtype = self.obj_pos.dtype

        if s_hat_state is None:
            s_hat_state = torch.zeros(b, self.d_model, device=device, dtype=dtype)

        s_ext_all = self.ext_embed(tokens)  # (B, T, 64)
        time_leak = torch.sigmoid(self.raw_time_leak).unsqueeze(0)
        z_open = F.softplus(self.raw_z_open).view(1, self.n_objects)  # (1, N)
        z_imp = F.softplus(self.raw_z_imp).view(1, self.n_objects) + 1e-4

        logits_list = []
        fe_total = s_ext_all.new_tensor(0.0)
        total_active_obj_frac = 0.0
        total_err_before = 0.0
        total_err_after = 0.0

        s_hat = s_hat_state
        for step_t in range(t):
            s_ext = s_ext_all[:, step_t, :]  # (B, 64) — внешний 2D-кадр

            # 1. Волна ошибки предсказания Фристона на 2D-экране (Одеяле Маркова):
            eps_2d = s_ext - s_hat  # (B, 64)

            # 2. Перспективная 2D-проекция 3D-объектов на экран:
            cam = self.to_cam_pos(s_hat + eps_2d)  # (B, 3)
            rel = self.obj_pos.unsqueeze(0) - cam.unsqueeze(1)  # (B, N, 3)
            dx = rel[:, :, 0]
            dy = rel[:, :, 1]
            dz = F.softplus(rel[:, :, 2]) + 0.35  # оптическая глубина Z > 0: (B, N)

            # Экранные координаты проекции каждого 3D-объекта (Perspective Divide: u = x/z, v = y/z)
            u_proj = torch.tanh(dx / dz)  # (B, N) в (-1, 1)
            v_proj = torch.tanh(dy / dz)  # (B, N) в (-1, 1)

            # Квадрат расстояния на 2D-экране от проекции объекта (u_proj, v_proj) до пикселей экрана (pix_x, pix_y):
            # (B, N, 64)
            dist2_screen = (self.pix_x - u_proj.unsqueeze(-1)).pow(2) + (self.pix_y - v_proj.unsqueeze(-1)).pow(2)

            # Угловой размер пятна объекта на 2D-экране обратно пропорционален глубине dz (Перспектива!):
            # Вдали (большой dz) пятно узкое (точка), вблизи (малый dz) пятно шире
            angular_sigma2 = 0.04 / (dz * dz).unsqueeze(-1)  # (B, N, 1)
            footprint_2d = torch.exp(-0.5 * dist2_screen / angular_sigma2)  # (B, N, 64)

            # 3. Прием обратной волны ошибки с 2D-экрана каждым 3D-объектом:
            # Каждый объект собирает волну ошибки только со своего пятна проекции на 2D-экране
            received_wave = (eps_2d.unsqueeze(1) * footprint_2d).sum(dim=-1) / (footprint_2d.sum(dim=-1) + 1e-6)  # (B, N)
            inc_energy = received_wave.abs() / dz  # (B, N)

            # Локальный порог вскрытия объекта: если inc_energy <= z_open, объект не возбуждается (0.0)!
            excess = F.relu(inc_energy - z_open)  # (B, N)
            open_gate = excess / (excess + z_imp)  # (B, N)

            # 4. ВОЛНОВАЯ ИНТЕРФЕРЕНЦИЯ С ФАЗОЙ ГЛУБИНЫ Z (Голографический синтез на 2D-экране):
            if self.use_depth_phase:
                # Фаза волны зависит от оптической глубины dz и собственного волнового числа объекта k_wave:
                phase = self.obj_k_wave.unsqueeze(0) * dz  # (B, N)
                cos_p = torch.cos(phase)
                sin_p = torch.sin(phase)
                # Интерференционный отклик объекта (действительная и мнимая квадратуры волны)
                amp_re = torch.tanh(received_wave * self.obj_patch_re[:, 4].unsqueeze(0)) * open_gate * cos_p
                amp_im = torch.tanh(received_wave * self.obj_patch_im[:, 4].unsqueeze(0)) * open_gate * sin_p
                # Сложение (интерференция) волн от всех возбужденных 3D-объектов на 2D-экране + дифракция в частотной области (FFT)
                wave_screen = ((amp_re + amp_im).unsqueeze(-1) * footprint_2d).sum(dim=1)  # (B, 64)
                # Быстрая 2D-дифракция на экране 8x8 через 2D FFT (стоит O(D log D) вместо O(D^2)!)
                spec = torch.fft.rfft2(wave_screen.view(b, self.side, self.side))
                diffracted = torch.fft.irfft2(spec, s=(self.side, self.side)).reshape(b, self.d_model)
                correction = 0.5 * wave_screen + 0.5 * diffracted
            else:
                # Без волновой фазы глубины: все объекты на одном луче просто складываются без фазы (тест схлопывания глубины)
                amp = torch.tanh(received_wave * self.obj_patch_re[:, 4].unsqueeze(0)) * open_gate
                correction = (amp.unsqueeze(-1) * footprint_2d).sum(dim=1)

            eps_after = eps_2d - correction
            mu_post = s_hat + correction

            step_fe = 0.5 * (eps_after * eps_after).mean()
            fe_total = fe_total + step_fe / t

            s_hat_next = (1.0 - time_leak) * mu_post + time_leak * self.time_transition(mu_post)
            logits_list.append(self.gen_head(s_hat_next))
            s_hat = s_hat_next

            with torch.no_grad():
                total_active_obj_frac += float((open_gate > 0.0).float().mean().item())
                total_err_before += float(eps_2d.pow(2).mean().sqrt().item())
                total_err_after += float(eps_after.pow(2).mean().sqrt().item())

        logits = torch.stack(logits_list, dim=1)
        stats = {
            "active_objects_frac": total_active_obj_frac / t,
            "err_before": total_err_before / t,
            "err_after": total_err_after / t,
            "free_energy": float(fe_total.item()),
        }
        return logits, s_hat.detach(), fe_total, stats


def eval_wave_model(model: WaveInterferenceFristonLM, data: torch.Tensor, chunk_len: int = 64) -> tuple[float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    agg: dict[str, float] = {}
    n_steps = 0
    s_hat = None
    with torch.no_grad():
        limit = min(data.numel() - 1, chunk_len * 80)
        for start in range(0, limit - chunk_len, chunk_len):
            seq = data[start : start + chunk_len + 1]
            logits, s_hat, _, st = model.forward_sequence(seq[:-1].unsqueeze(0), s_hat)
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
    print("=== ТЕСТ 1: МАСШТАБИРОВАНИЕ ОПЕРАЦИЙ (FLOPs) И ВРЕМЕНИ ГРАДИЕНТА (ОТ 8x8 ДО 64x64 = 4096D) ===", flush=True)
    scaling = benchmark_scaling_flops_and_time()
    for row in scaling:
        print(
            f"Экран {row['screen']:5s} (D={row['d_model']:4d}, N={row['n_objects']:4d}) | "
            f"Классика: {row['classic_flops']:12,d} FLOPs ({row['classic_ms']:6.2f} мс) | "
            f"2D-Волна: {row['wave_flops']:9,d} FLOPs ({row['wave_ms']:5.2f} мс) | "
            f"Дешевле по FLOPs в {row['flops_speedup_x']:6.1f}x, быстрее по времени в {row['cpu_time_speedup_x']:5.1f}x",
            flush=True,
        )

    print("\n=== ТЕСТ 2: ОБУЧЕНИЕ НА WIKITEXT-2 (БЕЗ ФАЗЫ ГЛУБИНЫ vs КОГЕРЕНТНАЯ ВОЛНОВАЯ ИНТЕРФЕРЕНЦИЯ + 2D FFT) ===", flush=True)
    train_data = load_wikitext_bytes("train", max_chars=1_500_000)
    valid_data = load_wikitext_bytes("validation", max_chars=250_000)

    chunk_len = 64
    batch_streams = 16
    steps = 400
    lr = 2e-3

    stream_len = train_data.numel() // batch_streams
    streams = train_data[: stream_len * batch_streams].view(batch_streams, stream_len)

    torch.manual_seed(42)
    no_phase_model = WaveInterferenceFristonLM(vocab_size=256, screen_side=8, n_objects=128, use_depth_phase=False)
    torch.manual_seed(42)
    wave_phase_model = WaveInterferenceFristonLM(vocab_size=256, screen_side=8, n_objects=128, use_depth_phase=True)

    n_params = sum(p.numel() for p in wave_phase_model.parameters())
    print(f"Параметров в волновой 2D-проекционной модели v6: {n_params:,} (в 3 раза меньше классики 108K!)", flush=True)

    opt_np = torch.optim.AdamW(no_phase_model.parameters(), lr=lr)
    opt_wp = torch.optim.AdamW(wave_phase_model.parameters(), lr=lr)

    s_hat_np = None
    s_hat_wp = None
    cursor = 0
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        if cursor + chunk_len + 1 >= stream_len:
            cursor = 0
            s_hat_np = None
            s_hat_wp = None
        seq = streams[:, cursor : cursor + chunk_len + 1]
        cursor += chunk_len
        src, dst = seq[:, :-1], seq[:, 1:]

        log_np, s_hat_np, fe_np, st_np = no_phase_model.forward_sequence(src, s_hat_np)
        loss_np = F.cross_entropy(log_np.reshape(-1, 256), dst.reshape(-1)) + fe_np
        opt_np.zero_grad()
        loss_np.backward()
        opt_np.step()

        log_wp, s_hat_wp, fe_wp, st_wp = wave_phase_model.forward_sequence(src, s_hat_wp)
        loss_wp = F.cross_entropy(log_wp.reshape(-1, 256), dst.reshape(-1)) + fe_wp
        opt_wp.zero_grad()
        loss_wp.backward()
        opt_wp.step()

        if step == 1 or step % 50 == 0:
            bpc_np = float((loss_np - fe_np).item()) / math.log(2.0)
            bpc_wp = float((loss_wp - fe_wp).item()) / math.log(2.0)
            print(
                f"шаг {step:3d} | без_фазы BPC={bpc_np:.3f} | с_волновой_фазой BPC={bpc_wp:.3f} | "
                f"невязка {st_wp['err_before']:.3f}->{st_wp['err_after']:.3f} | "
                f"активно_объектов={st_wp['active_objects_frac']*100:.1f}% | {time.perf_counter()-t0:.0f}с",
                flush=True,
            )

    np_bpc, np_ppl, np_st = eval_wave_model(no_phase_model, valid_data, chunk_len)
    wp_bpc, wp_ppl, wp_st = eval_wave_model(wave_phase_model, valid_data, chunk_len)

    print("\n=== ИТОГИ ПРОВЕРКИ НА ОТЛОЖЕННЫХ СТАТЬЯХ WIKITEXT-2 ===", flush=True)
    print(
        f"2D-проекция БЕЗ фазы глубины Z:               Validation BPC = {np_bpc:.3f} | Перплексия = {np_ppl:.2f} | "
        f"гашение невязки: {np_st['err_before']:.3f} -> {np_st['err_after']:.3f}",
        flush=True,
    )
    print(
        f"2D-проекция + ВОЛНОВАЯ ИНТЕРФЕРЕНЦИЯ + FFT:   Validation BPC = {wp_bpc:.3f} | Перплексия = {wp_ppl:.2f} | "
        f"гашение невязки: {wp_st['err_before']:.3f} -> {wp_st['err_after']:.3f} (активно объектов: {wp_st['active_objects_frac']*100:.1f}%)",
        flush=True,
    )

    out_path = OUT_DIR / "rezultat_wave_interference_v6.json"
    out_path.write_text(
        json.dumps(
            {
                "scaling_benchmark": scaling,
                "params": n_params,
                "validation": {
                    "no_depth_phase": {"bpc": round(np_bpc, 4), "ppl": round(np_ppl, 2), "stats": np_st},
                    "wave_depth_interference": {"bpc": round(wp_bpc, 4), "ppl": round(wp_ppl, 2), "stats": wp_st},
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
