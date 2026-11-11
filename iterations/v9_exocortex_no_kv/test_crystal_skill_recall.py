"""Проверка защиты внешних кристаллов памяти/навыков в 3D-комнате от затирания фоновым шумом
на дистанции 10 000 шагов при 0 байт KV-кэша!

Почему в наивном буфере (как в первом тесте v9) длинный фоновый текст затирал записанный навык:
  Потому что при постоянной скорости перезаписи write_rate = const фоновый шум за 1000 шагов
  экспоненциально вымывал старую запись: (1 - alpha)^1000 -> 0.

Как работает настоящая КРИСТАЛЛИЗАЦИЯ НАВЫКА во внешней 3D-комнате:
1. Каждый внешний узел-кристалл m в 3D-комнате хранит:
   - 3D-координату r_room[m] и цветовой ключ key_rgbl[m],
   - вектор контекста-триггера key_vec[m] (при каком состоянии внешней среды нужен этот навык),
   - вектор навыка skill_vec[m] = w_ext(t+1) - w_ext(t) (куда нужно сдвинуть прогноз, чтобы выполнить навык!),
   - твердость (массу кристаллизации) hardness[m] >= 0.
2. Запись вовне идет в СВОБОДНЫЙ кристалл (с минимальной твердостью hardness[m]) или в кристалл с тем же ключом,
   увеличивая его твердость hardness[m]! После кристаллизации обычный фоновый текст с другой темой
   НЕ МОЖЕТ перезаписать твердый кристалл!
3. Обращение по необходимости (Read on Need):
   Когда в потоке (хоть через 10 000 шагов!) снова возникает контекст, резонирующий с ключом кристалла
   выше локального порога необходимости z_need, кристалл мгновенно выдает точный вектор навыка skill_vec[m]!
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from progon_chestnaya_3d import load_wikitext_bytes

OUT_DIR = Path(__file__).resolve().parent


def run_crystal_exocortex_test():
    torch.manual_seed(42)
    valid_data = load_wikitext_bytes("validation", max_chars=50_000)

    d_model = 64
    n_crystals = 32
    embed = F.normalize(torch.randn(256, d_model), dim=-1)

    # Внешняя 3D-комната из 32 кристаллов (константный размер памяти, 0 байт KV-кэша)
    crystal_pos = F.normalize(torch.randn(n_crystals, 3), dim=-1) * 2.0
    crystal_keys = torch.zeros(n_crystals, d_model)
    crystal_skills = torch.zeros(n_crystals, d_model)
    crystal_hardness = torch.zeros(n_crystals)  # твердость кристалла (защита от затирания шумом!)

    def step_wave_state(wave_state: torch.Tensor, byte_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Волновое состояние контекста с фазовым сдвигом (различает "ALPHA->" и "TESLA->" в 3D-комнате!)
        rotated = torch.roll(wave_state, shifts=1, dims=0)
        new_state = F.normalize(0.55 * rotated + 0.45 * embed[byte_id], dim=-1)
        pos3d = F.normalize(new_state[:3], dim=-1) * 2.0
        return new_state, pos3d

    def write_skill_to_room(sig: torch.Tensor, pos3d: torch.Tensor, target_next_vec: torch.Tensor, surprise_lum: float):
        # Ищем, есть ли уже кристалл с таким же ключом в этом секторе 3D-комнаты
        cos_key = torch.mv(crystal_keys, sig)
        same_mask = cos_key > 0.85
        if bool(same_mask.any()):
            slot = int(torch.argmax(cos_key).item())
            # Укрепляем существующий кристалл
            w = 1.0 / (1.0 + crystal_hardness[slot])
            crystal_keys[slot] = F.normalize((1.0 - w) * crystal_keys[slot] + w * sig, dim=-1)
            crystal_skills[slot] = (1.0 - w) * crystal_skills[slot] + w * (target_next_vec - sig)
            crystal_hardness[slot] += surprise_lum
        else:
            # Иначе записываем только если яркость сюрприза превышает твердость самого свободного слота комнаты!
            min_hard, slot_t = torch.min(crystal_hardness, dim=0)
            slot = int(slot_t.item())
            if surprise_lum > float(min_hard.item()):
                crystal_pos[slot] = pos3d
                crystal_keys[slot] = sig
                crystal_skills[slot] = target_next_vec - sig
                crystal_hardness[slot] = surprise_lum

    def recall_skill_on_need(sig: torch.Tensor, pos3d: torch.Tensor, z_need: float = 0.75) -> tuple[torch.Tensor, int]:
        # 3D-близость луча + резонанс ключа
        cos_3d = F.relu(torch.mv(crystal_pos, pos3d) / 4.0)
        cos_key = torch.mv(crystal_keys, sig)
        resonance = cos_3d * cos_key
        # Порог необходимости: если резонанс ниже z_need, внешняя память не трогается (строгий 0.0)!
        excess = F.relu(resonance - z_need)
        active = int((excess > 0.0).sum().item())
        if active == 0:
            return torch.zeros_like(sig), 0
        weights = excess / (excess.sum() + 1e-8)
        recalled_delta = torch.mv(crystal_skills.T, weights)
        return recalled_delta, active

    # 1. Обучаем систему 5 редким навыкам-правилам во внешней комнате (высокая яркость сюрприза L = 5.0)
    rules = [
        b"ALPHA->OMEGA",
        b"SIGMA->DELTA",
        b"QUARK->GLUON",
        b"TESLA->GAUSS",
        b"LUMEN->NANIT",
    ]
    for phrase in rules:
        seq = list(phrase)
        w_state = torch.zeros(d_model)
        for i in range(len(seq) - 1):
            w_state, pos3d = step_wave_state(w_state, seq[i])
            if i >= 5:
                nxt_vec = embed[seq[i + 1]]
                write_skill_to_room(w_state, pos3d, nxt_vec, surprise_lum=5.0)

    # 2. Прогоняем 10 000 шагов реального фонового текста WikiText-2 (обычный фоновый поток с яркостью L = 0.8)
    distractor_len = 10_000
    false_recalls_during_noise = 0
    w_noise = torch.zeros(d_model)
    for t in range(distractor_len - 1):
        w_noise, pos3d = step_wave_state(w_noise, int(valid_data[t]))
        _, act = recall_skill_on_need(w_noise, pos3d)
        if act > 0:
            false_recalls_during_noise += 1
        write_skill_to_room(w_noise, pos3d, embed[int(valid_data[t + 1])], surprise_lum=0.8)

    # 3. Через 10 000 шагов подаем начала всех 5 правил и проверяем точность генерации хвоста
    correct_with_room = 0
    correct_without_room = 0
    total_queries = 0
    recalled_outputs = []

    for phrase in rules:
        seq = list(phrase)
        gen_bytes = list(seq[:7])  # например "ALPHA->"
        w_state = torch.zeros(d_model)
        for b_id in gen_bytes[:-1]:
            w_state, _ = step_wave_state(w_state, b_id)

        cur_b = gen_bytes[-1]
        for i in range(6, len(seq) - 1):
            w_state, pos3d = step_wave_state(w_state, cur_b)
            target_id = seq[i + 1]

            # Без внешней комнаты
            pred_no_room = int(torch.argmax(torch.mv(embed, w_state)).item())
            if pred_no_room == target_id:
                correct_without_room += 1

            # С обращением к внешней 3D-комнате по необходимости
            delta_skill, _ = recall_skill_on_need(w_state, pos3d)
            pred_vec = w_state + delta_skill
            pred_with_room = int(torch.argmax(torch.mv(embed, pred_vec)).item())
            if pred_with_room == target_id:
                correct_with_room += 1
            gen_bytes.append(pred_with_room)
            cur_b = pred_with_room
            total_queries += 1
        recalled_outputs.append(bytes(gen_bytes).decode("utf-8", errors="replace"))

    res = {
        "distractor_steps_passed": distractor_len,
        "kv_cache_bytes": 0,
        "room_crystals_count": n_crystals,
        "room_memory_bytes_fixed": n_crystals * d_model * 4 * 2,
        "false_recalls_rate_on_noise_pct": round(100.0 * false_recalls_during_noise / distractor_len, 2),
        "accuracy_without_external_room_pct": round(100.0 * correct_with_room * 0 / max(1, total_queries) + 100.0 * correct_without_room / total_queries, 1),
        "accuracy_with_external_3d_room_pct": round(100.0 * correct_with_room / total_queries, 1),
        "recalled_sequences_after_10000_steps": recalled_outputs,
    }
    print("=== ТЕСТ КРИСТАЛЛИЗАЦИИ НАВЫКОВ ВО ВНЕШНЕЙ 3D-КОМНАТЕ ЧЕРЕЗ 10 000 ШАГОВ БЕЗ KV-КЭША ===", flush=True)
    print(f"Пройдено фонового текста: {distractor_len:,} байт | Размер KV-кэша: 0 байт | Память комнаты: константа {res['room_memory_bytes_fixed']} байт", flush=True)
    print(f"Ложных срабатываний во время фонового текста: {res['false_recalls_rate_on_noise_pct']}% (система обращается только по необходимости!)", flush=True)
    print(f"Точность воспроизведения навыков БЕЗ внешней комнаты: {res['accuracy_without_external_room_pct']}%", flush=True)
    print(f"Точность воспроизведения навыков С внешней 3D-комнатой: {res['accuracy_with_external_3d_room_pct']}%", flush=True)
    print(f"Восстановленные последовательности через 10 000 шагов: {res['recalled_sequences_after_10000_steps']}", flush=True)
    (OUT_DIR / "rezultat_crystal_skills_10k.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_crystal_exocortex_test()
