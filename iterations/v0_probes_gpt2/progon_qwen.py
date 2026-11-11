"""Настоящая Qwen2.5-0.5B: обычный ответ и ответ через перспективу.

Колонка весов — объект. Текущая активация — взгляд.
Большая активация: колонка целиком.
Средняя: крупный блок из 16 ячеек.
Малая: одно число, среднее колонки.
Результат слоя становится взглядом следующего слоя.
"""

import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK = 16
QUESTIONS = [
    "Столица Франции? Ответь одним словом.",
    "Сколько будет 2+2? Ответь только числом.",
]


def prepare(linear: nn.Linear):
    weight = linear.weight.detach()
    out, inn = weight.shape
    pad = (BLOCK - out % BLOCK) % BLOCK
    padded = torch.nn.functional.pad(weight, (0, 0, 0, pad))
    coarse = padded.view(-1, BLOCK, inn).mean(dim=1).contiguous()
    col_mean = weight.mean(dim=0).contiguous()
    return weight, coarse, col_mean, pad


def perspective(flat, weight, coarse, col_mean, pad, near_t, mid_t):
    out, inn = weight.shape
    y = torch.zeros(flat.shape[0], out, dtype=flat.dtype)
    resident = 0
    mid_cost = coarse.shape[0]
    for row in range(flat.shape[0]):
        imp = flat[row].abs()
        order = torch.argsort(imp, descending=True)
        if near_t >= 1:
            budget = int(mid_t * out * inn)
            near_n = min(inn, budget // max(1, out))
            left = budget - near_n * out
            mid_n = min(inn - near_n, left // max(1, mid_cost))
            near_idx = order[:near_n]
            mid_idx = order[near_n : near_n + mid_n]
            far_idx = order[near_n + mid_n :]
        else:
            peak = imp.max().clamp_min(1e-8)
            close = (imp / peak).square()
            near_idx = torch.where(close > near_t)[0]
            mid_idx = torch.where((close > mid_t) & (close <= near_t))[0]
            far_idx = torch.where(close <= mid_t)[0]
        if near_idx.numel():
            y[row] = y[row] + flat[row, near_idx] @ weight[:, near_idx].T
        if mid_idx.numel():
            coarse_y = flat[row, mid_idx] @ coarse[:, mid_idx].T
            y[row] = y[row] + coarse_y.repeat_interleave(BLOCK)[:out]
        if far_idx.numel():
            y[row] = y[row] + (flat[row, far_idx] * col_mean[far_idx]).sum()
        resident += int(near_idx.numel()) * out + int(mid_idx.numel()) * mid_cost + int(far_idx.numel())
    total = flat.shape[0] * weight.numel()
    return y, resident, total


def patch(model, near_t, mid_t, stats):
    originals = []
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        weight, coarse, col_mean, pad = prepare(module)

        def fwd(x, weight=weight, coarse=coarse, col_mean=col_mean, pad=pad, bias=module.bias):
            lead = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            y, resident, total = perspective(flat, weight, coarse, col_mean, pad, near_t, mid_t)
            stats["resident"] += resident
            stats["total"] += total
            if bias is not None:
                y = y + bias
            return y.view(*lead, y.shape[-1])

        originals.append((module, module.forward))
        module.forward = fwd
    return originals


def restore(originals):
    for module, fwd in originals:
        module.forward = fwd


def ask(model, tokenizer, question):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **tokens,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    seconds = time.perf_counter() - t0
    text = tokenizer.decode(out[0][tokens["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip(), seconds


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    modes = [
        ("обычная", None, None),
        ("перспектива строгая", 0.50, 0.15),
        ("перспектива мягкая", 0.15, 0.02),
        ("бюджет карты 4/10", 1.0, 0.40),
    ]
    for name, near_t, mid_t in modes:
        print(f"\n=== {name} ===", flush=True)
        stats = {"resident": 0, "total": 0}
        originals = []
        if near_t is not None:
            originals = patch(model, near_t, mid_t, stats)
        for question in QUESTIONS:
            text, seconds = ask(model, tokenizer, question)
            print(f"вопрос: {question}", flush=True)
            print(f"ответ: {text}", flush=True)
            print(f"сек: {seconds:.1f}", flush=True)
        if originals:
            share = stats["resident"] / max(1, stats["total"])
            print(f"доля_ячеек_на_площадке: {share:.3f}", flush=True)
            restore(originals)


if __name__ == "__main__":
    main()
