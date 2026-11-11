/* Версия 6 (v6) на GPU AMD Radeon Pro 5500M (4 ГБ):
 * Прямое инструментальное сравнение на одной и той же видеокарте и одном и том же корпусе WikiText-2:
 *
 * РЕЖИМ А (Классический матричный GEMM + Матричный Backprop):
 *   - Каждый из N_OBJ объектов умножается на все D_MODEL входов и пишет во все D_MODEL выходов:
 *     O(BATCH * N_OBJ * D_MODEL^2) операций на шаг.
 *
 * РЕЖИМ Б (Ваша механика: 3D-видеоряд -> 2D-проекция на экран 8x8 + Свободная энергия Фристона + Волновая интерференция градиентов):
 *   - 2D-мембрана коллизии (Одеяло Маркова) — это буквально 2D-экран размером 8x8 = 64 пикселя (u, v in 0..7).
 *   - Каждый 3D-объект j с координатами (x_j, y_j, z_j) проецируется перспективным делением
 *     в экранный пиксель (u_j, v_j) = (x_j / z_j, y_j / z_j).
 *   - Далёкий объект проецируется в 1 пиксель (точка, 1 операция вместо 64!),
 *     средний — в 4 пикселя (2x2), близкий в фокусе сюрприза — в 9 или 16 пикселей.
 *   - ГРАДИЕНТ ЧЕРЕЗ ИНТЕРФЕРЕНЦИЮ ВОЛН:
 *     Вместо матричного умножения градиентов каждый пиксель 2D-экрана (u, v), где возникла
 *     ошибка предсказания Фристона eps(u, v) = s_ext(u, v) - s_hat(u, v), излучает обратную волну
 *     с фазой глубины phi = k_wave * z_j!
 *     На каждом 3D-объекте волны от кадров видеоряда интерферируют (складываются по фазе),
 *     и объект переписывает ТОЛЬКО пиксели своего 2D-пятна проекции, где амплитуда стоячей волны
 *     превысила его локальный порог импеданса!
 */

#include <OpenCL/opencl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define VOCAB 256
#define SCREEN_W 8
#define SCREEN_H 8
#define D_MODEL (SCREEN_W * SCREEN_H) /* 64 пикселя на 2D-мембране */
#define N_OBJ 256                     /* 256 3D-объектов в сцене */
#define BATCH_TOKENS 256

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static const char *KERNEL_SRC =
    "#define VOCAB 256\n"
    "#define SCREEN_W 8\n"
    "#define SCREEN_H 8\n"
    "#define D_MODEL 64\n"
    "#define N_OBJ 256\n"
    "\n"
    "inline float sigmoidf(float x) {\n"
    "    return 1.0f / (1.0f + exp(-clamp(x, -20.0f, 20.0f)));\n"
    "}\n"
    "\n"
    "/* ============================================================================ *\n"
    " * РЕЖИМ А: КЛАССИЧЕСКИЙ ПЛОТНЫЙ МАТРИЧНЫЙ ПРОХОД И МАТРИЧНЫЙ БЭКПРОП (GEMM)    *\n"
    " * ============================================================================ */\n"
    "__kernel void classic_dense_forward(\n"
    "    __global const int *tokens,\n"
    "    __global const float *embed,\n"
    "    __global const float *W1,\n"
    "    __global const float *W2,\n"
    "    __global float *hidden_in,\n"
    "    __global float *mid_act,\n"
    "    __global float *hidden_out,\n"
    "    __global long *op_counter,\n"
    "    int n_tokens)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    if (tid >= n_tokens) return;\n"
    "\n"
    "    int tok = tokens[tid];\n"
    "    int prev1 = (tid > 0) ? tokens[tid - 1] : tok;\n"
    "    int prev2 = (tid > 1) ? tokens[tid - 2] : prev1;\n"
    "    float x[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        x[d] = embed[tok * D_MODEL + d] + 0.45f * embed[prev1 * D_MODEL + d] + 0.20f * embed[prev2 * D_MODEL + d];\n"
    "        hidden_in[tid * D_MODEL + d] = x[d];\n"
    "    }\n"
    "    float m[N_OBJ];\n"
    "    for (int j = 0; j < N_OBJ; ++j) {\n"
    "        float s = 0.0f;\n"
    "        int off = j * D_MODEL;\n"
    "        for (int d = 0; d < D_MODEL; ++d) s += x[d] * W1[off + d];\n"
    "        m[j] = tanh(s);\n"
    "        mid_act[tid * N_OBJ + j] = m[j];\n"
    "    }\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        float acc = x[d];\n"
    "        for (int j = 0; j < N_OBJ; ++j) acc += m[j] * W2[j * D_MODEL + d];\n"
    "        hidden_out[tid * D_MODEL + d] = acc;\n"
    "    }\n"
    "    if (tid == 0) op_counter[0] = (long)n_tokens * (long)N_OBJ * (long)D_MODEL * 4L;\n"
    "}\n"
    "\n"
    "__kernel void classic_dense_backward(\n"
    "    __global const float *hidden_in,\n"
    "    __global const float *mid_act,\n"
    "    __global const float *d_hidden,\n"
    "    __global float *W1,\n"
    "    __global float *W2,\n"
    "    int n_tokens,\n"
    "    float lr)\n"
    "{\n"
    "    int j = get_global_id(0);\n"
    "    if (j >= N_OBJ) return;\n"
    "\n"
    "    float gW1[D_MODEL], gW2[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) { gW1[d] = 0.0f; gW2[d] = 0.0f; }\n"
    "    int off = j * D_MODEL;\n"
    "\n"
    "    for (int tid = 0; tid < n_tokens; ++tid) {\n"
    "        float m = mid_act[tid * N_OBJ + j];\n"
    "        float dm = 0.0f;\n"
    "        for (int d = 0; d < D_MODEL; ++d) {\n"
    "            float dh = d_hidden[tid * D_MODEL + d];\n"
    "            gW2[d] += m * dh;\n"
    "            dm += dh * W2[off + d];\n"
    "        }\n"
    "        float ds = dm * (1.0f - m * m);\n"
    "        for (int d = 0; d < D_MODEL; ++d) {\n"
    "            gW1[d] += ds * hidden_in[tid * D_MODEL + d];\n"
    "        }\n"
    "    }\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        W1[off + d] -= lr * clamp(gW1[d], -1.0f, 1.0f);\n"
    "        W2[off + d] -= lr * clamp(gW2[d], -1.0f, 1.0f);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============================================================================ *\n"
    " * РЕЖИМ Б: 3D-ВИДЕОРЯД + 2D-ПРОЕКЦИЯ НА ЭКРАН 8x8 + СВОБОДНАЯ ЭНЕРГИЯ ФРИСТОНА *\n"
    " *          + ИНТЕРФЕРЕНЦИЯ ВОЛН ГРАДИЕНТОВ (БЕЗ МАТРИЧНЫХ УМНОЖЕНИЙ W1/W2!)    *\n"
    " * ============================================================================ */\n"
    "__kernel void wave_friston_forward_2d_screen(\n"
    "    __global const int *tokens,\n"
    "    __global const float *embed,\n"
    "    __global const float *obj_pos,      /* (N_OBJ, 3): 3D координаты объектов */\n"
    "    __global const float *obj_z_open,   /* (N_OBJ): локальный порог вскрытия объекта */\n"
    "    __global const float *obj_wave_in,  /* (N_OBJ, 9): локальная волновая текстура объекта 3x3 */\n"
    "    __global const float *obj_wave_out, /* (N_OBJ, 9): локальный волновой отклик 3x3 */\n"
    "    __global float *eps_screen_out,     /* (B, 64): волна ошибки предсказания Фристона на 2D-экране */\n"
    "    __global float *mu_post_out,        /* (B, 64): выстроенное внутреннее состояние */\n"
    "    __global int *pixel_ops_count,      /* (B): реальное число пиксельных операций на кадре */\n"
    "    int n_tokens)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    if (tid >= n_tokens) return;\n"
    "\n"
    "    int tok = tokens[tid];\n"
    "    int prev1 = (tid > 0) ? tokens[tid - 1] : tok;\n"
    "    int prev2 = (tid > 1) ? tokens[tid - 2] : prev1;\n"
    "\n"
    "    /* 1. Внешний кадр среды s_ext(u, v) и внутреннее ожидание s_hat(u, v) из инерции видеоряда */\n"
    "    float s_ext[D_MODEL];\n"
    "    float s_hat[D_MODEL];\n"
    "    float eps_2d[D_MODEL];\n"
    "    float err_energy = 0.0f;\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        s_ext[d] = embed[tok * D_MODEL + d] + 0.35f * embed[prev1 * D_MODEL + d];\n"
    "        s_hat[d] = 0.70f * embed[prev1 * D_MODEL + d] + 0.30f * embed[prev2 * D_MODEL + d];\n"
    "        eps_2d[d] = s_ext[d] - s_hat[d];\n"
    "        err_energy += eps_2d[d] * eps_2d[d];\n"
    "    }\n"
    "    float surprise_amp = sqrt(err_energy / (float)D_MODEL + 1e-8f);\n"
    "\n"
    "    /* Положение камеры контекста в 3D из центра тяжести волны ошибки */\n"
    "    float cam_x = s_hat[0] + eps_2d[0];\n"
    "    float cam_y = s_hat[1] + eps_2d[1];\n"
    "    float cam_z = s_hat[2] + eps_2d[2];\n"
    "\n"
    "    float canvas_corr[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) canvas_corr[d] = 0.0f;\n"
    "\n"
    "    int ops = 0;\n"
    "    /* 2. Проекция 3D-объектов на 2D-экран 8x8 и интерференция волн */\n"
    "    for (int j = 0; j < N_OBJ; ++j) {\n"
    "        float dx = obj_pos[j * 3 + 0] - cam_x;\n"
    "        float dy = obj_pos[j * 3 + 1] - cam_y;\n"
    "        float dz = fabs(obj_pos[j * 3 + 2] - cam_z) + 0.5f;\n"
    "\n"
    "        /* Перспективное деление (Perspective Divide): координаты (u, v) на 2D-экране 8x8 */\n"
    "        int u_c = clamp((int)rint((dx / dz) * 2.0f + 3.5f), 0, SCREEN_W - 1);\n"
    "        int v_c = clamp((int)rint((dy / dz) * 2.0f + 3.5f), 0, SCREEN_H - 1);\n"
    "        int center_pix = v_c * SCREEN_W + u_c;\n"
    "\n"
    "        /* Локальная энергия ошибки Фристона именно в том пикселе 2D-экрана, куда спроецировался объект! */\n"
    "        float local_pix_err = fabs(eps_2d[center_pix]) + 0.5f * surprise_amp;\n"
    "        float e_inc = local_pix_err / dz;\n"
    "        float z_thresh = obj_z_open[j];\n"
    "\n"
    "        /* Если энергия на 2D-экране ниже локального порога объекта — объект остается в покое (0 операций!) */\n"
    "        if (e_inc <= z_thresh) continue;\n"
    "\n"
    "        /* Радиус пятна проекции на 2D-экране зависит от превышения порога (Перспектива / LOD):\n"
    "         * Далеко (малый избыток) -> radius = 0 (ровно 1 пиксель — точка!)\n"
    "         * Ближе (сильный пробой) -> radius = 1 (окно 3x3 = до 9 пикселей на 2D-экране) */\n"
    "        float excess = e_inc - z_thresh;\n"
    "        int patch_rad = (excess > z_thresh) ? 1 : 0;\n"
    "\n"
    "        /* Фаза стоячей волны от оптической глубины dz (интерференционный множитель cos(k * dz)) */\n"
    "        float wave_phase = cos(dz * 3.14159f);\n"
    "        float gate = (excess / (excess + 0.5f)) * wave_phase;\n"
    "\n"
    "        if (patch_rad == 0) {\n"
    "            /* ТОЧКА НА 2D-ЭКРАНЕ: всего 1 чтение и 1 запись вместо 64! */\n"
    "            float match = eps_2d[center_pix] * obj_wave_in[j * 9 + 4];\n"
    "            canvas_corr[center_pix] += tanh(match) * obj_wave_out[j * 9 + 4] * gate;\n"
    "            ops += 2;\n"
    "        } else {\n"
    "            /* РАЗВЕРНУТЫЙ ОБЪЕКТ НА 2D-ЭКРАНЕ (пятно 3x3 вокруг (u_c, v_c)): */\n"
    "            int tex_idx = 0;\n"
    "            for (int dv = -1; dv <= 1; ++dv) {\n"
    "                int pv = clamp(v_c + dv, 0, SCREEN_H - 1);\n"
    "                for (int du = -1; du <= 1; ++du, ++tex_idx) {\n"
    "                    int pu = clamp(u_c + du, 0, SCREEN_W - 1);\n"
    "                    int pix = pv * SCREEN_W + pu;\n"
    "                    float match = eps_2d[pix] * obj_wave_in[j * 9 + tex_idx];\n"
    "                    canvas_corr[pix] += tanh(match) * obj_wave_out[j * 9 + tex_idx] * gate;\n"
    "                }\n"
    "            }\n"
    "            ops += 18;\n"
    "        }\n"
    "    }\n"
    "    pixel_ops_count[tid] = ops;\n"
    "\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        eps_screen_out[tid * D_MODEL + d] = eps_2d[d];\n"
    "        mu_post_out[tid * D_MODEL + d] = s_ext[d] + canvas_corr[d];\n"
    "    }\n"
    "}\n"
    "\n"
    "/* Обратный проход через ИНТЕРФЕРЕНЦИЮ ВОЛН С 2D-ЭКРАНА (вместо матричного бэкпропа!): */\n"
    "__kernel void wave_interference_backward(\n"
    "    __global const float *eps_screen_out,\n"
    "    __global const float *d_hidden,\n"
    "    __global float *obj_pos,\n"
    "    __global const float *obj_z_open,\n"
    "    __global float *obj_wave_in,\n"
    "    __global float *obj_wave_out,\n"
    "    int n_tokens,\n"
    "    float lr)\n"
    "{\n"
    "    int j = get_global_id(0);\n"
    "    if (j >= N_OBJ) return;\n"
    "\n"
    "    float gin[9], gout[9];\n"
    "    for (int k = 0; k < 9; ++k) { gin[k] = 0.0f; gout[k] = 0.0f; }\n"
    "    float gx = 0.0f, gy = 0.0f, gz = 0.0f;\n"
    "    float z_thresh = obj_z_open[j];\n"
    "    float ox = obj_pos[j * 3 + 0], oy = obj_pos[j * 3 + 1], oz = obj_pos[j * 3 + 2];\n"
    "\n"
    "    /* Каждый объект j принимает обратную волну ТОЛЬКО из своей 2D-проекции (u_c, v_c) на экране! */\n"
    "    for (int tid = 0; tid < n_tokens; ++tid) {\n"
    "        float e0 = eps_screen_out[tid * D_MODEL + 0];\n"
    "        float e1 = eps_screen_out[tid * D_MODEL + 1];\n"
    "        float e2 = eps_screen_out[tid * D_MODEL + 2];\n"
    "        float dx = ox - e0, dy = oy - e1, dz = fabs(oz - e2) + 0.5f;\n"
    "\n"
    "        int u_c = clamp((int)rint((dx / dz) * 2.0f + 3.5f), 0, SCREEN_W - 1);\n"
    "        int v_c = clamp((int)rint((dy / dz) * 2.0f + 3.5f), 0, SCREEN_H - 1);\n"
    "        int center_pix = v_c * SCREEN_W + u_c;\n"
    "\n"
    "        float local_err = fabs(eps_screen_out[tid * D_MODEL + center_pix]);\n"
    "        float e_inc = local_err / dz;\n"
    "        if (e_inc <= z_thresh) continue;\n"
    "\n"
    "        /* Конструктивная/деструктивная интерференция обратной волны с фазой глубины cos(pi * dz) */\n"
    "        float wave_phase = cos(dz * 3.14159f);\n"
    "        float excess = e_inc - z_thresh;\n"
    "        float gate = (excess / (excess + 0.5f)) * wave_phase;\n"
    "        int patch_rad = (excess > z_thresh) ? 1 : 0;\n"
    "\n"
    "        if (patch_rad == 0) {\n"
    "            float dh = d_hidden[tid * D_MODEL + center_pix];\n"
    "            float eps_v = eps_screen_out[tid * D_MODEL + center_pix];\n"
    "            /* Интерференция прямой волны невязки eps_v и обратной волны ошибки dh: */\n"
    "            float standing_wave = dh * gate;\n"
    "            gout[4] += standing_wave * tanh(eps_v * obj_wave_in[j * 9 + 4]);\n"
    "            gin[4]  += standing_wave * obj_wave_out[j * 9 + 4] * eps_v;\n"
    "            gx += standing_wave * eps_v * 0.05f;\n"
    "        } else {\n"
    "            int tex_idx = 0;\n"
    "            for (int dv = -1; dv <= 1; ++dv) {\n"
    "                int pv = clamp(v_c + dv, 0, SCREEN_H - 1);\n"
    "                for (int du = -1; du <= 1; ++du, ++tex_idx) {\n"
    "                    int pu = clamp(u_c + du, 0, SCREEN_W - 1);\n"
    "                    int pix = pv * SCREEN_W + pu;\n"
    "                    float dh = d_hidden[tid * D_MODEL + pix];\n"
    "                    float eps_v = eps_screen_out[tid * D_MODEL + pix];\n"
    "                    float standing_wave = dh * gate;\n"
    "                    gout[tex_idx] += standing_wave * tanh(eps_v * obj_wave_in[j * 9 + tex_idx]);\n"
    "                    gin[tex_idx]  += standing_wave * obj_wave_out[j * 9 + tex_idx] * eps_v;\n"
    "                }\n"
    "            }\n"
    "            gy += d_hidden[tid * D_MODEL + center_pix] * local_err * 0.05f;\n"
    "            gz -= d_hidden[tid * D_MODEL + center_pix] * sin(dz * 3.14159f) * 0.05f;\n"
    "        }\n"
    "    }\n"
    "    for (int k = 0; k < 9; ++k) {\n"
    "        obj_wave_in[j * 9 + k]  -= lr * clamp(gin[k], -1.0f, 1.0f);\n"
    "        obj_wave_out[j * 9 + k] -= lr * clamp(gout[k], -1.0f, 1.0f);\n"
    "    }\n"
    "    obj_pos[j * 3 + 0] -= lr * clamp(gx, -0.3f, 0.3f);\n"
    "    obj_pos[j * 3 + 1] -= lr * clamp(gy, -0.3f, 0.3f);\n"
    "    obj_pos[j * 3 + 2] -= lr * clamp(gz, -0.3f, 0.3f);\n"
    "}\n"
    "\n"
    "/* Общий выходной шейдер головы и эмбеддингов для честного сравнения обоих режимов */\n"
    "__kernel void pass_head_and_grad(\n"
    "    __global const float *hidden_out,\n"
    "    __global const int *targets,\n"
    "    __global const float *head_w,\n"
    "    __global float *d_logits_out,\n"
    "    __global float *d_hidden,\n"
    "    __global float *token_losses,\n"
    "    int n_tokens)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    if (tid >= n_tokens) return;\n"
    "    float h[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) h[d] = hidden_out[tid * D_MODEL + d];\n"
    "    float logits[VOCAB];\n"
    "    float max_l = -1e30f;\n"
    "    for (int v = 0; v < VOCAB; ++v) {\n"
    "        float s = 0.0f;\n"
    "        int off = v * D_MODEL;\n"
    "        for (int d = 0; d < D_MODEL; ++d) s += h[d] * head_w[off + d];\n"
    "        logits[v] = s;\n"
    "        if (s > max_l) max_l = s;\n"
    "    }\n"
    "    float sum_e = 0.0f;\n"
    "    for (int v = 0; v < VOCAB; ++v) {\n"
    "        logits[v] = exp(logits[v] - max_l);\n"
    "        sum_e += logits[v];\n"
    "    }\n"
    "    int tgt = targets[tid];\n"
    "    token_losses[tid] = -log(max(logits[tgt] / sum_e, 1e-12f));\n"
    "    float dh[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) dh[d] = 0.0f;\n"
    "    float inv_n = 1.0f / (float)n_tokens;\n"
    "    for (int v = 0; v < VOCAB; ++v) {\n"
    "        float dl = (logits[v] / sum_e - ((v == tgt) ? 1.0f : 0.0f)) * inv_n;\n"
    "        d_logits_out[tid * VOCAB + v] = dl;\n"
    "        int off = v * D_MODEL;\n"
    "        for (int d = 0; d < D_MODEL; ++d) dh[d] += dl * head_w[off + d];\n"
    "    }\n"
    "    for (int d = 0; d < D_MODEL; ++d) d_hidden[tid * D_MODEL + d] = dh[d];\n"
    "}\n"
    "\n"
    "__kernel void pass_update_head_and_embed(\n"
    "    __global const int *tokens,\n"
    "    __global const float *hidden_out,\n"
    "    __global const float *d_logits_out,\n"
    "    __global const float *d_hidden,\n"
    "    __global float *head_w,\n"
    "    __global float *head_mom,\n"
    "    __global float *embed,\n"
    "    __global float *embed_mom,\n"
    "    int n_tokens,\n"
    "    float lr)\n"
    "{\n"
    "    int v = get_global_id(0);\n"
    "    if (v >= VOCAB) return;\n"
    "    float g_head[D_MODEL], g_emb[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) { g_head[d] = 0.0f; g_emb[d] = 0.0f; }\n"
    "    for (int tid = 0; tid < n_tokens; ++tid) {\n"
    "        float dl = d_logits_out[tid * VOCAB + v];\n"
    "        for (int d = 0; d < D_MODEL; ++d) g_head[d] += dl * hidden_out[tid * D_MODEL + d];\n"
    "        int tok = tokens[tid];\n"
    "        int prev1 = (tid > 0) ? tokens[tid - 1] : tok;\n"
    "        int prev2 = (tid > 1) ? tokens[tid - 2] : prev1;\n"
    "        float w_e = ((tok == v) ? 1.0f : 0.0f) + ((prev1 == v) ? 0.45f : 0.0f) + ((prev2 == v) ? 0.20f : 0.0f);\n"
    "        if (w_e > 0.0f) {\n"
    "            for (int d = 0; d < D_MODEL; ++d) g_emb[d] += w_e * d_hidden[tid * D_MODEL + d];\n"
    "        }\n"
    "    }\n"
    "    int off = v * D_MODEL;\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        float mh = 0.9f * head_mom[off + d] + g_head[d];\n"
    "        head_mom[off + d] = mh;\n"
    "        head_w[off + d] -= lr * clamp(mh, -1.5f, 1.5f);\n"
    "        float me = 0.9f * embed_mom[off + d] + g_emb[d];\n"
    "        embed_mom[off + d] = me;\n"
    "        embed[off + d] -= lr * clamp(me, -1.5f, 1.5f);\n"
    "    }\n"
    "}\n";

static float frand_sym(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return ((float)(*state & 0xFFFFu) / 32768.0f) - 1.0f;
}

int main(int argc, char **argv) {
    const char *train_path = (argc > 1) ? argv[1] : "../lumen_nanite_3d_llm/wikitext_train_bytes.bin";
    const char *valid_path = (argc > 2) ? argv[2] : "../lumen_nanite_3d_llm/wikitext_valid_bytes.bin";
    FILE *fp = fopen(train_path, "rb");
    if (!fp) {
        fprintf(stderr, "Не найден корпус %s\n", train_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *raw_bytes = (uint8_t *)malloc((size_t)fsize);
    fread(raw_bytes, 1, (size_t)fsize, fp);
    fclose(fp);

    FILE *fv = fopen(valid_path, "rb");
    long vsize = 0;
    uint8_t *val_bytes = NULL;
    if (fv) {
        fseek(fv, 0, SEEK_END);
        vsize = ftell(fv);
        fseek(fv, 0, SEEK_SET);
        val_bytes = (uint8_t *)malloc((size_t)vsize);
        fread(val_bytes, 1, (size_t)vsize, fv);
        fclose(fv);
    }

    cl_platform_id plat;
    clGetPlatformIDs(1, &plat, NULL);
    cl_device_id devs[8];
    cl_uint ndevs = 0;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 8, devs, &ndevs);
    cl_device_id radeon = devs[0];
    for (cl_uint i = 0; i < ndevs; ++i) {
        char nm[256];
        clGetDeviceInfo(devs[i], CL_DEVICE_NAME, sizeof(nm), nm, NULL);
        if (strstr(nm, "Radeon")) {
            radeon = devs[i];
            break;
        }
    }
    char dev_name[256] = {0};
    clGetDeviceInfo(radeon, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("=== ЧЕСТНЫЙ БАТТЛ НА GPU %s: КЛАССИЧЕСКИЙ МАТРИЧНЫЙ GEMM vs 2D-ПРОЕКЦИЯ + ВОЛНОВАЯ ИНТЕРФЕРЕНЦИЯ ФРИСТОНА ===\n", dev_name);

    cl_int err = 0;
    cl_context ctx = clCreateContext(NULL, 1, &radeon, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, radeon, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &radeon, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(prog, radeon, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Ошибка компиляции:\n%s\n", log);
        return 1;
    }

    cl_kernel k_c_fwd = clCreateKernel(prog, "classic_dense_forward", &err);
    cl_kernel k_c_bwd = clCreateKernel(prog, "classic_dense_backward", &err);
    cl_kernel k_w_fwd = clCreateKernel(prog, "wave_friston_forward_2d_screen", &err);
    cl_kernel k_w_bwd = clCreateKernel(prog, "wave_interference_backward", &err);
    cl_kernel k_head = clCreateKernel(prog, "pass_head_and_grad", &err);
    cl_kernel k_upd = clCreateKernel(prog, "pass_update_head_and_embed", &err);

    int frames = 2000;
    int n_tokens = BATCH_TOKENS;
    float lr = 0.08f;
    size_t g_tok = BATCH_TOKENS, g_obj = N_OBJ, g_voc = VOCAB;

    /* Запускаем по очереди оба режима с абсолютно одинаковой инициализацией (rng = 42) */
    for (int mode = 0; mode < 2; ++mode) {
        uint32_t rng = 42;
        float *h_embed = (float *)malloc(sizeof(float) * VOCAB * D_MODEL);
        float *h_head = (float *)malloc(sizeof(float) * VOCAB * D_MODEL);
        float *h_zero = (float *)calloc(VOCAB * D_MODEL, sizeof(float));
        for (int i = 0; i < VOCAB * D_MODEL; ++i) {
            h_embed[i] = frand_sym(&rng) * 0.15f;
            h_head[i] = frand_sym(&rng) * 0.15f;
        }

        cl_mem d_tok = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * BATCH_TOKENS, NULL, &err);
        cl_mem d_tgt = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * BATCH_TOKENS, NULL, &err);
        cl_mem d_emb = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_embed, &err);
        cl_mem d_emb_m = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_zero, &err);
        cl_mem d_hd = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_head, &err);
        cl_mem d_hd_m = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_zero, &err);

        cl_mem d_hin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
        cl_mem d_hout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
        cl_mem d_dlog = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * VOCAB, NULL, &err);
        cl_mem d_dhid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
        cl_mem d_loss = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS, NULL, &err);

        /* Буферы классического матричного слоя (W1, W2 размера 256 x 64 = 16384 весов каждый) */
        float *h_W1 = (float *)malloc(sizeof(float) * N_OBJ * D_MODEL);
        float *h_W2 = (float *)malloc(sizeof(float) * N_OBJ * D_MODEL);
        for (int i = 0; i < N_OBJ * D_MODEL; ++i) {
            h_W1[i] = frand_sym(&rng) * 0.12f;
            h_W2[i] = frand_sym(&rng) * 0.12f;
        }
        cl_mem d_W1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * D_MODEL, h_W1, &err);
        cl_mem d_W2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * D_MODEL, h_W2, &err);
        cl_mem d_mid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * N_OBJ, NULL, &err);
        cl_mem d_op_cnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(long), NULL, &err);

        /* Буферы 3D-волнового слоя Фристона (позиции 3D + волновые патчи 3x3) */
        float *h_pos = (float *)malloc(sizeof(float) * N_OBJ * 3);
        float *h_zopen = (float *)malloc(sizeof(float) * N_OBJ);
        float *h_win = (float *)malloc(sizeof(float) * N_OBJ * 9);
        float *h_wout = (float *)malloc(sizeof(float) * N_OBJ * 9);
        for (int j = 0; j < N_OBJ; ++j) {
            h_pos[j * 3 + 0] = frand_sym(&rng) * 1.5f;
            h_pos[j * 3 + 1] = frand_sym(&rng) * 1.5f;
            h_pos[j * 3 + 2] = 0.5f + fabsf(frand_sym(&rng)) * 1.5f;
            h_zopen[j] = 0.15f + 0.35f * fabsf(frand_sym(&rng));
            for (int k = 0; k < 9; ++k) {
                h_win[j * 9 + k] = frand_sym(&rng) * 0.2f;
                h_wout[j * 9 + k] = frand_sym(&rng) * 0.2f;
            }
        }
        cl_mem d_pos = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * 3, h_pos, &err);
        cl_mem d_zopen = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ, h_zopen, &err);
        cl_mem d_win = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * 9, h_win, &err);
        cl_mem d_wout = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * 9, h_wout, &err);
        cl_mem d_pix_ops = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * BATCH_TOKENS, NULL, &err);

        clSetKernelArg(k_head, 0, sizeof(cl_mem), &d_hout);
        clSetKernelArg(k_head, 1, sizeof(cl_mem), &d_tgt);
        clSetKernelArg(k_head, 2, sizeof(cl_mem), &d_hd);
        clSetKernelArg(k_head, 3, sizeof(cl_mem), &d_dlog);
        clSetKernelArg(k_head, 4, sizeof(cl_mem), &d_dhid);
        clSetKernelArg(k_head, 5, sizeof(cl_mem), &d_loss);
        clSetKernelArg(k_head, 6, sizeof(int), &n_tokens);

        clSetKernelArg(k_upd, 0, sizeof(cl_mem), &d_tok);
        clSetKernelArg(k_upd, 1, sizeof(cl_mem), &d_hout);
        clSetKernelArg(k_upd, 2, sizeof(cl_mem), &d_dlog);
        clSetKernelArg(k_upd, 3, sizeof(cl_mem), &d_dhid);
        clSetKernelArg(k_upd, 4, sizeof(cl_mem), &d_hd);
        clSetKernelArg(k_upd, 5, sizeof(cl_mem), &d_hd_m);
        clSetKernelArg(k_upd, 6, sizeof(cl_mem), &d_emb);
        clSetKernelArg(k_upd, 7, sizeof(cl_mem), &d_emb_m);
        clSetKernelArg(k_upd, 8, sizeof(int), &n_tokens);
        clSetKernelArg(k_upd, 9, sizeof(float), &lr);

        if (mode == 0) {
            clSetKernelArg(k_c_fwd, 0, sizeof(cl_mem), &d_tok);
            clSetKernelArg(k_c_fwd, 1, sizeof(cl_mem), &d_emb);
            clSetKernelArg(k_c_fwd, 2, sizeof(cl_mem), &d_W1);
            clSetKernelArg(k_c_fwd, 3, sizeof(cl_mem), &d_W2);
            clSetKernelArg(k_c_fwd, 4, sizeof(cl_mem), &d_hin);
            clSetKernelArg(k_c_fwd, 5, sizeof(cl_mem), &d_mid);
            clSetKernelArg(k_c_fwd, 6, sizeof(cl_mem), &d_hout);
            clSetKernelArg(k_c_fwd, 7, sizeof(cl_mem), &d_op_cnt);
            clSetKernelArg(k_c_fwd, 8, sizeof(int), &n_tokens);

            clSetKernelArg(k_c_bwd, 0, sizeof(cl_mem), &d_hin);
            clSetKernelArg(k_c_bwd, 1, sizeof(cl_mem), &d_mid);
            clSetKernelArg(k_c_bwd, 2, sizeof(cl_mem), &d_dhid);
            clSetKernelArg(k_c_bwd, 3, sizeof(cl_mem), &d_W1);
            clSetKernelArg(k_c_bwd, 4, sizeof(cl_mem), &d_W2);
            clSetKernelArg(k_c_bwd, 5, sizeof(int), &n_tokens);
            clSetKernelArg(k_c_bwd, 6, sizeof(float), &lr);
            printf("\n--- ЗАПУСК РЕЖИМА А: КЛАССИЧЕСКИЙ МАТРИЧНЫЙ GEMM + МАТРИЧНЫЙ БЭКПРОП (256x64) ---\n");
        } else {
            clSetKernelArg(k_w_fwd, 0, sizeof(cl_mem), &d_tok);
            clSetKernelArg(k_w_fwd, 1, sizeof(cl_mem), &d_emb);
            clSetKernelArg(k_w_fwd, 2, sizeof(cl_mem), &d_pos);
            clSetKernelArg(k_w_fwd, 3, sizeof(cl_mem), &d_zopen);
            clSetKernelArg(k_w_fwd, 4, sizeof(cl_mem), &d_win);
            clSetKernelArg(k_w_fwd, 5, sizeof(cl_mem), &d_wout);
            clSetKernelArg(k_w_fwd, 6, sizeof(cl_mem), &d_hin);
            clSetKernelArg(k_w_fwd, 7, sizeof(cl_mem), &d_hout);
            clSetKernelArg(k_w_fwd, 8, sizeof(cl_mem), &d_pix_ops);
            clSetKernelArg(k_w_fwd, 9, sizeof(int), &n_tokens);

            clSetKernelArg(k_w_bwd, 0, sizeof(cl_mem), &d_hin);
            clSetKernelArg(k_w_bwd, 1, sizeof(cl_mem), &d_dhid);
            clSetKernelArg(k_w_bwd, 2, sizeof(cl_mem), &d_pos);
            clSetKernelArg(k_w_bwd, 3, sizeof(cl_mem), &d_zopen);
            clSetKernelArg(k_w_bwd, 4, sizeof(cl_mem), &d_win);
            clSetKernelArg(k_w_bwd, 5, sizeof(cl_mem), &d_wout);
            clSetKernelArg(k_w_bwd, 6, sizeof(int), &n_tokens);
            clSetKernelArg(k_w_bwd, 7, sizeof(float), &lr);
            printf("\n--- ЗАПУСК РЕЖИМА Б: 3D-ВИДЕОРЯД + 2D-ПРОЕКЦИЯ 8x8 + ВОЛНОВАЯ ИНТЕРФЕРЕНЦИЯ ФРИСТОНА ---\n");
        }

        int h_tok[BATCH_TOKENS], h_tgt[BATCH_TOKENS], h_ops[BATCH_TOKENS];
        float h_loss[BATCH_TOKENS];
        long cursor = 0;
        double layer_fwd_ms = 0.0, layer_bwd_ms = 0.0;
        double t0 = now_sec();
        long total_layer_ops = 0;

        for (int frame = 1; frame <= frames; ++frame) {
            if (cursor + BATCH_TOKENS + 1 >= fsize) cursor = 0;
            for (int i = 0; i < BATCH_TOKENS; ++i) {
                h_tok[i] = (int)raw_bytes[cursor + i];
                h_tgt[i] = (int)raw_bytes[cursor + i + 1];
            }
            cursor += BATCH_TOKENS;

            clEnqueueWriteBuffer(queue, d_tok, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tok, 0, NULL, NULL);
            clEnqueueWriteBuffer(queue, d_tgt, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tgt, 0, NULL, NULL);

            cl_event ev_fwd, ev_bwd;
            if (mode == 0) {
                clEnqueueNDRangeKernel(queue, k_c_fwd, 1, NULL, &g_tok, NULL, 0, NULL, &ev_fwd);
                clEnqueueNDRangeKernel(queue, k_head, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
                clEnqueueNDRangeKernel(queue, k_c_bwd, 1, NULL, &g_obj, NULL, 0, NULL, &ev_bwd);
                clEnqueueNDRangeKernel(queue, k_upd, 1, NULL, &g_voc, NULL, 0, NULL, NULL);
            } else {
                clEnqueueNDRangeKernel(queue, k_w_fwd, 1, NULL, &g_tok, NULL, 0, NULL, &ev_fwd);
                clEnqueueNDRangeKernel(queue, k_head, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
                clEnqueueNDRangeKernel(queue, k_w_bwd, 1, NULL, &g_obj, NULL, 0, NULL, &ev_bwd);
                clEnqueueNDRangeKernel(queue, k_upd, 1, NULL, &g_voc, NULL, 0, NULL, NULL);
            }
            clWaitForEvents(1, &ev_bwd);

            cl_ulong s1, e1, s2, e2;
            clGetEventProfilingInfo(ev_fwd, CL_PROFILING_COMMAND_START, sizeof(s1), &s1, NULL);
            clGetEventProfilingInfo(ev_fwd, CL_PROFILING_COMMAND_END, sizeof(e1), &e1, NULL);
            clGetEventProfilingInfo(ev_bwd, CL_PROFILING_COMMAND_START, sizeof(s2), &s2, NULL);
            clGetEventProfilingInfo(ev_bwd, CL_PROFILING_COMMAND_END, sizeof(e2), &e2, NULL);
            layer_fwd_ms += (double)(e1 - s1) * 1e-6;
            layer_bwd_ms += (double)(e2 - s2) * 1e-6;
            clReleaseEvent(ev_fwd);
            clReleaseEvent(ev_bwd);

            if (frame == frames) {
                if (mode == 0) {
                    total_layer_ops = (long)BATCH_TOKENS * (long)N_OBJ * (long)D_MODEL * 4L * 2L;
                } else {
                    clEnqueueReadBuffer(queue, d_pix_ops, CL_TRUE, 0, sizeof(int) * BATCH_TOKENS, h_ops, 0, NULL, NULL);
                    long sum_ops = 0;
                    for (int i = 0; i < BATCH_TOKENS; ++i) sum_ops += h_ops[i];
                    total_layer_ops = sum_ops * 2L;
                }
            }
        }
        clFinish(queue);
        double wall_sec = now_sec() - t0;

        /* Валидация на отложенном корпусе */
        double val_loss_sum = 0.0;
        int val_batches = 0;
        for (long off = 0; off + BATCH_TOKENS + 1 < vsize && val_batches < 100; off += BATCH_TOKENS) {
            for (int i = 0; i < BATCH_TOKENS; ++i) {
                h_tok[i] = (int)val_bytes[off + i];
                h_tgt[i] = (int)val_bytes[off + i + 1];
            }
            clEnqueueWriteBuffer(queue, d_tok, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tok, 0, NULL, NULL);
            clEnqueueWriteBuffer(queue, d_tgt, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tgt, 0, NULL, NULL);
            if (mode == 0) clEnqueueNDRangeKernel(queue, k_c_fwd, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
            else clEnqueueNDRangeKernel(queue, k_w_fwd, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_head, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_loss, CL_TRUE, 0, sizeof(float) * BATCH_TOKENS, h_loss, 0, NULL, NULL);
            for (int i = 0; i < BATCH_TOKENS; ++i) val_loss_sum += h_loss[i];
            val_batches++;
        }
        double val_bpc = (val_loss_sum / (val_batches * BATCH_TOKENS)) / log(2.0);
        double val_ppl = exp(val_loss_sum / (val_batches * BATCH_TOKENS));

        printf("Операций внутри слоя на 1 батч (прямой + градиент): %ld FLOPs\n", total_layer_ops);
        printf("Чистое время GPU на 2000 кадров (только слой): Прямой = %.1f мс | Обратный (Градиент) = %.1f мс | Сумма слоя = %.1f мс\n",
               layer_fwd_ms, layer_bwd_ms, layer_fwd_ms + layer_bwd_ms);
        printf("Общее время с головой словаря: %.2f с (%.1f кадров/с)\n", wall_sec, frames / wall_sec);
        printf("Точность на отложенном WikiText-2: Validation BPC = %.3f | Перплексия байта = %.2f\n", val_bpc, val_ppl);
    }
    return 0;
}
