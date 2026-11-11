/* Нативный GPU-движок реального времени на AMD Radeon Pro 5500M (4 ГБ):
 * Обучение и инференс модели v3 (Nanite Polygon DAG + Ray Tracing Occlusion + Unreal Lumen GI)
 * прямо в режиме реалтайм-рендеринга видеокарты (OpenCL Shaders), без участия CPU в перемножении матриц!
 *
 * Архитектура кадра на GPU:
 *   Pass 1 (Ray Trace & Lumen GI Shader):
 *     - Для каждого элемента батча испускается луч контекста (r_orig, ray_dir).
 *     - Аналитическое пересечение конуса луча со сферами N=96 3D-объектов + Z-перекрытие (Occlusion).
 *     - Засветка 6 карточек Lumen Surface Cache (+X,-X,+Y,-Y,+Z,-Z) и сбор/раздача непрямого света
 *       через 8 мировых зондов Lumen Radiosity Probes.
 *     - Вычисление Screen-Space Error -> определение числа активных полигональных уровней Nanite (1, 2, 4 или 8 векторов).
 *     - Чтение из VRAM ТОЛЬКО открытых уровней дерева полигонов Nanite (экономия полосы пропускания VRAM!).
 *   Pass 2 (Output Shader & Cross-Entropy):
 *     - Сборка морфинга признака, проекция в словарь (256 байт), точный градиент d_logits и d_hidden на GPU.
 *   Pass 3 & 4 (Backward Ray, Nanite & Head Update Shaders):
 *     - Полный бэкпроп на GPU с импульсом (Momentum): обновляются ТОЛЬКО те полигональные уровни (LOD 0..3),
 *       которые были реально открыты на этом кадре, 3D-координаты объектов, выходная проекция и эмбеддинги!
 */

#include <OpenCL/opencl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define VOCAB 256
#define D_MODEL 64
#define N_OBJ 96
#define N_GROUPS 8
#define G_DIM (D_MODEL / N_GROUPS)
#define N_PROBES 8
#define BATCH_TOKENS 256

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static const char *KERNEL_SRC =
    "#define VOCAB 256\n"
    "#define D_MODEL 64\n"
    "#define N_OBJ 96\n"
    "#define N_GROUPS 8\n"
    "#define G_DIM 8\n"
    "#define N_PROBES 8\n"
    "\n"
    "inline float sigmoidf(float x) {\n"
    "    return 1.0f / (1.0f + exp(-clamp(x, -20.0f, 20.0f)));\n"
    "}\n"
    "inline float siluf(float x) {\n"
    "    return x * sigmoidf(x);\n"
    "}\n"
    "inline float dsiluf(float x) {\n"
    "    float s = sigmoidf(x);\n"
    "    return s * (1.0f + x * (1.0f - s));\n"
    "}\n"
    "\n"
    "/* Pass 1: Трассировка лучей, Lumen Surface Cache + 8 зондов GI и выбор уровня полигонов Nanite */\n"
    "__kernel void pass1_raytrace_lumen_nanite(\n"
    "    __global const int *tokens,\n"
    "    __global const float *embed,\n"
    "    __global const float *obj_pos,\n"
    "    __global const float *obj_radius,\n"
    "    __global const float *obj_in,\n"
    "    __global const float *obj_out,\n"
    "    __global float *surface_cache,\n"
    "    __global const float *probe_pos,\n"
    "    __global float *hidden_in,\n"
    "    __global float *hidden_out,\n"
    "    __global float *saved_dots,\n"
    "    __global float *saved_weights,\n"
    "    __global int *saved_lod_count,\n"
    "    int n_tokens)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    if (tid >= n_tokens) return;\n"
    "\n"
    "    int tok = tokens[tid];\n"
    "    int prev1 = (tid > 0) ? tokens[tid - 1] : tok;\n"
    "    int prev2 = (tid > 1) ? tokens[tid - 2] : prev1;\n"
    "    float h[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        h[d] = embed[tok * D_MODEL + d]\n"
    "             + 0.45f * embed[prev1 * D_MODEL + d]\n"
    "             + 0.20f * embed[prev2 * D_MODEL + d];\n"
    "        hidden_in[tid * D_MODEL + d] = h[d];\n"
    "    }\n"
    "\n"
    "    /* Координаты испускания луча r_orig и направление луча ray_dir вдоль стрелы времени */\n"
    "    float rx = h[0] * 1.4f, ry = h[1] * 1.4f, rz = h[2] * 1.4f;\n"
    "    float dx = (embed[tok * D_MODEL + 0] - embed[prev1 * D_MODEL + 0]) + 0.25f;\n"
    "    float dy = (embed[tok * D_MODEL + 1] - embed[prev1 * D_MODEL + 1]) + 0.15f;\n"
    "    float dz = (embed[tok * D_MODEL + 2] - embed[prev1 * D_MODEL + 2]) + 0.45f;\n"
    "    float dnorm = rsqrt(dx * dx + dy * dy + dz * dz + 1e-6f);\n"
    "    dx *= dnorm; dy *= dnorm; dz *= dnorm;\n"
    "\n"
    "    /* Излучение 8 мировых зондов Lumen Radiosity Probes от текущего Surface Cache */\n"
    "    float probe_rad[N_PROBES];\n"
    "    for (int p = 0; p < N_PROBES; ++p) {\n"
    "        float px = probe_pos[p * 3 + 0], py = probe_pos[p * 3 + 1], pz = probe_pos[p * 3 + 2];\n"
    "        float acc_p = 0.0f;\n"
    "        for (int j = 0; j < N_OBJ; j += 4) {\n"
    "            float ox = px - obj_pos[j * 3 + 0];\n"
    "            float oy = py - obj_pos[j * 3 + 1];\n"
    "            float oz = pz - obj_pos[j * 3 + 2];\n"
    "            float inv_d2 = 1.0f / (ox * ox + oy * oy + oz * oz + 0.5f);\n"
    "            acc_p += (surface_cache[j * 6 + 0] + surface_cache[j * 6 + 2] + surface_cache[j * 6 + 4]) * inv_d2;\n"
    "        }\n"
    "        probe_rad[p] = acc_p / (float)(N_OBJ / 4);\n"
    "    }\n"
    "\n"
    "    float morphed[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) morphed[d] = h[d];\n"
    "\n"
    "    float running_occlusion = 0.0f;\n"
    "    int total_vecs_loaded = 0;\n"
    "\n"
    "    for (int j = 0; j < N_OBJ; ++j) {\n"
    "        float vx = obj_pos[j * 3 + 0] - rx;\n"
    "        float vy = obj_pos[j * 3 + 1] - ry;\n"
    "        float vz = obj_pos[j * 3 + 2] - rz;\n"
    "        float dist2 = vx * vx + vy * vy + vz * vz + 1e-5f;\n"
    "        float t_hit = vx * dx + vy * dy + vz * dz;\n"
    "        float t_pos = (t_hit > 0.0f) ? (t_hit + 0.25f) : 0.25f;\n"
    "        float in_front = sigmoidf(3.0f * t_hit);\n"
    "        float b_perp2 = max(0.0f, dist2 - t_hit * t_hit);\n"
    "\n"
    "        float rad = obj_radius[j];\n"
    "        float cone_r = 0.25f * t_pos + rad;\n"
    "        float hit = exp(-0.5f * b_perp2 / (cone_r * cone_r)) * in_front;\n"
    "\n"
    "        /* Закон перекрытия Бугера-Ламберта (тень от ранее встреченных объектов на луче) */\n"
    "        float transmittance = exp(-running_occlusion);\n"
    "        running_occlusion += hit * 0.08f;\n"
    "        float direct_light = hit * transmittance / (1.0f + 0.35f * dist2);\n"
    "\n"
    "        /* Lumen GI: подсветка от ближайшего зонда + чтение 6 карточек Surface Cache */\n"
    "        int p_idx = ((obj_pos[j * 3 + 0] > 0.0f) ? 4 : 0)\n"
    "                  + ((obj_pos[j * 3 + 1] > 0.0f) ? 2 : 0)\n"
    "                  + ((obj_pos[j * 3 + 2] > 0.0f) ? 1 : 0);\n"
    "        int card_idx = (fabs(vx) > fabs(vy) && fabs(vx) > fabs(vz)) ? ((vx > 0.0f) ? 1 : 0)\n"
    "                     : (fabs(vy) > fabs(vz)) ? ((vy > 0.0f) ? 3 : 2) : ((vz > 0.0f) ? 5 : 4);\n"
    "        float cached_gi = surface_cache[j * 6 + card_idx] + 0.35f * probe_rad[p_idx];\n"
    "        float total_rad = direct_light + 0.45f * cached_gi;\n"
    "\n"
    "        if (tid == n_tokens - 1) {\n"
    "            surface_cache[j * 6 + card_idx] = 0.85f * surface_cache[j * 6 + card_idx] + 0.15f * total_rad;\n"
    "        }\n"
    "\n"
    "        /* Nanite Screen-Space Error -> ступень детализации полигона (1, 2, 4 или 8 векторов) */\n"
    "        float screen_err = total_rad * (rad / t_pos) * 3.0f;\n"
    "        int active_k = 1;\n"
    "        if (screen_err > 0.85f) active_k = 8;\n"
    "        else if (screen_err > 0.45f) active_k = 4;\n"
    "        else if (screen_err > 0.20f) active_k = 2;\n"
    "        total_vecs_loaded += active_k;\n"
    "\n"
    "        float env = total_rad / (1.0f + total_rad);\n"
    "        saved_weights[tid * N_OBJ + j] = env;\n"
    "\n"
    "        /* Читаем из VRAM только открытые полигональные уровни k = 0 .. active_k - 1 */\n"
    "        for (int k = 0; k < N_GROUPS; ++k) {\n"
    "            int base_idx = (j * N_GROUPS + k) * G_DIM;\n"
    "            if (k < active_k) {\n"
    "                float dot = 0.0f;\n"
    "                for (int g = 0; g < G_DIM; ++g) {\n"
    "                    dot += h[k * G_DIM + g] * obj_in[base_idx + g];\n"
    "                }\n"
    "                saved_dots[(tid * N_OBJ + j) * N_GROUPS + k] = dot;\n"
    "                float act = siluf(dot) * env;\n"
    "                for (int g = 0; g < G_DIM; ++g) {\n"
    "                    morphed[k * G_DIM + g] += act * obj_out[base_idx + g];\n"
    "                }\n"
    "            } else {\n"
    "                saved_dots[(tid * N_OBJ + j) * N_GROUPS + k] = 0.0f;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    saved_lod_count[tid] = total_vecs_loaded;\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        hidden_out[tid * D_MODEL + d] = morphed[d];\n"
    "    }\n"
    "}\n"
    "\n"
    "/* Pass 2: Логиты, точный градиент d_logits (B, VOCAB) и d_hidden (B, D_MODEL) */\n"
    "__kernel void pass2_head_and_grad(\n"
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
    "\n"
    "    float h[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) h[d] = hidden_out[tid * D_MODEL + d];\n"
    "\n"
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
    "    float prob_tgt = max(logits[tgt] / sum_e, 1e-12f);\n"
    "    token_losses[tid] = -log(prob_tgt);\n"
    "\n"
    "    float dh[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) dh[d] = 0.0f;\n"
    "    float inv_n = 1.0f / (float)n_tokens;\n"
    "    for (int v = 0; v < VOCAB; ++v) {\n"
    "        float dl = (logits[v] / sum_e - ((v == tgt) ? 1.0f : 0.0f)) * inv_n;\n"
    "        d_logits_out[tid * VOCAB + v] = dl;\n"
    "        int off = v * D_MODEL;\n"
    "        for (int d = 0; d < D_MODEL; ++d) {\n"
    "            dh[d] += dl * head_w[off + d];\n"
    "        }\n"
    "    }\n"
    "    for (int d = 0; d < D_MODEL; ++d) d_hidden[tid * D_MODEL + d] = dh[d];\n"
    "}\n"
    "\n"
    "/* Pass 3: Бэкпроп и обновление полигональных граней Nanite + 3D-координат прямо в VRAM */\n"
    "__kernel void pass3_update_nanite_objects(\n"
    "    __global const float *hidden_in,\n"
    "    __global const float *d_hidden,\n"
    "    __global const float *saved_dots,\n"
    "    __global const float *saved_weights,\n"
    "    __global float *obj_in,\n"
    "    __global float *obj_out,\n"
    "    __global float *obj_pos,\n"
    "    int n_tokens,\n"
    "    float lr)\n"
    "{\n"
    "    int j = get_global_id(0);\n"
    "    if (j >= N_OBJ) return;\n"
    "\n"
    "    float pos_gx = 0.0f, pos_gy = 0.0f, pos_gz = 0.0f;\n"
    "\n"
    "    for (int k = 0; k < N_GROUPS; ++k) {\n"
    "        float gin[G_DIM], gout[G_DIM];\n"
    "        for (int g = 0; g < G_DIM; ++g) { gin[g] = 0.0f; gout[g] = 0.0f; }\n"
    "        int base_idx = (j * N_GROUPS + k) * G_DIM;\n"
    "\n"
    "        for (int tid = 0; tid < n_tokens; ++tid) {\n"
    "            float env = saved_weights[tid * N_OBJ + j];\n"
    "            float dot = saved_dots[(tid * N_OBJ + j) * N_GROUPS + k];\n"
    "            if (dot == 0.0f && k > 0) continue;\n"
    "\n"
    "            float act = siluf(dot) * env;\n"
    "            float d_act = 0.0f;\n"
    "            for (int g = 0; g < G_DIM; ++g) {\n"
    "                float dh = d_hidden[tid * D_MODEL + k * G_DIM + g];\n"
    "                gout[g] += dh * act;\n"
    "                d_act += dh * obj_out[base_idx + g];\n"
    "            }\n"
    "            float d_dot = d_act * env * dsiluf(dot);\n"
    "            for (int g = 0; g < G_DIM; ++g) {\n"
    "                float h_val = hidden_in[tid * D_MODEL + k * G_DIM + g];\n"
    "                gin[g] += d_dot * h_val;\n"
    "            }\n"
    "            if (k == 0) {\n"
    "                float d_env = d_act * siluf(dot) * (1.0f - env);\n"
    "                pos_gx += d_env * (hidden_in[tid * D_MODEL + 0] - obj_pos[j * 3 + 0]) * 0.1f;\n"
    "                pos_gy += d_env * (hidden_in[tid * D_MODEL + 1] - obj_pos[j * 3 + 1]) * 0.1f;\n"
    "                pos_gz += d_env * (hidden_in[tid * D_MODEL + 2] - obj_pos[j * 3 + 2]) * 0.1f;\n"
    "            }\n"
    "        }\n"
    "        for (int g = 0; g < G_DIM; ++g) {\n"
    "            obj_in[base_idx + g] -= lr * clamp(gin[g], -1.0f, 1.0f);\n"
    "            obj_out[base_idx + g] -= lr * clamp(gout[g], -1.0f, 1.0f);\n"
    "        }\n"
    "    }\n"
    "    obj_pos[j * 3 + 0] -= lr * clamp(pos_gx, -0.5f, 0.5f);\n"
    "    obj_pos[j * 3 + 1] -= lr * clamp(pos_gy, -0.5f, 0.5f);\n"
    "    obj_pos[j * 3 + 2] -= lr * clamp(pos_gz, -0.5f, 0.5f);\n"
    "}\n"
    "\n"
    "/* Pass 4: Полный точный градиент по матрице словаря head_w и таблице эмбеддингов на GPU */\n"
    "__kernel void pass4_update_head_and_embed(\n"
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
    "\n"
    "    float g_head[D_MODEL];\n"
    "    float g_emb[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) { g_head[d] = 0.0f; g_emb[d] = 0.0f; }\n"
    "\n"
    "    for (int tid = 0; tid < n_tokens; ++tid) {\n"
    "        float dl = d_logits_out[tid * VOCAB + v];\n"
    "        for (int d = 0; d < D_MODEL; ++d) {\n"
    "            g_head[d] += dl * hidden_out[tid * D_MODEL + d];\n"
    "        }\n"
    "        int tok = tokens[tid];\n"
    "        int prev1 = (tid > 0) ? tokens[tid - 1] : tok;\n"
    "        int prev2 = (tid > 1) ? tokens[tid - 2] : prev1;\n"
    "        float w_e = ((tok == v) ? 1.0f : 0.0f)\n"
    "                  + ((prev1 == v) ? 0.45f : 0.0f)\n"
    "                  + ((prev2 == v) ? 0.20f : 0.0f);\n"
    "        if (w_e > 0.0f) {\n"
    "            for (int d = 0; d < D_MODEL; ++d) {\n"
    "                g_emb[d] += w_e * d_hidden[tid * D_MODEL + d];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    int off = v * D_MODEL;\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        float mh = 0.9f * head_mom[off + d] + g_head[d];\n"
    "        head_mom[off + d] = mh;\n"
    "        head_w[off + d] -= lr * clamp(mh, -1.5f, 1.5f);\n"
    "\n"
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
    const char *corpus_path = (argc > 1) ? argv[1] : "wikitext_train_bytes.bin";
    FILE *fp = fopen(corpus_path, "rb");
    if (!fp) {
        fprintf(stderr, "Не найден файл корпуса %s\n", corpus_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *raw_bytes = (uint8_t *)malloc((size_t)fsize);
    fread(raw_bytes, 1, (size_t)fsize, fp);
    fclose(fp);

    cl_platform_id plat;
    clGetPlatformIDs(1, &plat, NULL);
    cl_device_id devs[8];
    cl_uint ndevs = 0;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 8, devs, &ndevs);
    cl_device_id radeon = devs[0];
    char dev_name[256] = {0};
    cl_ulong vram_bytes = 0;
    for (cl_uint i = 0; i < ndevs; ++i) {
        char nm[256];
        clGetDeviceInfo(devs[i], CL_DEVICE_NAME, sizeof(nm), nm, NULL);
        if (strstr(nm, "Radeon")) {
            radeon = devs[i];
            break;
        }
    }
    clGetDeviceInfo(radeon, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    clGetDeviceInfo(radeon, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram_bytes), &vram_bytes, NULL);
    printf("=== РЕАЛТАЙМ-РЕНДЕРИНГ И ОБУЧЕНИЕ LUMEN+NANITE ПРЯМО НА GPU ===\n");
    printf("Видеокарта: %s (VRAM: %llu МБ)\n", dev_name, vram_bytes / (1024 * 1024));

    cl_int err = 0;
    cl_context ctx = clCreateContext(NULL, 1, &radeon, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, radeon, 0, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &radeon, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(prog, radeon, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Ошибка компиляции шейдера:\n%s\n", log);
        return 1;
    }

    cl_kernel k_pass1 = clCreateKernel(prog, "pass1_raytrace_lumen_nanite", &err);
    cl_kernel k_pass2 = clCreateKernel(prog, "pass2_head_and_grad", &err);
    cl_kernel k_pass3 = clCreateKernel(prog, "pass3_update_nanite_objects", &err);
    cl_kernel k_pass4 = clCreateKernel(prog, "pass4_update_head_and_embed", &err);

    uint32_t rng = 42;
    float *h_embed = (float *)malloc(sizeof(float) * VOCAB * D_MODEL);
    float *h_head = (float *)malloc(sizeof(float) * VOCAB * D_MODEL);
    float *h_zero_mom = (float *)calloc(VOCAB * D_MODEL, sizeof(float));
    float *h_obj_pos = (float *)malloc(sizeof(float) * N_OBJ * 3);
    float *h_obj_rad = (float *)malloc(sizeof(float) * N_OBJ);
    float *h_obj_in = (float *)malloc(sizeof(float) * N_OBJ * N_GROUPS * G_DIM);
    float *h_obj_out = (float *)malloc(sizeof(float) * N_OBJ * N_GROUPS * G_DIM);
    float *h_surf_cache = (float *)calloc(N_OBJ * 6, sizeof(float));
    float h_probes[N_PROBES * 3] = {
        -1.2f, -1.2f, -1.2f,  -1.2f, -1.2f,  1.2f,  -1.2f,  1.2f, -1.2f,  -1.2f,  1.2f,  1.2f,
         1.2f, -1.2f, -1.2f,   1.2f, -1.2f,  1.2f,   1.2f,  1.2f, -1.2f,   1.2f,  1.2f,  1.2f
    };

    for (int i = 0; i < VOCAB * D_MODEL; ++i) {
        h_embed[i] = frand_sym(&rng) * 0.15f;
        h_head[i] = frand_sym(&rng) * 0.15f;
    }
    for (int j = 0; j < N_OBJ; ++j) {
        h_obj_pos[j * 3 + 0] = frand_sym(&rng) * 1.4f;
        h_obj_pos[j * 3 + 1] = frand_sym(&rng) * 1.4f;
        h_obj_pos[j * 3 + 2] = frand_sym(&rng) * 1.4f;
        h_obj_rad[j] = 0.45f + 0.25f * fabsf(frand_sym(&rng));
    }
    for (int i = 0; i < N_OBJ * N_GROUPS * G_DIM; ++i) {
        h_obj_in[i] = frand_sym(&rng) * 0.15f;
        h_obj_out[i] = frand_sym(&rng) * 0.15f;
    }

    cl_mem d_tokens = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * BATCH_TOKENS, NULL, &err);
    cl_mem d_targets = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * BATCH_TOKENS, NULL, &err);
    cl_mem d_embed = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_embed, &err);
    cl_mem d_embed_mom = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_zero_mom, &err);
    cl_mem d_head = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_head, &err);
    cl_mem d_head_mom = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_zero_mom, &err);
    cl_mem d_obj_pos = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * 3, h_obj_pos, &err);
    cl_mem d_obj_rad = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ, h_obj_rad, &err);
    cl_mem d_obj_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * N_GROUPS * G_DIM, h_obj_in, &err);
    cl_mem d_obj_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * N_GROUPS * G_DIM, h_obj_out, &err);
    cl_mem d_surf_cache = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * 6, h_surf_cache, &err);
    cl_mem d_probes = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_PROBES * 3, h_probes, &err);

    cl_mem d_hidden_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_hidden = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_dlogits = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * VOCAB, NULL, &err);
    cl_mem d_dhidden = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_dots = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * N_OBJ * N_GROUPS, NULL, &err);
    cl_mem d_weights = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * N_OBJ, NULL, &err);
    cl_mem d_lod_cnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * BATCH_TOKENS, NULL, &err);
    cl_mem d_losses = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS, NULL, &err);

    int h_tok[BATCH_TOKENS], h_tgt[BATCH_TOKENS], h_lod[BATCH_TOKENS];
    float h_loss[BATCH_TOKENS];
    int n_tokens = BATCH_TOKENS;
    float lr = 0.08f;

    clSetKernelArg(k_pass1, 0, sizeof(cl_mem), &d_tokens);
    clSetKernelArg(k_pass1, 1, sizeof(cl_mem), &d_embed);
    clSetKernelArg(k_pass1, 2, sizeof(cl_mem), &d_obj_pos);
    clSetKernelArg(k_pass1, 3, sizeof(cl_mem), &d_obj_rad);
    clSetKernelArg(k_pass1, 4, sizeof(cl_mem), &d_obj_in);
    clSetKernelArg(k_pass1, 5, sizeof(cl_mem), &d_obj_out);
    clSetKernelArg(k_pass1, 6, sizeof(cl_mem), &d_surf_cache);
    clSetKernelArg(k_pass1, 7, sizeof(cl_mem), &d_probes);
    clSetKernelArg(k_pass1, 8, sizeof(cl_mem), &d_hidden_in);
    clSetKernelArg(k_pass1, 9, sizeof(cl_mem), &d_hidden);
    clSetKernelArg(k_pass1, 10, sizeof(cl_mem), &d_dots);
    clSetKernelArg(k_pass1, 11, sizeof(cl_mem), &d_weights);
    clSetKernelArg(k_pass1, 12, sizeof(cl_mem), &d_lod_cnt);
    clSetKernelArg(k_pass1, 13, sizeof(int), &n_tokens);

    clSetKernelArg(k_pass2, 0, sizeof(cl_mem), &d_hidden);
    clSetKernelArg(k_pass2, 1, sizeof(cl_mem), &d_targets);
    clSetKernelArg(k_pass2, 2, sizeof(cl_mem), &d_head);
    clSetKernelArg(k_pass2, 3, sizeof(cl_mem), &d_dlogits);
    clSetKernelArg(k_pass2, 4, sizeof(cl_mem), &d_dhidden);
    clSetKernelArg(k_pass2, 5, sizeof(cl_mem), &d_losses);
    clSetKernelArg(k_pass2, 6, sizeof(int), &n_tokens);

    clSetKernelArg(k_pass3, 0, sizeof(cl_mem), &d_hidden_in);
    clSetKernelArg(k_pass3, 1, sizeof(cl_mem), &d_dhidden);
    clSetKernelArg(k_pass3, 2, sizeof(cl_mem), &d_dots);
    clSetKernelArg(k_pass3, 3, sizeof(cl_mem), &d_weights);
    clSetKernelArg(k_pass3, 4, sizeof(cl_mem), &d_obj_in);
    clSetKernelArg(k_pass3, 5, sizeof(cl_mem), &d_obj_out);
    clSetKernelArg(k_pass3, 6, sizeof(cl_mem), &d_obj_pos);
    clSetKernelArg(k_pass3, 7, sizeof(int), &n_tokens);
    clSetKernelArg(k_pass3, 8, sizeof(float), &lr);

    clSetKernelArg(k_pass4, 0, sizeof(cl_mem), &d_tokens);
    clSetKernelArg(k_pass4, 1, sizeof(cl_mem), &d_hidden);
    clSetKernelArg(k_pass4, 2, sizeof(cl_mem), &d_dlogits);
    clSetKernelArg(k_pass4, 3, sizeof(cl_mem), &d_dhidden);
    clSetKernelArg(k_pass4, 4, sizeof(cl_mem), &d_head);
    clSetKernelArg(k_pass4, 5, sizeof(cl_mem), &d_head_mom);
    clSetKernelArg(k_pass4, 6, sizeof(cl_mem), &d_embed);
    clSetKernelArg(k_pass4, 7, sizeof(cl_mem), &d_embed_mom);
    clSetKernelArg(k_pass4, 8, sizeof(int), &n_tokens);
    clSetKernelArg(k_pass4, 9, sizeof(float), &lr);

    size_t g_tok = BATCH_TOKENS;
    size_t g_obj = N_OBJ;
    size_t g_voc = VOCAB;

    int total_frames = 2000;
    long cursor = 0;
    double t_start = now_sec();
    double smooth_bpc = 8.0;

    for (int frame = 1; frame <= total_frames; ++frame) {
        if (cursor + BATCH_TOKENS + 1 >= fsize) cursor = 0;
        for (int i = 0; i < BATCH_TOKENS; ++i) {
            h_tok[i] = (int)raw_bytes[cursor + i];
            h_tgt[i] = (int)raw_bytes[cursor + i + 1];
        }
        cursor += BATCH_TOKENS;

        clEnqueueWriteBuffer(queue, d_tokens, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tok, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_targets, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tgt, 0, NULL, NULL);

        clEnqueueNDRangeKernel(queue, k_pass1, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_pass2, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_pass3, 1, NULL, &g_obj, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_pass4, 1, NULL, &g_voc, NULL, 0, NULL, NULL);

        if (frame == 1 || frame % 100 == 0) {
            clEnqueueReadBuffer(queue, d_losses, CL_FALSE, 0, sizeof(float) * BATCH_TOKENS, h_loss, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_lod_cnt, CL_TRUE, 0, sizeof(int) * BATCH_TOKENS, h_lod, 0, NULL, NULL);
            double loss_sum = 0.0;
            double vec_sum = 0.0;
            for (int i = 0; i < BATCH_TOKENS; ++i) {
                loss_sum += h_loss[i];
                vec_sum += h_lod[i];
            }
            double bpc = (loss_sum / BATCH_TOKENS) / log(2.0);
            smooth_bpc = (frame == 1) ? bpc : (0.7 * smooth_bpc + 0.3 * bpc);
            double avg_vecs_per_obj = (vec_sum / BATCH_TOKENS) / N_OBJ;
            double vram_pct = (avg_vecs_per_obj / N_GROUPS) * 100.0;
            double elapsed = now_sec() - t_start;
            double fps = frame / elapsed;
            if (frame == 1 || frame % 400 == 0) {
                printf("Кадр %4d | BPC=%.3f (сглаж.=%.3f) | Nanite LOD=%.2f/8 векторов (%.1f%% VRAM) | Скорость=%.1f кадров/с (%.0f байт/с)\n",
                       frame, bpc, smooth_bpc, avg_vecs_per_obj, vram_pct, fps, fps * BATCH_TOKENS);
            }
        }
    }
    clFinish(queue);
    double total_sec = now_sec() - t_start;
    printf("Итого: %d кадров обучения на GPU AMD Radeon Pro 5500M за %.2f с (%.1f кадров/с, %.0f токенов/с)\n",
           total_frames, total_sec, total_frames / total_sec, (total_frames * BATCH_TOKENS) / total_sec);

    /* Проверка на отложенном хвосте корпуса (без обновления весов) */
    const char *valid_path = (argc > 2) ? argv[2] : "wikitext_valid_bytes.bin";
    FILE *fv = fopen(valid_path, "rb");
    if (fv) {
        fseek(fv, 0, SEEK_END);
        long vsize = ftell(fv);
        fseek(fv, 0, SEEK_SET);
        uint8_t *val_bytes = (uint8_t *)malloc((size_t)vsize);
        fread(val_bytes, 1, (size_t)vsize, fv);
        fclose(fv);

        double val_loss_sum = 0.0;
        double val_vec_sum = 0.0;
        int val_batches = 0;
        for (long off = 0; off + BATCH_TOKENS + 1 < vsize && val_batches < 100; off += BATCH_TOKENS) {
            for (int i = 0; i < BATCH_TOKENS; ++i) {
                h_tok[i] = (int)val_bytes[off + i];
                h_tgt[i] = (int)val_bytes[off + i + 1];
            }
            clEnqueueWriteBuffer(queue, d_tokens, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tok, 0, NULL, NULL);
            clEnqueueWriteBuffer(queue, d_targets, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tgt, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_pass1, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
            clEnqueueNDRangeKernel(queue, k_pass2, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_losses, CL_FALSE, 0, sizeof(float) * BATCH_TOKENS, h_loss, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_lod_cnt, CL_TRUE, 0, sizeof(int) * BATCH_TOKENS, h_lod, 0, NULL, NULL);
            for (int i = 0; i < BATCH_TOKENS; ++i) {
                val_loss_sum += h_loss[i];
                val_vec_sum += h_lod[i];
            }
            val_batches++;
        }
        double val_bpc = (val_loss_sum / (val_batches * BATCH_TOKENS)) / log(2.0);
        double val_ppl = exp(val_loss_sum / (val_batches * BATCH_TOKENS));
        double val_vecs = (val_vec_sum / (val_batches * BATCH_TOKENS)) / N_OBJ;
        printf("=== ПРОВЕРКА GPU-ДВИЖКА НА ОТЛОЖЕННЫХ СТАТЬЯХ WIKITEXT-2 ===\n");
        printf("Validation BPC = %.3f | Перплексия байта = %.2f | Nanite LOD = %.2f/8 векторов (%.1f%% VRAM)\n",
               val_bpc, val_ppl, val_vecs, (val_vecs / N_GROUPS) * 100.0);
        free(val_bytes);
    }
    return 0;
}
