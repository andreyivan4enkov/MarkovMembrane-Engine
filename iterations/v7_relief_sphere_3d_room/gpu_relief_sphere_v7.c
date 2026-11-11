/* Версия 7 (v7) на GPU AMD Radeon Pro 5500M (4 ГБ):
 * 3D-буфер комнаты (X, Y, Z) в видеопамяти, в центре которого находится динамическая фигура
 * с рельефной 2D-поверхностью (горы, впадины, шероховатости от коллизии внутреннего состояния и внешней среды).
 *
 * Что происходит на каждом кадре рендеринга в шейдере GPU:
 *   1. В центре 3D-комнаты (X, Y, Z) живет замкнутая 2D-оболочка фигуры (сетка 8x8 = 64 сферических сектора).
 *   2. Изнутри фигуры давит внутреннее ожидание R_int(u, v), снаружи бьет внешний контекст P_ext(u, v).
 *   3. Разница eps(u, v) = P_ext(u, v) - R_int(u, v) деформирует поверхность фигуры:
 *      образуются горы и впадины с крутизной склонов dR/du, dR/dv и шероховатостью laplacian(R).
 *   4. Вычисляется реальная площадь рельефа ("1 км по равнине vs 3 км по горе"):
 *      gamma_area(u, v) = sqrt(1 + (dR/du)^2 + (dR/dv)^2 + roughness^2) >= 1.0.
 *   5. Из каждой точки рельефной поверхности взгляд идет вглубь 3D-комнаты (X, Y, Z) с учетом наклона склона горы
 *      и перспективной дальности до 3D-узлов памяти!
 */

#include <OpenCL/opencl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define VOCAB 256
#define SIDE 8
#define D_MODEL (SIDE * SIDE) /* 64 сектора рельефной поверхности фигуры */
#define N_OBJ 128             /* 3D-узлы в комнате (X, Y, Z) */
#define VOXEL_SIDE 32         /* 3D-воксельная сетка комнаты 32x32x32 в VRAM */
#define BATCH_TOKENS 256

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static const char *KERNEL_SRC =
    "#define VOCAB 256\n"
    "#define SIDE 8\n"
    "#define D_MODEL 64\n"
    "#define N_OBJ 128\n"
    "#define VOXEL_SIDE 32\n"
    "\n"
    "inline float softplusf(float x) {\n"
    "    return (x > 20.0f) ? x : log(1.0f + exp(clamp(x, -20.0f, 20.0f)));\n"
    "}\n"
    "\n"
    "/* Pass 1: Деформация центральной фигуры в 3D-комнате (X,Y,Z), расчет горного рельефа gamma_area\n"
    " * и перспективная коллизия с 3D-объектами вокруг и внутри фигуры */\n"
    "__kernel void pass1_relief_figure_3d_room(\n"
    "    __global const int *tokens,\n"
    "    __global const float *embed,\n"
    "    __global const float *u_dir,        /* (64, 3): единичные лучи сферы */\n"
    "    __global const float *e_theta,      /* (64, 3): касательные орты широты */\n"
    "    __global const float *e_phi,        /* (64, 3): касательные орты долготы */\n"
    "    __global const float *obj_pos,      /* (N_OBJ, 3): 3D координаты узлов в комнате (X,Y,Z) */\n"
    "    __global const float *obj_z_open,   /* (N_OBJ): локальные пороги узлов */\n"
    "    __global const float *obj_in,       /* (N_OBJ, 64): входные грани */\n"
    "    __global const float *obj_out,      /* (N_OBJ, 64): выходные грани */\n"
    "    __global float *room_voxels,        /* (32*32*32): 3D-буфер комнаты (X,Y,Z) в VRAM */\n"
    "    __global float *eps_amplified_out,  /* (B, 64): усиленная рельефом невязка */\n"
    "    __global float *gamma_area_out,     /* (B, 64): коэффициент площади рельефа */\n"
    "    __global float *mu_post_out,        /* (B, 64): адаптированное внутреннее состояние */\n"
    "    __global int *active_facets_cnt,    /* (B): сколько граней вскрылось */\n"
    "    int n_tokens)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    if (tid >= n_tokens) return;\n"
    "\n"
    "    int tok = tokens[tid];\n"
    "    int prev1 = (tid > 0) ? tokens[tid - 1] : tok;\n"
    "    int prev2 = (tid > 1) ? tokens[tid - 2] : prev1;\n"
    "\n"
    "    float p_ext[D_MODEL];\n"
    "    float r_int[D_MODEL];\n"
    "    float eps[D_MODEL];\n"
    "    float R_surf[D_MODEL];\n"
    "\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        p_ext[d] = embed[tok * D_MODEL + d] + 0.35f * embed[prev1 * D_MODEL + d];\n"
    "        r_int[d] = 0.70f * embed[prev1 * D_MODEL + d] + 0.30f * embed[prev2 * D_MODEL + d];\n"
    "        eps[d] = p_ext[d] - r_int[d];\n"
    "        /* Высота поверхности фигуры в направлении d = (u, v) от столкновения внутреннего и внешнего */\n"
    "        R_surf[d] = softplusf(0.8f + 0.45f * tanh(r_int[d]) + 0.55f * eps[d]);\n"
    "    }\n"
    "\n"
    "    /* Вычисляем рельеф (горы, впадины, шероховатость), площадь gamma_area >= 1 и главный хребет в 3D-комнате */\n"
    "    float gamma[D_MODEL];\n"
    "    float eps_amp[D_MODEL];\n"
    "    float ridge_x = 0.0f, ridge_y = 0.0f, ridge_z = 0.0f;\n"
    "    float norm_x = 0.0f, norm_y = 0.0f, norm_z = 0.0f;\n"
    "    float w_sum = 1e-6f;\n"
    "\n"
    "    for (int u = 0; u < SIDE; ++u) {\n"
    "        int u_up = (u > 0) ? (u - 1) : u;\n"
    "        int u_dn = (u < SIDE - 1) ? (u + 1) : u;\n"
    "        for (int v = 0; v < SIDE; ++v) {\n"
    "            int v_lt = (v + SIDE - 1) & (SIDE - 1);\n"
    "            int v_rt = (v + 1) & (SIDE - 1);\n"
    "            int d = u * SIDE + v;\n"
    "\n"
    "            float dR_du = 0.5f * (R_surf[u_dn * SIDE + v] - R_surf[u_up * SIDE + v]);\n"
    "            float dR_dv = 0.5f * (R_surf[u * SIDE + v_rt] - R_surf[u * SIDE + v_lt]);\n"
    "            float lap = R_surf[u * SIDE + v_rt] + R_surf[u * SIDE + v_lt] - 2.0f * R_surf[d];\n"
    "\n"
    "            /* Развертка площади по горе: sqrt(1 + dR_du^2 + dR_dv^2 + roughness^2) */\n"
    "            float g_area = sqrt(1.0f + dR_du * dR_du + dR_dv * dR_dv + 0.5f * lap * lap);\n"
    "            gamma[d] = g_area;\n"
    "            eps_amp[d] = eps[d] * g_area;\n"
    "            gamma_area_out[tid * D_MODEL + d] = g_area;\n"
    "            eps_amplified_out[tid * D_MODEL + d] = eps_amp[d];\n"
    "\n"
    "            float rad = R_surf[d];\n"
    "            float px = rad * u_dir[d * 3 + 0];\n"
    "            float py = rad * u_dir[d * 3 + 1];\n"
    "            float pz = rad * u_dir[d * 3 + 2];\n"
    "\n"
    "            /* Записываем положение оболочки фигуры в 3D-буфер комнаты (X, Y, Z) на последнем кадре */\n"
    "            if (tid == n_tokens - 1) {\n"
    "                int vx = clamp((int)((px + 2.0f) * 8.0f), 0, VOXEL_SIDE - 1);\n"
    "                int vy = clamp((int)((py + 2.0f) * 8.0f), 0, VOXEL_SIDE - 1);\n"
    "                int vz = clamp((int)((pz + 2.0f) * 8.0f), 0, VOXEL_SIDE - 1);\n"
    "                room_voxels[(vz * VOXEL_SIDE + vy) * VOXEL_SIDE + vx] = g_area;\n"
    "            }\n"
    "\n"
    "            /* Наклон нормали на склоне горы */\n"
    "            float nx = u_dir[d * 3 + 0] - dR_du * e_theta[d * 3 + 0] - dR_dv * e_phi[d * 3 + 0];\n"
    "            float ny = u_dir[d * 3 + 1] - dR_du * e_theta[d * 3 + 1] - dR_dv * e_phi[d * 3 + 1];\n"
    "            float nz = u_dir[d * 3 + 2] - dR_du * e_theta[d * 3 + 2] - dR_dv * e_phi[d * 3 + 2];\n"
    "\n"
    "            float w = fabs(eps[d]);\n"
    "            ridge_x += px * w; ridge_y += py * w; ridge_z += pz * w;\n"
    "            norm_x += nx * w;  norm_y += ny * w;  norm_z += nz * w;\n"
    "            w_sum += w;\n"
    "        }\n"
    "    }\n"
    "    ridge_x /= w_sum; ridge_y /= w_sum; ridge_z /= w_sum;\n"
    "    float n_len = rsqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z + 1e-8f);\n"
    "    norm_x *= n_len; norm_y *= n_len; norm_z *= n_len;\n"
    "\n"
    "    float corr[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) corr[d] = 0.0f;\n"
    "    int opened_facets = 0;\n"
    "\n"
    "    /* Перспектива и дальность от рельефной фигуры к 3D-объектам в комнате (X, Y, Z) */\n"
    "    for (int j = 0; j < N_OBJ; ++j) {\n"
    "        float dx = obj_pos[j * 3 + 0] - ridge_x;\n"
    "        float dy = obj_pos[j * 3 + 1] - ridge_y;\n"
    "        float dz = obj_pos[j * 3 + 2] - ridge_z;\n"
    "        float dist = sqrt(dx * dx + dy * dy + dz * dz + 1e-6f);\n"
    "        float cos_view = max(0.0f, (dx * norm_x + dy * norm_y + dz * norm_z) / dist);\n"
    "        float persp = (0.35f + 0.65f * cos_view) * exp(-dist * 0.65f);\n"
    "        float z_thresh = obj_z_open[j];\n"
    "        if (persp * 2.5f <= z_thresh) continue;\n"
    "\n"
    "        int off = j * D_MODEL;\n"
    "        float dot = 0.0f;\n"
    "        int obj_opened = 0;\n"
    "        for (int d = 0; d < D_MODEL; ++d) {\n"
    "            /* Сектор с крутой горой (gamma[d] > 1) сильнее пробивает порог грани d */\n"
    "            float sec_e = persp * gamma[d];\n"
    "            if (sec_e > z_thresh) {\n"
    "                float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);\n"
    "                dot += eps_amp[d] * obj_in[off + d] * gate;\n"
    "                obj_opened++;\n"
    "            }\n"
    "        }\n"
    "        if (obj_opened > 0) {\n"
    "            opened_facets += obj_opened;\n"
    "            float resp = tanh(dot);\n"
    "            for (int d = 0; d < D_MODEL; ++d) {\n"
    "                float sec_e = persp * gamma[d];\n"
    "                if (sec_e > z_thresh) {\n"
    "                    float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);\n"
    "                    corr[d] += resp * obj_out[off + d] * gate;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    active_facets_cnt[tid] = opened_facets;\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        mu_post_out[tid * D_MODEL + d] = p_ext[d] + corr[d];\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void pass2_head_and_grad(\n"
    "    __global const float *mu_post_out,\n"
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
    "    for (int d = 0; d < D_MODEL; ++d) h[d] = mu_post_out[tid * D_MODEL + d];\n"
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
    "__kernel void pass3_update_relief_objects(\n"
    "    __global const float *eps_amplified_out,\n"
    "    __global const float *gamma_area_out,\n"
    "    __global const float *d_hidden,\n"
    "    __global const float *obj_z_open,\n"
    "    __global float *obj_in,\n"
    "    __global float *obj_out,\n"
    "    int n_tokens,\n"
    "    float lr)\n"
    "{\n"
    "    int j = get_global_id(0);\n"
    "    if (j >= N_OBJ) return;\n"
    "    int off = j * D_MODEL;\n"
    "    float z_thresh = obj_z_open[j];\n"
    "    float gin[D_MODEL], gout[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) { gin[d] = 0.0f; gout[d] = 0.0f; }\n"
    "\n"
    "    for (int tid = 0; tid < n_tokens; ++tid) {\n"
    "        for (int d = 0; d < D_MODEL; ++d) {\n"
    "            float g_area = gamma_area_out[tid * D_MODEL + d];\n"
    "            float sec_e = 0.45f * g_area;\n"
    "            if (sec_e <= z_thresh) continue;\n"
    "            float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);\n"
    "            float dh = d_hidden[tid * D_MODEL + d] * gate;\n"
    "            float ea = eps_amplified_out[tid * D_MODEL + d];\n"
    "            gout[d] += dh * tanh(ea * obj_in[off + d]);\n"
    "            gin[d]  += dh * obj_out[off + d] * ea;\n"
    "        }\n"
    "    }\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        obj_in[off + d]  -= lr * clamp(gin[d], -1.0f, 1.0f);\n"
    "        obj_out[off + d] -= lr * clamp(gout[d], -1.0f, 1.0f);\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void pass4_update_head_and_embed(\n"
    "    __global const int *tokens,\n"
    "    __global const float *mu_post_out,\n"
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
    "        for (int d = 0; d < D_MODEL; ++d) g_head[d] += dl * mu_post_out[tid * D_MODEL + d];\n"
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
    if (!fp) return 1;
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
        if (strstr(nm, "Radeon")) { radeon = devs[i]; break; }
    }
    char dev_name[256] = {0};
    cl_ulong vram_mb = 0;
    clGetDeviceInfo(radeon, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    clGetDeviceInfo(radeon, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram_mb), &vram_mb, NULL);
    printf("=== V7 НА GPU %s (%llu МБ VRAM): 3D-КОМНАТА С ДИНАМИЧЕСКОЙ РЕЛЬЕФНОЙ ФИГУРОЙ ===\n",
           dev_name, vram_mb / (1024 * 1024));

    cl_int err = 0;
    cl_context ctx = clCreateContext(NULL, 1, &radeon, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, radeon, 0, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &radeon, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(prog, radeon, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Ошибка компиляции:\n%s\n", log);
        return 1;
    }

    cl_kernel k_p1 = clCreateKernel(prog, "pass1_relief_figure_3d_room", &err);
    cl_kernel k_p2 = clCreateKernel(prog, "pass2_head_and_grad", &err);
    cl_kernel k_p3 = clCreateKernel(prog, "pass3_update_relief_objects", &err);
    cl_kernel k_p4 = clCreateKernel(prog, "pass4_update_head_and_embed", &err);

    /* Сферическая сетка 8x8 в центре 3D-комнаты */
    float h_udir[D_MODEL * 3], h_etheta[D_MODEL * 3], h_ephi[D_MODEL * 3];
    for (int u = 0; u < SIDE; ++u) {
        float th = (0.15f + 0.70f * ((float)u / (SIDE - 1))) * 3.14159265f;
        for (int v = 0; v < SIDE; ++v) {
            float ph = 2.0f * 3.14159265f * ((float)v / SIDE);
            int d = u * SIDE + v;
            h_udir[d * 3 + 0] = sinf(th) * cosf(ph);
            h_udir[d * 3 + 1] = sinf(th) * sinf(ph);
            h_udir[d * 3 + 2] = cosf(th);
            h_etheta[d * 3 + 0] = cosf(th) * cosf(ph);
            h_etheta[d * 3 + 1] = cosf(th) * sinf(ph);
            h_etheta[d * 3 + 2] = -sinf(th);
            h_ephi[d * 3 + 0] = -sinf(ph);
            h_ephi[d * 3 + 1] = cosf(ph);
            h_ephi[d * 3 + 2] = 0.0f;
        }
    }

    uint32_t rng = 42;
    float *h_embed = (float *)malloc(sizeof(float) * VOCAB * D_MODEL);
    float *h_head = (float *)malloc(sizeof(float) * VOCAB * D_MODEL);
    float *h_zero = (float *)calloc(VOCAB * D_MODEL, sizeof(float));
    float *h_pos = (float *)malloc(sizeof(float) * N_OBJ * 3);
    float *h_zopen = (float *)malloc(sizeof(float) * N_OBJ);
    float *h_in = (float *)malloc(sizeof(float) * N_OBJ * D_MODEL);
    float *h_out = (float *)malloc(sizeof(float) * N_OBJ * D_MODEL);
    float *h_voxels = (float *)calloc(VOXEL_SIDE * VOXEL_SIDE * VOXEL_SIDE, sizeof(float));

    for (int i = 0; i < VOCAB * D_MODEL; ++i) {
        h_embed[i] = frand_sym(&rng) * 0.15f;
        h_head[i] = frand_sym(&rng) * 0.15f;
    }
    for (int j = 0; j < N_OBJ; ++j) {
        h_pos[j * 3 + 0] = frand_sym(&rng) * 1.4f;
        h_pos[j * 3 + 1] = frand_sym(&rng) * 1.4f;
        h_pos[j * 3 + 2] = frand_sym(&rng) * 1.4f;
        h_zopen[j] = 0.45f + 0.35f * fabsf(frand_sym(&rng));
    }
    for (int i = 0; i < N_OBJ * D_MODEL; ++i) {
        h_in[i] = frand_sym(&rng) * 0.12f;
        h_out[i] = frand_sym(&rng) * 0.12f;
    }

    cl_mem d_tok = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * BATCH_TOKENS, NULL, &err);
    cl_mem d_tgt = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * BATCH_TOKENS, NULL, &err);
    cl_mem d_emb = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_embed, &err);
    cl_mem d_emb_m = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_zero, &err);
    cl_mem d_hd = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_head, &err);
    cl_mem d_hd_m = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOCAB * D_MODEL, h_zero, &err);

    cl_mem d_udir = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(h_udir), h_udir, &err);
    cl_mem d_etheta = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(h_etheta), h_etheta, &err);
    cl_mem d_ephi = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(h_ephi), h_ephi, &err);
    cl_mem d_pos = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * 3, h_pos, &err);
    cl_mem d_zopen = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ, h_zopen, &err);
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * D_MODEL, h_in, &err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * D_MODEL, h_out, &err);
    cl_mem d_vox = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * VOXEL_SIDE * VOXEL_SIDE * VOXEL_SIDE, h_voxels, &err);

    cl_mem d_eps_amp = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_gamma = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_mupost = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_dlog = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * VOCAB, NULL, &err);
    cl_mem d_dhid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_loss = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS, NULL, &err);
    cl_mem d_act_cnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * BATCH_TOKENS, NULL, &err);

    int n_tokens = BATCH_TOKENS;
    float lr = 0.08f;

    clSetKernelArg(k_p1, 0, sizeof(cl_mem), &d_tok);
    clSetKernelArg(k_p1, 1, sizeof(cl_mem), &d_emb);
    clSetKernelArg(k_p1, 2, sizeof(cl_mem), &d_udir);
    clSetKernelArg(k_p1, 3, sizeof(cl_mem), &d_etheta);
    clSetKernelArg(k_p1, 4, sizeof(cl_mem), &d_ephi);
    clSetKernelArg(k_p1, 5, sizeof(cl_mem), &d_pos);
    clSetKernelArg(k_p1, 6, sizeof(cl_mem), &d_zopen);
    clSetKernelArg(k_p1, 7, sizeof(cl_mem), &d_in);
    clSetKernelArg(k_p1, 8, sizeof(cl_mem), &d_out);
    clSetKernelArg(k_p1, 9, sizeof(cl_mem), &d_vox);
    clSetKernelArg(k_p1, 10, sizeof(cl_mem), &d_eps_amp);
    clSetKernelArg(k_p1, 11, sizeof(cl_mem), &d_gamma);
    clSetKernelArg(k_p1, 12, sizeof(cl_mem), &d_mupost);
    clSetKernelArg(k_p1, 13, sizeof(cl_mem), &d_act_cnt);
    clSetKernelArg(k_p1, 14, sizeof(int), &n_tokens);

    clSetKernelArg(k_p2, 0, sizeof(cl_mem), &d_mupost);
    clSetKernelArg(k_p2, 1, sizeof(cl_mem), &d_tgt);
    clSetKernelArg(k_p2, 2, sizeof(cl_mem), &d_hd);
    clSetKernelArg(k_p2, 3, sizeof(cl_mem), &d_dlog);
    clSetKernelArg(k_p2, 4, sizeof(cl_mem), &d_dhid);
    clSetKernelArg(k_p2, 5, sizeof(cl_mem), &d_loss);
    clSetKernelArg(k_p2, 6, sizeof(int), &n_tokens);

    clSetKernelArg(k_p3, 0, sizeof(cl_mem), &d_eps_amp);
    clSetKernelArg(k_p3, 1, sizeof(cl_mem), &d_gamma);
    clSetKernelArg(k_p3, 2, sizeof(cl_mem), &d_dhid);
    clSetKernelArg(k_p3, 3, sizeof(cl_mem), &d_zopen);
    clSetKernelArg(k_p3, 4, sizeof(cl_mem), &d_in);
    clSetKernelArg(k_p3, 5, sizeof(cl_mem), &d_out);
    clSetKernelArg(k_p3, 6, sizeof(int), &n_tokens);
    clSetKernelArg(k_p3, 7, sizeof(float), &lr);

    clSetKernelArg(k_p4, 0, sizeof(cl_mem), &d_tok);
    clSetKernelArg(k_p4, 1, sizeof(cl_mem), &d_mupost);
    clSetKernelArg(k_p4, 2, sizeof(cl_mem), &d_dlog);
    clSetKernelArg(k_p4, 3, sizeof(cl_mem), &d_dhid);
    clSetKernelArg(k_p4, 4, sizeof(cl_mem), &d_hd);
    clSetKernelArg(k_p4, 5, sizeof(cl_mem), &d_hd_m);
    clSetKernelArg(k_p4, 6, sizeof(cl_mem), &d_emb);
    clSetKernelArg(k_p4, 7, sizeof(cl_mem), &d_emb_m);
    clSetKernelArg(k_p4, 8, sizeof(int), &n_tokens);
    clSetKernelArg(k_p4, 9, sizeof(float), &lr);

    size_t g_tok = BATCH_TOKENS, g_obj = N_OBJ, g_voc = VOCAB;
    int h_tok[BATCH_TOKENS], h_tgt[BATCH_TOKENS], h_cnt[BATCH_TOKENS];
    float h_loss[BATCH_TOKENS], h_gam[BATCH_TOKENS * D_MODEL];
    int total_frames = 2000;
    long cursor = 0;
    double t0 = now_sec();

    for (int frame = 1; frame <= total_frames; ++frame) {
        if (cursor + BATCH_TOKENS + 1 >= fsize) cursor = 0;
        for (int i = 0; i < BATCH_TOKENS; ++i) {
            h_tok[i] = (int)raw_bytes[cursor + i];
            h_tgt[i] = (int)raw_bytes[cursor + i + 1];
        }
        cursor += BATCH_TOKENS;

        clEnqueueWriteBuffer(queue, d_tok, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tok, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_tgt, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tgt, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_p1, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_p2, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_p3, 1, NULL, &g_obj, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_p4, 1, NULL, &g_voc, NULL, 0, NULL, NULL);

        if (frame == 1 || frame % 400 == 0) {
            clEnqueueReadBuffer(queue, d_loss, CL_FALSE, 0, sizeof(float) * BATCH_TOKENS, h_loss, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_gamma, CL_FALSE, 0, sizeof(float) * BATCH_TOKENS * D_MODEL, h_gam, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_act_cnt, CL_TRUE, 0, sizeof(int) * BATCH_TOKENS, h_cnt, 0, NULL, NULL);
            double lsum = 0.0, gsum = 0.0, fsum = 0.0;
            for (int i = 0; i < BATCH_TOKENS; ++i) {
                lsum += h_loss[i];
                fsum += h_cnt[i];
            }
            for (int i = 0; i < BATCH_TOKENS * D_MODEL; ++i) gsum += h_gam[i];
            double bpc = (lsum / BATCH_TOKENS) / log(2.0);
            double avg_gamma = gsum / (BATCH_TOKENS * D_MODEL);
            double facet_pct = (fsum / (BATCH_TOKENS * N_OBJ * D_MODEL)) * 100.0;
            double fps = frame / (now_sec() - t0);
            printf("Кадр %4d | BPC=%.3f | Площадь рельефа=%.2fx | Открыто граней=%.1f%% | %.1f кадров/с\n",
                   frame, bpc, avg_gamma, facet_pct, fps);
        }
    }
    clFinish(queue);
    double elapsed = now_sec() - t0;

    double val_loss_sum = 0.0;
    int val_batches = 0;
    for (long off = 0; off + BATCH_TOKENS + 1 < vsize && val_batches < 100; off += BATCH_TOKENS) {
        for (int i = 0; i < BATCH_TOKENS; ++i) {
            h_tok[i] = (int)val_bytes[off + i];
            h_tgt[i] = (int)val_bytes[off + i + 1];
        }
        clEnqueueWriteBuffer(queue, d_tok, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tok, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, d_tgt, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_tgt, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_p1, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
        clEnqueueNDRangeKernel(queue, k_p2, 1, NULL, &g_tok, NULL, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, d_loss, CL_TRUE, 0, sizeof(float) * BATCH_TOKENS, h_loss, 0, NULL, NULL);
        for (int i = 0; i < BATCH_TOKENS; ++i) val_loss_sum += h_loss[i];
        val_batches++;
    }
    double val_bpc = (val_loss_sum / (val_batches * BATCH_TOKENS)) / log(2.0);
    double val_ppl = exp(val_loss_sum / (val_batches * BATCH_TOKENS));
    printf("Итого GPU v7: 2000 кадров за %.2f с (%.1f кадров/с) | Validation BPC = %.3f | Перплексия = %.2f\n",
           elapsed, total_frames / elapsed, val_bpc, val_ppl);
    return 0;
}
