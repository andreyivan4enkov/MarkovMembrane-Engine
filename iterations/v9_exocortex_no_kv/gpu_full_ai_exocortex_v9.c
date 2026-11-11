/* Версия 9 (v9) на GPU AMD Radeon Pro 5500M (4 ГБ):
 * Полноценный AI-алгоритм на видеокарте:
 * - Полигональный 2D-рельеф в центре 3D-комнаты (X, Y, Z) + Голографический принцип V^(2/3)
 * - Аппаратные векторы float4(R, G, B, L) (3 цвета + яркость канала)
 * - Внешняя 3D-среда комнаты (32 кристалла памяти фактов и навыков перехода) БЕЗ KV-КЭША!
 * - Запись навыков вовне и обращение к внешней памяти по необходимости (только на непогашенный внутренним миром остаток ошибки!).
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
#define D_MODEL (SIDE * SIDE)
#define N_OBJ 128
#define N_ROOM 32
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
    "#define N_ROOM 32\n"
    "\n"
    "__kernel void pass1_full_ai_exocortex(\n"
    "    __global const int *tokens,\n"
    "    __global const float *embed,\n"
    "    __global const float *u_dir,\n"
    "    __global const float *base_poly_area,\n"
    "    __global const float *obj_pos,\n"
    "    __global const float *obj_z_open,\n"
    "    __global const float4 *obj_rgbl_flt,\n"
    "    __global const float *obj_in,\n"
    "    __global const float *obj_out,\n"
    "    __global const float *room_anchors,\n"
    "    __global const float *room_z_need,\n"
    "    __global float *room_facts,\n"
    "    __global float *room_skills,\n"
    "    __global float *flux_out,\n"
    "    __global float *lum_out,\n"
    "    __global float *sig_out,\n"
    "    __global float *mu_post_out,\n"
    "    __global int *active_facets_cnt,\n"
    "    __global int *recalled_slots_cnt,\n"
    "    int n_tokens)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    if (tid >= n_tokens) return;\n"
    "\n"
    "    int tok = tokens[tid];\n"
    "    int prev1 = (tid > 0) ? tokens[tid - 1] : tok;\n"
    "    int prev2 = (tid > 1) ? tokens[tid - 2] : prev1;\n"
    "\n"
    "    float w_ext[D_MODEL], prev_ext[D_MODEL], w_int[D_MODEL], eps[D_MODEL], c_int[D_MODEL], R_vert[D_MODEL];\n"
    "    float3 p_surf[D_MODEL];\n"
    "\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        prev_ext[d] = embed[prev1 * D_MODEL + d] + 0.35f * embed[prev2 * D_MODEL + d];\n"
    "        w_ext[d] = embed[tok * D_MODEL + d] + 0.35f * embed[prev1 * D_MODEL + d];\n"
    "        w_int[d] = 0.70f * embed[prev1 * D_MODEL + d] + 0.30f * embed[prev2 * D_MODEL + d];\n"
    "        eps[d] = w_ext[d] - w_int[d];\n"
    "        c_int[d] = tanh(fabs(w_int[d]) * 1.5f);\n"
    "        float vol = 0.8f + c_int[d] + 0.5f * tanh(0.5f * eps[d] * eps[d]);\n"
    "        float r_val = cbrt(vol) * (1.0f + 0.35f * tanh(w_int[d]) + 0.45f * tanh(eps[d]));\n"
    "        R_vert[d] = r_val;\n"
    "        p_surf[d] = (float3)(r_val * u_dir[d * 3 + 0], r_val * u_dir[d * 3 + 1], r_val * u_dir[d * 3 + 2]);\n"
    "    }\n"
    "\n"
    "    float4 proc_rgbl[D_MODEL];\n"
    "    float rad_flux[D_MODEL];\n"
    "    float3 active_pos = (float3)(0.0f, 0.0f, 0.0f);\n"
    "    float3 active_norm = (float3)(0.0f, 0.0f, 0.0f);\n"
    "    float w_sum = 1e-6f;\n"
    "\n"
    "    for (int u = 0; u < SIDE; ++u) {\n"
    "        int u_next = (u < SIDE - 1) ? (u + 1) : (SIDE - 2);\n"
    "        for (int v = 0; v < SIDE; ++v) {\n"
    "            int v_next = (v + 1) & (SIDE - 1);\n"
    "            int v_prev = (v + SIDE - 1) & (SIDE - 1);\n"
    "            int d = u * SIDE + v;\n"
    "\n"
    "            float3 edge_u = p_surf[u_next * SIDE + v] - p_surf[d];\n"
    "            float3 edge_v = p_surf[u * SIDE + v_next] - p_surf[d];\n"
    "            float3 poly_cross = cross(edge_u, edge_v);\n"
    "            float area_3d = length(poly_cross) + 1e-6f;\n"
    "            float3 n_poly = poly_cross / area_3d;\n"
    "            float3 u_vec = (float3)(u_dir[d * 3 + 0], u_dir[d * 3 + 1], u_dir[d * 3 + 2]);\n"
    "            if (dot(n_poly, u_vec) < 0.0f) n_poly = -n_poly;\n"
    "\n"
    "            float crease = R_vert[u * SIDE + v_next] + R_vert[u * SIDE + v_prev] - 2.0f * R_vert[d];\n"
    "            float dA_poly = (area_3d / (base_poly_area[d] + 1e-5f)) * sqrt(1.0f + 0.5f * crease * crease);\n"
    "\n"
    "            float eps_u = tanh(eps[d]);\n"
    "            float tilt = 1.0f - fabs(dot(n_poly, u_vec));\n"
    "            float col_R = eps_u * cos(1.4f * R_vert[d]);\n"
    "            float col_G = eps_u * tilt * cos(2.8f * R_vert[d]) + tanh(crease) * sin(2.8f * R_vert[d]);\n"
    "            float col_B = tanh(crease) * cos(5.2f * eps_u) + 0.5f * c_int[d] * eps_u;\n"
    "            float lum_L = 1.0f + 0.65f * tanh(0.5f * eps[d] * eps[d] + c_int[d] * fabs(eps[d]));\n"
    "\n"
    "            proc_rgbl[d] = (float4)(col_R * lum_L, col_G * lum_L, col_B * lum_L, (lum_L - 1.0f) * eps_u);\n"
    "            float flux = lum_L * dA_poly;\n"
    "            rad_flux[d] = flux;\n"
    "            flux_out[tid * D_MODEL + d] = dA_poly;\n"
    "            lum_out[tid * D_MODEL + d] = lum_L;\n"
    "            sig_out[tid * D_MODEL + d] = eps[d] + 0.25f * (col_R + col_G + col_B) * lum_L;\n"
    "\n"
    "            float fw = fabs(eps[d]) * flux;\n"
    "            active_pos += p_surf[d] * fw;\n"
    "            active_norm += n_poly * fw;\n"
    "            w_sum += fw;\n"
    "        }\n"
    "    }\n"
    "    active_pos /= w_sum;\n"
    "    active_norm = normalize(active_norm + (float3)(1e-6f, 0.0f, 0.0f));\n"
    "\n"
    "    float corr[D_MODEL];\n"
    "    for (int d = 0; d < D_MODEL; ++d) corr[d] = 0.0f;\n"
    "    int opened = 0;\n"
    "\n"
    "    for (int j = 0; j < N_OBJ; ++j) {\n"
    "        float3 opos = (float3)(obj_pos[j * 3 + 0], obj_pos[j * 3 + 1], obj_pos[j * 3 + 2]);\n"
    "        float3 delta = opos - active_pos;\n"
    "        float dist = length(delta) + 1e-6f;\n"
    "        float cos_v = max(0.0f, dot(active_norm, delta / dist));\n"
    "        float persp = (0.35f + 0.65f * cos_v) * exp(-dist * 0.65f);\n"
    "        float z_thresh = obj_z_open[j];\n"
    "        if (persp * 3.5f <= z_thresh) continue;\n"
    "\n"
    "        float4 flt = obj_rgbl_flt[j];\n"
    "        int off = j * D_MODEL;\n"
    "        float dot_acc = 0.0f;\n"
    "        int obj_open = 0;\n"
    "        for (int d = 0; d < D_MODEL; ++d) {\n"
    "            float sec_e = persp * rad_flux[d];\n"
    "            if (sec_e > z_thresh) {\n"
    "                float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);\n"
    "                float rgbl_sig = eps[d] + dot(proc_rgbl[d], flt);\n"
    "                dot_acc += rgbl_sig * obj_in[off + d] * gate;\n"
    "                obj_open++;\n"
    "            }\n"
    "        }\n"
    "        if (obj_open > 0) {\n"
    "            opened += obj_open;\n"
    "            float resp = tanh(dot_acc);\n"
    "            for (int d = 0; d < D_MODEL; ++d) {\n"
    "                float sec_e = persp * rad_flux[d];\n"
    "                if (sec_e > z_thresh) {\n"
    "                    float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);\n"
    "                    corr[d] += resp * obj_out[off + d] * gate;\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "\n"
    "    /* Обращение к внешним кристаллам 3D-комнаты по необходимости (на непогашенный остаток ошибки!) */\n"
    "    float unsolved_norm2 = 0.0f;\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        float u_err = eps[d] - corr[d];\n"
    "        unsolved_norm2 += u_err * u_err;\n"
    "    }\n"
    "    float need_amp = sqrt(unsolved_norm2 + 1e-6f);\n"
    "    int recalled = 0;\n"
    "    for (int m = 0; m < N_ROOM; ++m) {\n"
    "        float3 rpos = (float3)(room_anchors[m * 3 + 0], room_anchors[m * 3 + 1], room_anchors[m * 3 + 2]);\n"
    "        float3 to_s = rpos - active_pos;\n"
    "        float d_s = length(to_s) + 1e-6f;\n"
    "        float cos_b = max(0.0f, dot(active_norm, to_s / d_s));\n"
    "        float beam = cos_b * cos_b * cos_b;\n"
    "        float q_energy = need_amp * beam;\n"
    "        if (q_energy > room_z_need[m]) {\n"
    "            recalled++;\n"
    "            float rg = (q_energy - room_z_need[m]) / (q_energy - room_z_need[m] + 1.0f) * 0.15f;\n"
    "            int m_off = m * D_MODEL;\n"
    "            for (int d = 0; d < D_MODEL; ++d) {\n"
    "                corr[d] += rg * (room_facts[m_off + d] + room_skills[m_off + d]);\n"
    "                if (tid == n_tokens - 1) {\n"
    "                    room_facts[m_off + d] = 0.90f * room_facts[m_off + d] + 0.10f * prev_ext[d];\n"
    "                    room_skills[m_off + d] = 0.90f * room_skills[m_off + d] + 0.10f * (w_ext[d] - prev_ext[d]);\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    active_facets_cnt[tid] = opened;\n"
    "    recalled_slots_cnt[tid] = recalled;\n"
    "    for (int d = 0; d < D_MODEL; ++d) {\n"
    "        mu_post_out[tid * D_MODEL + d] = w_ext[d] + corr[d];\n"
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
    "__kernel void pass3_update_poly_rgbl_objects(\n"
    "    __global const float *sig_out,\n"
    "    __global const float *flux_out,\n"
    "    __global const float *lum_out,\n"
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
    "            float sec_e = 0.35f * flux_out[tid * D_MODEL + d] * lum_out[tid * D_MODEL + d];\n"
    "            if (sec_e <= z_thresh) continue;\n"
    "            float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);\n"
    "            float dh = d_hidden[tid * D_MODEL + d] * gate;\n"
    "            float s_val = sig_out[tid * D_MODEL + d];\n"
    "            gout[d] += dh * tanh(s_val * obj_in[off + d]);\n"
    "            gin[d]  += dh * obj_out[off + d] * s_val;\n"
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
    clGetDeviceInfo(radeon, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("=== V9 НА GPU %s: ПОЛНЫЙ AI С ВНЕШНЕЙ 3D-ПАМЯТЬЮ НАВЫКОВ (0 БАЙТ KV-КЭША) ===\n", dev_name);

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

    cl_kernel k_p1 = clCreateKernel(prog, "pass1_full_ai_exocortex", &err);
    cl_kernel k_p2 = clCreateKernel(prog, "pass2_head_and_grad", &err);
    cl_kernel k_p3 = clCreateKernel(prog, "pass3_update_poly_rgbl_objects", &err);
    cl_kernel k_p4 = clCreateKernel(prog, "pass4_update_head_and_embed", &err);

    float h_udir[D_MODEL * 3], h_base_area[D_MODEL], h_ranch[N_ROOM * 3], h_rzneed[N_ROOM];
    for (int u = 0; u < SIDE; ++u) {
        float th = (0.15f + 0.70f * ((float)u / (SIDE - 1))) * 3.14159265f;
        for (int v = 0; v < SIDE; ++v) {
            float ph = 2.0f * 3.14159265f * ((float)v / SIDE);
            int d = u * SIDE + v;
            h_udir[d * 3 + 0] = sinf(th) * cosf(ph);
            h_udir[d * 3 + 1] = sinf(th) * sinf(ph);
            h_udir[d * 3 + 2] = cosf(th);
        }
    }
    for (int u = 0; u < SIDE; ++u) {
        int u_next = (u < SIDE - 1) ? (u + 1) : (SIDE - 2);
        for (int v = 0; v < SIDE; ++v) {
            int v_next = (v + 1) & (SIDE - 1);
            int d = u * SIDE + v;
            float ax = h_udir[(u_next * SIDE + v) * 3 + 0] - h_udir[d * 3 + 0];
            float ay = h_udir[(u_next * SIDE + v) * 3 + 1] - h_udir[d * 3 + 1];
            float az = h_udir[(u_next * SIDE + v) * 3 + 2] - h_udir[d * 3 + 2];
            float bx = h_udir[(u * SIDE + v_next) * 3 + 0] - h_udir[d * 3 + 0];
            float by = h_udir[(u * SIDE + v_next) * 3 + 1] - h_udir[d * 3 + 1];
            float bz = h_udir[(u * SIDE + v_next) * 3 + 2] - h_udir[d * 3 + 2];
            float cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
            h_base_area[d] = sqrtf(cx * cx + cy * cy + cz * cz + 1e-8f);
        }
    }
    for (int m = 0; m < N_ROOM; ++m) {
        float idx = (float)m + 0.5f;
        float phi = acosf(1.0f - 2.0f * idx / N_ROOM);
        float th = 3.14159265f * (1.0f + 2.2360679f) * idx;
        h_ranch[m * 3 + 0] = 2.0f * sinf(phi) * cosf(th);
        h_ranch[m * 3 + 1] = 2.0f * sinf(phi) * sinf(th);
        h_ranch[m * 3 + 2] = 2.0f * cosf(phi);
        h_rzneed[m] = 0.25f + 0.85f * ((float)m / N_ROOM);
    }

    uint32_t rng = 42;
    float *h_embed = (float *)malloc(sizeof(float) * VOCAB * D_MODEL);
    float *h_head = (float *)malloc(sizeof(float) * VOCAB * D_MODEL);
    float *h_zero = (float *)calloc(VOCAB * D_MODEL, sizeof(float));
    float *h_pos = (float *)malloc(sizeof(float) * N_OBJ * 3);
    float *h_zopen = (float *)malloc(sizeof(float) * N_OBJ);
    float *h_flt = (float *)malloc(sizeof(float) * N_OBJ * 4);
    float *h_in = (float *)malloc(sizeof(float) * N_OBJ * D_MODEL);
    float *h_out = (float *)malloc(sizeof(float) * N_OBJ * D_MODEL);
    float *h_rfacts = (float *)calloc(N_ROOM * D_MODEL, sizeof(float));
    float *h_rskills = (float *)calloc(N_ROOM * D_MODEL, sizeof(float));

    for (int i = 0; i < VOCAB * D_MODEL; ++i) {
        h_embed[i] = frand_sym(&rng) * 0.15f;
        h_head[i] = frand_sym(&rng) * 0.15f;
    }
    for (int j = 0; j < N_OBJ; ++j) {
        h_pos[j * 3 + 0] = frand_sym(&rng) * 1.4f;
        h_pos[j * 3 + 1] = frand_sym(&rng) * 1.4f;
        h_pos[j * 3 + 2] = frand_sym(&rng) * 1.4f;
        h_zopen[j] = 0.45f + 0.35f * fabsf(frand_sym(&rng));
        for (int c = 0; c < 4; ++c) h_flt[j * 4 + c] = frand_sym(&rng) * 0.25f;
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
    cl_mem d_barea = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(h_base_area), h_base_area, &err);
    cl_mem d_pos = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * 3, h_pos, &err);
    cl_mem d_zopen = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ, h_zopen, &err);
    cl_mem d_flt = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * 4, h_flt, &err);
    cl_mem d_in = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * D_MODEL, h_in, &err);
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_OBJ * D_MODEL, h_out, &err);

    cl_mem d_ranch = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(h_ranch), h_ranch, &err);
    cl_mem d_rzneed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(h_rzneed), h_rzneed, &err);
    cl_mem d_rfacts = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_ROOM * D_MODEL, h_rfacts, &err);
    cl_mem d_rskills = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * N_ROOM * D_MODEL, h_rskills, &err);

    cl_mem d_flux = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_lum = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_sig = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_mupost = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_dlog = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * VOCAB, NULL, &err);
    cl_mem d_dhid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS * D_MODEL, NULL, &err);
    cl_mem d_loss = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * BATCH_TOKENS, NULL, &err);
    cl_mem d_act_cnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * BATCH_TOKENS, NULL, &err);
    cl_mem d_rec_cnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(int) * BATCH_TOKENS, NULL, &err);

    int n_tokens = BATCH_TOKENS;
    float lr = 0.08f;

    clSetKernelArg(k_p1, 0, sizeof(cl_mem), &d_tok);
    clSetKernelArg(k_p1, 1, sizeof(cl_mem), &d_emb);
    clSetKernelArg(k_p1, 2, sizeof(cl_mem), &d_udir);
    clSetKernelArg(k_p1, 3, sizeof(cl_mem), &d_barea);
    clSetKernelArg(k_p1, 4, sizeof(cl_mem), &d_pos);
    clSetKernelArg(k_p1, 5, sizeof(cl_mem), &d_zopen);
    clSetKernelArg(k_p1, 6, sizeof(cl_mem), &d_flt);
    clSetKernelArg(k_p1, 7, sizeof(cl_mem), &d_in);
    clSetKernelArg(k_p1, 8, sizeof(cl_mem), &d_out);
    clSetKernelArg(k_p1, 9, sizeof(cl_mem), &d_ranch);
    clSetKernelArg(k_p1, 10, sizeof(cl_mem), &d_rzneed);
    clSetKernelArg(k_p1, 11, sizeof(cl_mem), &d_rfacts);
    clSetKernelArg(k_p1, 12, sizeof(cl_mem), &d_rskills);
    clSetKernelArg(k_p1, 13, sizeof(cl_mem), &d_flux);
    clSetKernelArg(k_p1, 14, sizeof(cl_mem), &d_lum);
    clSetKernelArg(k_p1, 15, sizeof(cl_mem), &d_sig);
    clSetKernelArg(k_p1, 16, sizeof(cl_mem), &d_mupost);
    clSetKernelArg(k_p1, 17, sizeof(cl_mem), &d_act_cnt);
    clSetKernelArg(k_p1, 18, sizeof(cl_mem), &d_rec_cnt);
    clSetKernelArg(k_p1, 19, sizeof(int), &n_tokens);

    clSetKernelArg(k_p2, 0, sizeof(cl_mem), &d_mupost);
    clSetKernelArg(k_p2, 1, sizeof(cl_mem), &d_tgt);
    clSetKernelArg(k_p2, 2, sizeof(cl_mem), &d_hd);
    clSetKernelArg(k_p2, 3, sizeof(cl_mem), &d_dlog);
    clSetKernelArg(k_p2, 4, sizeof(cl_mem), &d_dhid);
    clSetKernelArg(k_p2, 5, sizeof(cl_mem), &d_loss);
    clSetKernelArg(k_p2, 6, sizeof(int), &n_tokens);

    clSetKernelArg(k_p3, 0, sizeof(cl_mem), &d_sig);
    clSetKernelArg(k_p3, 1, sizeof(cl_mem), &d_flux);
    clSetKernelArg(k_p3, 2, sizeof(cl_mem), &d_lum);
    clSetKernelArg(k_p3, 3, sizeof(cl_mem), &d_dhid);
    clSetKernelArg(k_p3, 4, sizeof(cl_mem), &d_zopen);
    clSetKernelArg(k_p3, 5, sizeof(cl_mem), &d_in);
    clSetKernelArg(k_p3, 6, sizeof(cl_mem), &d_out);
    clSetKernelArg(k_p3, 7, sizeof(int), &n_tokens);
    clSetKernelArg(k_p3, 8, sizeof(float), &lr);

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
    int h_tok[BATCH_TOKENS], h_tgt[BATCH_TOKENS], h_cnt[BATCH_TOKENS], h_rcnt[BATCH_TOKENS];
    float h_loss[BATCH_TOKENS];
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
            clEnqueueReadBuffer(queue, d_act_cnt, CL_FALSE, 0, sizeof(int) * BATCH_TOKENS, h_cnt, 0, NULL, NULL);
            clEnqueueReadBuffer(queue, d_rec_cnt, CL_TRUE, 0, sizeof(int) * BATCH_TOKENS, h_rcnt, 0, NULL, NULL);
            double lsum = 0.0, fsum = 0.0, rsum = 0.0;
            for (int i = 0; i < BATCH_TOKENS; ++i) {
                lsum += h_loss[i];
                fsum += h_cnt[i];
                rsum += h_rcnt[i];
            }
            double bpc = (lsum / BATCH_TOKENS) / log(2.0);
            double facet_pct = (fsum / (BATCH_TOKENS * N_OBJ * D_MODEL)) * 100.0;
            double rec_pct = (rsum / (BATCH_TOKENS * N_ROOM)) * 100.0;
            double fps = frame / (now_sec() - t0);
            printf("Кадр %4d | BPC=%.3f | Внутр.граней=%.1f%% | Внеш.память=%.1f%% | %.1f кадров/с\n",
                   frame, bpc, facet_pct, rec_pct, fps);
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
    printf("Итого GPU v9 (Полный AI + Экзокортекс без KV-кэша): 2000 кадров за %.2f с (%.1f кадров/с) | Validation BPC = %.3f | Перплексия = %.2f\n",
           elapsed, total_frames / elapsed, val_bpc, val_ppl);
    return 0;
}
