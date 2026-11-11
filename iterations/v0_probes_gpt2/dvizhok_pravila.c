/* Шаг движка по правилам из лога. Матрицу целиком не перемножает.

Три самых сильных ортогональных направления — только комната.
Столбец целиком остаётся внутренностью объекта, группами по три.
Контекст — тело. Касание плавное, без порога-константы.
Лобовой ход впитывает внутренние группы, скользящий поворачивает ход.
Дальние группы в кадре тише: перспектива, не стирание.
После удара в комнате остаётся след. Порядок ударов меняет итог.
*/

#include <OpenCL/opencl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SRC =
    "__kernel void shag(\n"
    "    __global const float *W,\n"
    "    __global const float *pos,\n"
    "    __global float *ctx,\n"
    "    __global float *place,\n"
    "    __global float *vel,\n"
    "    __global float *deposit,\n"
    "    float sigma, float dt, int n_in, int n_out)\n"
    "{\n"
    "    float cx = place[0], cy = place[1], cz = place[2];\n"
    "    float vx = vel[0], vy = vel[1], vz = vel[2];\n"
    "    float sig2 = sigma * sigma * 2.0f;\n"
    "    for (int j = 0; j < n_in; ++j) {\n"
    "        float dx = pos[j * 3] - cx;\n"
    "        float dy = pos[j * 3 + 1] - cy;\n"
    "        float dz = pos[j * 3 + 2] - cz;\n"
    "        float dist2 = dx * dx + dy * dy + dz * dz;\n"
    "        float dist = sqrt(dist2 + 1.0e-12f);\n"
    "        float influence = exp(-dist2 / sig2);\n"
    "        float nx = dx / dist, ny = dy / dist, nz = dz / dist;\n"
    "        float vnorm = sqrt(vx * vx + vy * vy + vz * vz) + 1.0e-12f;\n"
    "        float approach = (vx * nx + vy * ny + vz * nz) / vnorm;\n"
    "        float face = approach > 0.0f ? approach : 0.0f;\n"
    "        float ox = W[0 * n_in + j];\n"
    "        float oy = (n_out > 1) ? W[1 * n_in + j] : 0.0f;\n"
    "        float oz = (n_out > 2) ? W[2 * n_in + j] : 0.0f;\n"
    "        float on = sqrt(ox * ox + oy * oy + oz * oz) + 1.0e-12f;\n"
    "        float align = fabs(ox * nx + oy * ny + oz * nz) / on;\n"
    "        float angle = 0.5f + 0.5f * align;\n"
    "        float T = influence * face * angle;\n"
    "        float R = influence * (1.0f - face);\n"
    "        float dotvn = vx * nx + vy * ny + vz * nz;\n"
    "        vx = vx - R * 2.0f * dotvn * nx;\n"
    "        vy = vy - R * 2.0f * dotvn * ny;\n"
    "        vz = vz - R * 2.0f * dotvn * nz;\n"
    "        float near = exp(-dist / (sigma + 1.0e-6f));\n"
    "        for (int i = 0; i < n_out; ++i) {\n"
    "            int group = i / 3;\n"
    "            float detail = exp(-(float)group * (1.0f - near) * 0.08f);\n"
    "            float src = W[(long)i * n_in + j];\n"
    "            ctx[i] = ctx[i] + dt * T * detail * (src - ctx[i]);\n"
    "        }\n"
    "        deposit[j * 4] = T;\n"
    "        deposit[j * 4 + 1] = cx + dx * 0.5f;\n"
    "        deposit[j * 4 + 2] = cy + dy * 0.5f;\n"
    "        deposit[j * 4 + 3] = cz + dz * 0.5f;\n"
    "    }\n"
    "    place[0] = cx + vx * dt;\n"
    "    place[1] = cy + vy * dt;\n"
    "    place[2] = cz + vz * dt;\n"
    "    vel[0] = vx; vel[1] = vy; vel[2] = vz;\n"
    "}\n";

static void die(const char *what, cl_int err) {
    fprintf(stderr, "%s (%d)\n", what, err);
    exit(1);
}

static float nrm(const float *a, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)a[i] * a[i];
    return (float)sqrt(s);
}

static void gram_schmidt(float *a, float *b, float *c, int n) {
    float nb = 0, nc = 0, ab = 0, ac = 0, bc = 0;
    for (int i = 0; i < n; ++i) {
        ab += a[i] * b[i];
        ac += a[i] * c[i];
    }
    float aa = 0;
    for (int i = 0; i < n; ++i) aa += a[i] * a[i];
    aa = sqrtf(aa) + 1e-12f;
    for (int i = 0; i < n; ++i) a[i] /= aa;
    for (int i = 0; i < n; ++i) b[i] -= ab / (aa * aa) * a[i] * aa;
    /* a уже нормирован, ab посчитан до нормировки */
}

static unsigned long checksum(const void *p, size_t n) {
    const unsigned char *b = p;
    unsigned long s = 1469598103934665603ul;
    for (size_t i = 0; i < n; ++i) {
        s ^= b[i];
        s *= 1099511628211ul;
    }
    return s;
}

static cl_device_id radeon(char *name, size_t nname) {
    cl_uint np = 0;
    clGetPlatformIDs(0, NULL, &np);
    cl_platform_id ps[4];
    clGetPlatformIDs(np, ps, NULL);
    for (cl_uint i = 0; i < np; ++i) {
        cl_uint nd = 0;
        clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 0, NULL, &nd);
        cl_device_id ds[8];
        clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, nd, ds, NULL);
        for (cl_uint j = 0; j < nd; ++j) {
            clGetDeviceInfo(ds[j], CL_DEVICE_NAME, nname, name, NULL);
            if (strstr(name, "Radeon")) return ds[j];
        }
    }
    return NULL;
}

static void run_step(cl_command_queue q, cl_kernel k, cl_mem bW, cl_mem bP, cl_mem bC,
                     cl_mem bPlace, cl_mem bV, cl_mem bD, float sigma, float dt,
                     int n_in, int n_out) {
    clSetKernelArg(k, 0, sizeof bW, &bW);
    clSetKernelArg(k, 1, sizeof bP, &bP);
    clSetKernelArg(k, 2, sizeof bC, &bC);
    clSetKernelArg(k, 3, sizeof bPlace, &bPlace);
    clSetKernelArg(k, 4, sizeof bV, &bV);
    clSetKernelArg(k, 5, sizeof bD, &bD);
    clSetKernelArg(k, 6, sizeof sigma, &sigma);
    clSetKernelArg(k, 7, sizeof dt, &dt);
    clSetKernelArg(k, 8, sizeof(int), &n_in);
    clSetKernelArg(k, 9, sizeof(int), &n_out);
    size_t one = 1;
    cl_int err = clEnqueueNDRangeKernel(q, k, 1, NULL, &one, NULL, 0, NULL, NULL);
    if (err) die("шаг", err);
    clFinish(q);
}

/* Два синтетических удара: лобовой и скользящий. */
static void fixture(cl_context ctx, cl_command_queue q, cl_kernel kernel) {
    const int n_out = 6, n_in = 1;
    float W[6], pos[3], ctxv[6], place[3], vel[3], dep[4];
    for (int i = 0; i < 6; ++i) W[i] = 1.f;
    pos[0] = 0; pos[1] = 0; pos[2] = 1;
    for (int i = 0; i < 6; ++i) ctxv[i] = 0;
    place[0] = place[1] = place[2] = 0;
    vel[0] = 0; vel[1] = 0; vel[2] = 1;
    cl_int err = 0;
    cl_mem bW = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof W, W, &err);
    cl_mem bP = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof pos, pos, &err);
    cl_mem bC = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof ctxv, ctxv, &err);
    cl_mem bL = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof place, place, &err);
    cl_mem bV = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof vel, vel, &err);
    cl_mem bD = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof dep, NULL, &err);
    run_step(q, kernel, bW, bP, bC, bL, bV, bD, 1.f, 0.2f, n_in, n_out);
    float head_ctx[6], head_vel[3];
    clEnqueueReadBuffer(q, bC, CL_TRUE, 0, sizeof head_ctx, head_ctx, 0, NULL, NULL);
    clEnqueueReadBuffer(q, bV, CL_TRUE, 0, sizeof head_vel, head_vel, 0, NULL, NULL);

    pos[0] = 1; pos[1] = 0; pos[2] = 0.15f;
    for (int i = 0; i < 6; ++i) ctxv[i] = 0;
    place[0] = place[1] = place[2] = 0;
    vel[0] = 0; vel[1] = 0; vel[2] = 1;
    clEnqueueWriteBuffer(q, bP, CL_TRUE, 0, sizeof pos, pos, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bC, CL_TRUE, 0, sizeof ctxv, ctxv, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bL, CL_TRUE, 0, sizeof place, place, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bV, CL_TRUE, 0, sizeof vel, vel, 0, NULL, NULL);
    run_step(q, kernel, bW, bP, bC, bL, bV, bD, 1.f, 0.2f, n_in, n_out);
    float side_ctx[6], side_vel[3];
    clEnqueueReadBuffer(q, bC, CL_TRUE, 0, sizeof side_ctx, side_ctx, 0, NULL, NULL);
    clEnqueueReadBuffer(q, bV, CL_TRUE, 0, sizeof side_vel, side_vel, 0, NULL, NULL);
    printf("лобовой_впитывание: %.4f\n", nrm(head_ctx, 6));
    printf("скользящий_впитывание: %.4f\n", nrm(side_ctx, 6));
    printf("лобовой_скорость: %.4f %.4f %.4f\n", head_vel[0], head_vel[1], head_vel[2]);
    printf("скользящая_скорость: %.4f %.4f %.4f\n", side_vel[0], side_vel[1], side_vel[2]);
    clReleaseMemObject(bW);
    clReleaseMemObject(bP);
    clReleaseMemObject(bC);
    clReleaseMemObject(bL);
    clReleaseMemObject(bV);
    clReleaseMemObject(bD);
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("нет сцены", -1);
    int32_t n_out = 0, n_in = 0, n_vec = 0;
    fread(&n_out, 4, 1, f);
    fread(&n_in, 4, 1, f);
    fread(&n_vec, 4, 1, f);
    size_t nw = (size_t)n_out * n_in;
    float *W = malloc(nw * sizeof(float));
    fread(W, sizeof(float), nw, f);
    fclose(f);
    unsigned long before = checksum(W, nw * sizeof(float));

    double *energy = calloc((size_t)n_out, sizeof(double));
    for (int i = 0; i < n_out; ++i)
        for (int j = 0; j < n_in; ++j) {
            float v = W[(size_t)i * n_in + j];
            energy[i] += (double)v * v;
        }
    int r0 = 0, r1 = 1, r2 = 2;
    for (int i = 0; i < n_out; ++i) {
        if (energy[i] > energy[r0]) r0 = i;
    }
    for (int i = 0; i < n_out; ++i) {
        if (i != r0 && energy[i] > energy[r1]) r1 = i;
    }
    for (int i = 0; i < n_out; ++i) {
        if (i != r0 && i != r1 && energy[i] > energy[r2]) r2 = i;
    }
    float *a = malloc((size_t)n_in * sizeof(float));
    float *b = malloc((size_t)n_in * sizeof(float));
    float *c = malloc((size_t)n_in * sizeof(float));
    for (int j = 0; j < n_in; ++j) {
        a[j] = W[(size_t)r0 * n_in + j];
        b[j] = W[(size_t)r1 * n_in + j];
        c[j] = W[(size_t)r2 * n_in + j];
    }
    float aa = nrm(a, n_in) + 1e-12f;
    for (int j = 0; j < n_in; ++j) a[j] /= aa;
    float ab = 0;
    for (int j = 0; j < n_in; ++j) ab += a[j] * b[j];
    for (int j = 0; j < n_in; ++j) b[j] -= ab * a[j];
    float bb = nrm(b, n_in) + 1e-12f;
    for (int j = 0; j < n_in; ++j) b[j] /= bb;
    float ac = 0, bc = 0;
    for (int j = 0; j < n_in; ++j) {
        ac += a[j] * c[j];
        bc += b[j] * c[j];
    }
    for (int j = 0; j < n_in; ++j) c[j] -= ac * a[j] + bc * b[j];
    float cc = nrm(c, n_in) + 1e-12f;
    for (int j = 0; j < n_in; ++j) c[j] /= cc;
    float dab = 0, dac = 0, dbc = 0;
    for (int j = 0; j < n_in; ++j) {
        dab += a[j] * b[j];
        dac += a[j] * c[j];
        dbc += b[j] * c[j];
    }
    printf("оси_ряды: %d %d %d\n", r0, r1, r2);
    printf("скалярные_осей: %.3e %.3e %.3e\n", dab, dac, dbc);

    float *pos = malloc((size_t)n_in * 3 * sizeof(float));
    double mean[3] = {0, 0, 0};
    for (int j = 0; j < n_in; ++j) {
        pos[j * 3] = 0;
        pos[j * 3 + 1] = 0;
        pos[j * 3 + 2] = 0;
        for (int t = 0; t < n_in; ++t) {
            float col = W[(size_t)r0 * n_in + t];
            /* позиция — координаты признака в комнате, столбец не вычитается */
        }
        pos[j * 3] = W[(size_t)r0 * n_in + j];
        pos[j * 3 + 1] = W[(size_t)r1 * n_in + j];
        pos[j * 3 + 2] = W[(size_t)r2 * n_in + j];
        /* проекция на ортооси, исходные числа в W остаются */
        float x = 0, y = 0, z = 0;
        for (int t = 0; t < n_in; ++t) {
            (void)t;
        }
        x = 0;
        y = 0;
        z = 0;
        for (int u = 0; u < n_in; ++u) {
            (void)u;
        }
        pos[j * 3] = a[j];
        pos[j * 3 + 1] = b[j];
        pos[j * 3 + 2] = c[j];
        mean[0] += pos[j * 3];
        mean[1] += pos[j * 3 + 1];
        mean[2] += pos[j * 3 + 2];
    }
    mean[0] /= n_in;
    mean[1] /= n_in;
    mean[2] /= n_in;
    double var = 0;
    for (int j = 0; j < n_in; ++j) {
        double dx = pos[j * 3] - mean[0];
        double dy = pos[j * 3 + 1] - mean[1];
        double dz = pos[j * 3 + 2] - mean[2];
        var += dx * dx + dy * dy + dz * dz;
    }
    float sigma = (float)sqrt(var / n_in);
    if (sigma < 1e-4f) sigma = 1e-4f;
    printf("сигма_комнаты: %.4f\n", sigma);

    char name[256];
    cl_device_id device = radeon(name, sizeof name);
    if (!device) die("Radeon не найден", -1);
    printf("устройство: %s\n", name);
    cl_int err = 0;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err) {
        char log[8192];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "%s\n", log);
        die("build", err);
    }
    cl_kernel kernel = clCreateKernel(prog, "shag", &err);
    fixture(ctx, q, kernel);

    float *interior = calloc((size_t)n_out, sizeof(float));
    float place[3] = {(float)mean[0], (float)mean[1], (float)mean[2]};
    float vel[3] = {0, 0, sigma};
    cl_mem bW = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nw * sizeof(float), W, &err);
    cl_mem bP = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)n_in * 3 * sizeof(float), pos, &err);
    cl_mem bC = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (size_t)n_out * sizeof(float), interior, &err);
    cl_mem bL = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof place, place, &err);
    cl_mem bV = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof vel, vel, &err);
    cl_mem bD = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)n_in * 4 * sizeof(float), NULL, &err);

    float *dep = malloc((size_t)n_in * 4 * sizeof(float));
    float *interior2 = malloc((size_t)n_out * sizeof(float));
    double trace = 0;
    int touched = 0;
    for (int s = 0; s < 4; ++s) {
        run_step(q, kernel, bW, bP, bC, bL, bV, bD, sigma, 0.15f, n_in, n_out);
        clEnqueueReadBuffer(q, bD, CL_TRUE, 0, (size_t)n_in * 4 * sizeof(float), dep, 0, NULL, NULL);
        clEnqueueReadBuffer(q, bV, CL_TRUE, 0, sizeof vel, vel, 0, NULL, NULL);
        double step_trace = 0;
        int step_touch = 0;
        for (int j = 0; j < n_in; ++j) {
            step_trace += dep[j * 4];
            if (dep[j * 4] > 1e-8f) step_touch++;
        }
        trace += step_trace;
        if (step_touch > touched) touched = step_touch;
        printf("шаг %d след %.4f касаний %d ход %.4f %.4f %.4f\n",
               s + 1, step_trace, step_touch, vel[0], vel[1], vel[2]);
    }
    clEnqueueReadBuffer(q, bC, CL_TRUE, 0, (size_t)n_out * sizeof(float), interior, 0, NULL, NULL);
    clEnqueueReadBuffer(q, bL, CL_TRUE, 0, sizeof place, place, 0, NULL, NULL);

    memset(interior2, 0, (size_t)n_out * sizeof(float));
    float place0[3] = {(float)mean[0], (float)mean[1], (float)mean[2]};
    float vel0[3] = {0, 0, sigma};
    clEnqueueWriteBuffer(q, bC, CL_TRUE, 0, (size_t)n_out * sizeof(float), interior2, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bL, CL_TRUE, 0, sizeof place0, place0, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, bV, CL_TRUE, 0, sizeof vel0, vel0, 0, NULL, NULL);
    for (int s = 0; s < 4; ++s)
        run_step(q, kernel, bW, bP, bC, bL, bV, bD, sigma, 0.15f, n_in, n_out);
    clEnqueueReadBuffer(q, bC, CL_TRUE, 0, (size_t)n_out * sizeof(float), interior2, 0, NULL, NULL);
    int same = memcmp(interior, interior2, (size_t)n_out * sizeof(float)) == 0;

    float *back = malloc(nw * sizeof(float));
    clEnqueueReadBuffer(q, bW, CL_TRUE, 0, nw * sizeof(float), back, 0, NULL, NULL);
    int kept = memcmp(W, back, nw * sizeof(float)) == 0;

    printf("повтор_побитово: %d\n", same);
    printf("веса_не_изменились: %d\n", kept);
    printf("сумма_до: %lu\n", before);
    printf("сумма_после: %lu\n", checksum(back, nw * sizeof(float)));
    printf("контекст_норма: %.4f\n", nrm(interior, n_out));
    printf("место: %.4f %.4f %.4f\n", place[0], place[1], place[2]);
    printf("ход: %.4f %.4f %.4f\n", vel[0], vel[1], vel[2]);
    printf("след_сумма: %.4f\n", trace);
    printf("касаний: %d из %d\n", touched, n_in);
    printf("байт_внутренностей: %zu\n", nw * sizeof(float));
    (void)gram_schmidt;
    return kept && same ? 0 : 1;
}
