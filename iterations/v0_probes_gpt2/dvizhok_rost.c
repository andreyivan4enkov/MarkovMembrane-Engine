/* Рост от ядра, без переписывания весов.

Дальний признак в кадре — одна точка (среднее всего столбца).
Ближе он становится крупным объектом (среднее блока из 16 выходов).
У ядра он дорастает до полного столбца.
Переход плавный: точка + a*(блок-точка) + b*(вес-блок).
Если a и b равны 1 у всех, сумма совпадает с обычным умножением.
Буфер весов только читается.
*/

#include <OpenCL/opencl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { BLOCK = 16 };

static const char *SRC =
    "__kernel void rost(\n"
    "    __global const float *W,\n"
    "    __global const float *x,\n"
    "    __global const float *blocks,\n"
    "    __global const float *point,\n"
    "    __global const int *body,\n"
    "    __global const float *a_body,\n"
    "    __global const int *fine,\n"
    "    __global const float *a_fine,\n"
    "    __global float *y,\n"
    "    float sky, int n_in, int n_body, int n_fine, int n_blocks)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    int b = i / 16;\n"
    "    float acc = sky;\n"
    "    for (int k = 0; k < n_body; ++k) {\n"
    "        int j = body[k];\n"
    "        float delta = blocks[(long)j * n_blocks + b] - point[j];\n"
    "        acc = acc + x[j] * a_body[k] * delta;\n"
    "    }\n"
    "    for (int k = 0; k < n_fine; ++k) {\n"
    "        int j = fine[k];\n"
    "        float delta = W[(long)i * n_in + j] - blocks[(long)j * n_blocks + b];\n"
    "        acc = acc + x[j] * a_fine[k] * delta;\n"
    "    }\n"
    "    y[i] = acc;\n"
    "}\n";

static void die(const char *what, cl_int err) {
    fprintf(stderr, "%s (%d)\n", what, err);
    exit(1);
}

static int side_of(int n) {
    int s = 1;
    while (s * s * s < n) s++;
    return s;
}

static float clampf(float v, float a, float b) {
    if (v < a) return a;
    if (v > b) return b;
    return v;
}

static float smoothstep(float e0, float e1, float x) {
    float u = clampf((x - e0) / (e1 - e0), 0.f, 1.f);
    return u * u * (3.f - 2.f * u);
}

static void cell_of(int j, int side, int *x, int *y, int *z) {
    *x = j % side;
    *y = (j / side) % side;
    *z = j / (side * side);
}

static float lattice_dist(int a, int b, int side) {
    int ax, ay, az, bx, by, bz;
    cell_of(a, side, &ax, &ay, &az);
    cell_of(b, side, &bx, &by, &bz);
    float dx = (float)(ax - bx), dy = (float)(ay - by), dz = (float)(az - bz);
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

static int core_of(const float *x, int n) {
    int best = 0;
    float m = fabsf(x[0]);
    for (int j = 1; j < n; ++j) {
        float v = fabsf(x[j]);
        if (v > m) {
            m = v;
            best = j;
        }
    }
    return best;
}

static float closeness(float dist, float radius) {
    if (dist == 0.f) return 1.f;
    if (radius <= 0.f) return 0.f;
    return 1.f - smoothstep(0.f, radius, dist);
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

static void self_check(void) {
    int n_out = 16, n_in = 4, n_blocks = 1;
    float W[64], x[4] = {1.f, -2.f, 0.5f, 0.25f};
    for (int i = 0; i < 64; ++i) W[i] = (float)((i % 7) - 3) * 0.1f;
    float point[4] = {0}, block[4] = {0};
    for (int j = 0; j < n_in; ++j) {
        float s = 0.f;
        for (int i = 0; i < n_out; ++i) s += W[i * n_in + j];
        point[j] = s / (float)n_out;
        block[j] = point[j];
    }
    float full[16] = {0}, grown[16] = {0}, far[16] = {0};
    float sky = 0.f;
    for (int j = 0; j < n_in; ++j) sky += x[j] * point[j];
    for (int i = 0; i < n_out; ++i) {
        float acc = 0.f;
        for (int j = 0; j < n_in; ++j) acc += W[i * n_in + j] * x[j];
        full[i] = acc;
        grown[i] = sky;
        far[i] = sky;
        for (int j = 0; j < n_in; ++j) {
            float delta = W[i * n_in + j] - block[j];
            grown[i] += x[j] * 1.f * delta;
        }
    }
    for (int i = 0; i < n_out; ++i) {
        if (fabsf(grown[i] - full[i]) > 1e-5f) die("полный рост не совпал с умножением", i);
        if (fabsf(far[i] - sky) > 1e-6f) die("дальняя точка размазалась", i);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    self_check();
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("нет сцены", -1);
    int32_t n_out = 0, n_in = 0, n_vec = 0;
    fread(&n_out, 4, 1, f);
    fread(&n_in, 4, 1, f);
    fread(&n_vec, 4, 1, f);
    if (n_out % BLOCK) die("выходы не делятся на блок", n_out);
    int n_blocks = n_out / BLOCK;
    size_t nw = (size_t)n_out * (size_t)n_in;
    float *W = malloc(nw * sizeof(float));
    float *X = malloc((size_t)n_vec * n_in * sizeof(float));
    fread(W, sizeof(float), nw, f);
    fread(X, sizeof(float), (size_t)n_vec * n_in, f);
    fclose(f);
    unsigned long before = checksum(W, nw * sizeof(float));

    float *point = calloc((size_t)n_in, sizeof(float));
    float *blocks = malloc((size_t)n_in * n_blocks * sizeof(float));
    for (int j = 0; j < n_in; ++j) {
        double all = 0;
        for (int b = 0; b < n_blocks; ++b) {
            double s = 0;
            for (int k = 0; k < BLOCK; ++k) {
                int i = b * BLOCK + k;
                s += W[(size_t)i * n_in + j];
            }
            blocks[(size_t)j * n_blocks + b] = (float)(s / BLOCK);
            all += s;
        }
        point[j] = (float)(all / n_out);
    }

    int side = side_of(n_in);
    cl_uint np = 0;
    clGetPlatformIDs(0, NULL, &np);
    cl_platform_id ps[4];
    clGetPlatformIDs(np, ps, NULL);
    cl_device_id device = NULL;
    char dname[256] = {0};
    for (cl_uint pi = 0; pi < np && !device; ++pi) {
        cl_uint nd = 0;
        clGetDeviceIDs(ps[pi], CL_DEVICE_TYPE_GPU, 0, NULL, &nd);
        cl_device_id ds[8];
        clGetDeviceIDs(ps[pi], CL_DEVICE_TYPE_GPU, nd, ds, NULL);
        for (cl_uint j = 0; j < nd; ++j) {
            clGetDeviceInfo(ds[j], CL_DEVICE_NAME, sizeof dname, dname, NULL);
            if (strstr(dname, "Radeon")) device = ds[j];
        }
    }
    if (!device) die("Radeon не найден", -1);
    printf("устройство: %s\n", dname);

    cl_int err = 0;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err) {
        char log[8192];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "%s\n", log);
        die("build", err);
    }
    cl_kernel kernel = clCreateKernel(prog, "rost", &err);
    cl_mem bW = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nw * sizeof(float), W, &err);
    cl_mem bX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size_t)n_in * sizeof(float), NULL, &err);
    cl_mem bBlocks = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    (size_t)n_in * n_blocks * sizeof(float), blocks, &err);
    cl_mem bPoint = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (size_t)n_in * sizeof(float), point, &err);
    cl_mem bBody = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size_t)n_in * sizeof(int), NULL, &err);
    cl_mem bFine = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size_t)n_in * sizeof(int), NULL, &err);
    cl_mem bAB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size_t)n_in * sizeof(float), NULL, &err);
    cl_mem bAF = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size_t)n_in * sizeof(float), NULL, &err);
    cl_mem bY = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)n_out * sizeof(float), NULL, &err);

    int *body = malloc((size_t)n_in * sizeof(int));
    int *fine = malloc((size_t)n_in * sizeof(int));
    float *ab = malloc((size_t)n_in * sizeof(float));
    float *af = malloc((size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_out * sizeof(float));
    float *y2 = malloc((size_t)n_out * sizeof(float));
    float radii[] = {0.f, 4.f, 8.f, 14.f, 200.f};
    long classical = (long)n_out * n_in;

    printf("радиус ядро_точка крупные полные умножения_доля ошибка мс биты\n");
    for (unsigned r = 0; r < sizeof radii / sizeof radii[0]; ++r) {
        float radius = radii[r];
        double err_sum = 0, ms_sum = 0;
        int bits_ok = 1, runs = 0;
        long mul_example = 0, load_example = 0;
        int n_point = 0, n_body = 0, n_fine = 0;
        for (int v = 0; v < n_vec; ++v) {
            const float *x = X + (size_t)v * n_in;
            int core = core_of(x, n_in);
            n_body = 0;
            n_fine = 0;
            n_point = 0;
            float sky = 0.f;
            for (int j = 0; j < n_in; ++j) {
                float t = closeness(lattice_dist(j, core, side), radius);
                float body_w = smoothstep(0.05f, 0.50f, t);
                float fine_w = smoothstep(0.50f, 0.95f, t);
                sky += x[j] * point[j];
                if (body_w > 0.f) {
                    body[n_body] = j;
                    ab[n_body] = body_w;
                    n_body++;
                }
                if (fine_w > 0.f) {
                    fine[n_fine] = j;
                    af[n_fine] = fine_w;
                    n_fine++;
                }
                if (body_w == 0.f && fine_w == 0.f) n_point++;
            }
            clEnqueueWriteBuffer(q, bX, CL_TRUE, 0, (size_t)n_in * sizeof(float), x, 0, NULL, NULL);
            clEnqueueWriteBuffer(q, bBody, CL_TRUE, 0, (size_t)n_body * sizeof(int), body, 0, NULL, NULL);
            clEnqueueWriteBuffer(q, bFine, CL_TRUE, 0, (size_t)n_fine * sizeof(int), fine, 0, NULL, NULL);
            clEnqueueWriteBuffer(q, bAB, CL_TRUE, 0, (size_t)n_body * sizeof(float), ab, 0, NULL, NULL);
            clEnqueueWriteBuffer(q, bAF, CL_TRUE, 0, (size_t)n_fine * sizeof(float), af, 0, NULL, NULL);
            clSetKernelArg(kernel, 0, sizeof bW, &bW);
            clSetKernelArg(kernel, 1, sizeof bX, &bX);
            clSetKernelArg(kernel, 2, sizeof bBlocks, &bBlocks);
            clSetKernelArg(kernel, 3, sizeof bPoint, &bPoint);
            clSetKernelArg(kernel, 4, sizeof bBody, &bBody);
            clSetKernelArg(kernel, 5, sizeof bAB, &bAB);
            clSetKernelArg(kernel, 6, sizeof bFine, &bFine);
            clSetKernelArg(kernel, 7, sizeof bAF, &bAF);
            clSetKernelArg(kernel, 8, sizeof bY, &bY);
            clSetKernelArg(kernel, 9, sizeof sky, &sky);
            clSetKernelArg(kernel, 10, sizeof(int), &n_in);
            clSetKernelArg(kernel, 11, sizeof(int), &n_body);
            clSetKernelArg(kernel, 12, sizeof(int), &n_fine);
            clSetKernelArg(kernel, 13, sizeof(int), &n_blocks);
            size_t global = (size_t)n_out, local = 64;
            cl_event ev;
            err = clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
            if (err) die("kernel", err);
            clFinish(q);
            cl_ulong t0 = 0, t1 = 0;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
            clReleaseEvent(ev);
            ms_sum += (double)(t1 - t0) / 1e6;
            clEnqueueReadBuffer(q, bY, CL_TRUE, 0, (size_t)n_out * sizeof(float), y, 0, NULL, NULL);
            clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            clFinish(q);
            clEnqueueReadBuffer(q, bY, CL_TRUE, 0, (size_t)n_out * sizeof(float), y2, 0, NULL, NULL);
            if (memcmp(y, y2, (size_t)n_out * sizeof(float)) != 0) bits_ok = 0;

            double num = 0, den = 0;
            for (int i = 0; i < n_out; ++i) {
                int b = i / BLOCK;
                float acc = sky;
                for (int k = 0; k < n_body; ++k) {
                    int j = body[k];
                    acc += x[j] * ab[k] * (blocks[(size_t)j * n_blocks + b] - point[j]);
                }
                for (int k = 0; k < n_fine; ++k) {
                    int j = fine[k];
                    acc += x[j] * af[k] * (W[(size_t)i * n_in + j] - blocks[(size_t)j * n_blocks + b]);
                }
                if (y[i] != acc) bits_ok = 0;
                float exact = 0.f;
                for (int j = 0; j < n_in; ++j) exact += W[(size_t)i * n_in + j] * x[j];
                double d = (double)y[i] - (double)exact;
                num += d * d;
                den += (double)exact * (double)exact;
            }
            err_sum += sqrt(num / (den + 1e-30));
            mul_example = (long)n_in + (long)n_body * n_blocks + (long)n_fine * n_out;
            load_example = (long)n_fine * n_out + (long)n_body * n_blocks + n_in;
            runs++;
        }
        printf("%.0f %d %d %d %.4f %.4f %.3f %d\n",
               radius, n_point, n_body, n_fine,
               (double)mul_example / (double)classical,
               (double)load_example / (double)classical,
               err_sum / runs, bits_ok);
        printf("  мс %.4f\n", ms_sum / runs);
    }

    float *back = malloc(nw * sizeof(float));
    clEnqueueReadBuffer(q, bW, CL_TRUE, 0, nw * sizeof(float), back, 0, NULL, NULL);
    printf("веса_не_изменились: %d\n", memcmp(W, back, nw * sizeof(float)) == 0);
    printf("сумма_до: %lu\n", before);
    printf("сумма_после: %lu\n", checksum(back, nw * sizeof(float)));
    printf("байт_весов: %zu\n", nw * sizeof(float));
    printf("байт_крупных_объектов: %zu\n", (size_t)n_in * n_blocks * sizeof(float));
    return 0;
}
