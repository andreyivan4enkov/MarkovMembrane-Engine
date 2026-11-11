/* Детерминированный движок на Radeon.

Мир хранит каждый вес как есть. Позиция на решётке обратима:
из координат клетки восстанавливается номер признака.
Точный кадр перемножает полные наборы в фиксированном порядке.
Перспектива меняет только кадр: вклад источника умножается на near/z.
Буфер весов после обоих кадров читается обратно и сравнивается побайтно.
*/

#include <OpenCL/opencl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SRC =
    "__kernel void kadr(\n"
    "    __global const float *W,\n"
    "    __global const float *x,\n"
    "    __global float *y,\n"
    "    int n_in, int side, int perspective)\n"
    "{\n"
    "    int i = get_global_id(0);\n"
    "    float acc = 0.0f;\n"
    "    int side2 = side * side;\n"
    "    for (int j = 0; j < n_in; ++j) {\n"
    "        float scale = 1.0f;\n"
    "        if (perspective) {\n"
    "            int iz = j / side2;\n"
    "            float depth = (float)iz + (float)side;\n"
    "            scale = (float)side / depth;\n"
    "        }\n"
    "        acc = acc + W[(long)i * n_in + j] * x[j] * scale;\n"
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

static int unlattice(int x, int y, int z, int side) {
    return x + y * side + z * side * side;
}

static float cpu_row(const float *W, const float *x, int i, int n_in, int side, int perspective) {
    float acc = 0.f;
    int side2 = side * side;
    for (int j = 0; j < n_in; ++j) {
        float scale = 1.f;
        if (perspective) {
            int iz = j / side2;
            float depth = (float)iz + (float)side;
            scale = (float)side / depth;
        }
        acc = acc + W[(size_t)i * n_in + j] * x[j] * scale;
    }
    return acc;
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: dvizhok сцена.bin\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("нет файла сцены", -1);
    int32_t n_out = 0, n_in = 0, n_vec = 0;
    if (fread(&n_out, 4, 1, f) != 1 || fread(&n_in, 4, 1, f) != 1 || fread(&n_vec, 4, 1, f) != 1)
        die("заголовок", -1);
    size_t nw = (size_t)n_out * (size_t)n_in;
    float *W = malloc(nw * sizeof(float));
    float *X = malloc((size_t)n_vec * n_in * sizeof(float));
    if (fread(W, sizeof(float), nw, f) != nw) die("веса", -1);
    if (fread(X, sizeof(float), (size_t)n_vec * n_in, f) != (size_t)n_vec * n_in) die("входы", -1);
    fclose(f);

    int side = side_of(n_in);
    for (int j = 0; j < n_in; ++j) {
        int z = j / (side * side);
        int y = (j / side) % side;
        int x = j % side;
        if (unlattice(x, y, z, side) != j) die("решётка необратима", j);
    }

    cl_uint np = 0;
    clGetPlatformIDs(0, NULL, &np);
    cl_platform_id ps[4];
    clGetPlatformIDs(np, ps, NULL);
    cl_device_id device = NULL;
    char dname[256] = {0};
    cl_ulong mem = 0;
    for (cl_uint i = 0; i < np && !device; ++i) {
        cl_uint nd = 0;
        clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 0, NULL, &nd);
        cl_device_id ds[8];
        clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, nd, ds, NULL);
        for (cl_uint j = 0; j < nd; ++j) {
            clGetDeviceInfo(ds[j], CL_DEVICE_NAME, sizeof dname, dname, NULL);
            if (strstr(dname, "Radeon")) {
                device = ds[j];
                clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, NULL);
                break;
            }
        }
    }
    if (!device) die("Radeon не найден", -1);
    printf("устройство: %s\n", dname);
    printf("память_мб: %llu\n", mem / (1024ull * 1024ull));
    printf("сцена: выходы=%d входы=%d векторы=%d сторона_решётки=%d\n", n_out, n_in, n_vec, side);

    cl_int err = 0;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err) die("context", err);
    cl_command_queue q = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err) die("queue", err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err) {
        char log[8192];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "%s\n", log);
        die("build", err);
    }
    cl_kernel kernel = clCreateKernel(prog, "kadr", &err);
    cl_mem bW = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nw * sizeof(float), W, &err);
    cl_mem bX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (size_t)n_in * sizeof(float), NULL, &err);
    cl_mem bY = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)n_out * sizeof(float), NULL, &err);
    unsigned long before = checksum(W, nw * sizeof(float));

    float *y_gpu = malloc((size_t)n_out * sizeof(float));
    float *y_gpu2 = malloc((size_t)n_out * sizeof(float));
    float *y_exact0 = calloc((size_t)n_out, sizeof(float));
    float *y_view0 = calloc((size_t)n_out, sizeof(float));
    double worst_exact = 0, worst_view = 0;
    int exact_bits = 1, view_bits = 1, world_bits = 1;

    for (int v = 0; v < n_vec; ++v) {
        const float *x = X + (size_t)v * n_in;
        err = clEnqueueWriteBuffer(q, bX, CL_TRUE, 0, (size_t)n_in * sizeof(float), x, 0, NULL, NULL);
        for (int perspective = 0; perspective <= 1; ++perspective) {
            clSetKernelArg(kernel, 0, sizeof bW, &bW);
            clSetKernelArg(kernel, 1, sizeof bX, &bX);
            clSetKernelArg(kernel, 2, sizeof bY, &bY);
            clSetKernelArg(kernel, 3, sizeof(int), &n_in);
            clSetKernelArg(kernel, 4, sizeof(int), &side);
            clSetKernelArg(kernel, 5, sizeof(int), &perspective);
            size_t global = (size_t)n_out;
            size_t local = 64;
            err = clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            if (err) die("kernel", err);
            clFinish(q);
            clEnqueueReadBuffer(q, bY, CL_TRUE, 0, (size_t)n_out * sizeof(float), y_gpu, 0, NULL, NULL);
            err = clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
            clFinish(q);
            clEnqueueReadBuffer(q, bY, CL_TRUE, 0, (size_t)n_out * sizeof(float), y_gpu2, 0, NULL, NULL);
            if (memcmp(y_gpu, y_gpu2, (size_t)n_out * sizeof(float)) != 0) {
                if (perspective) view_bits = 0;
                else exact_bits = 0;
            }
            double max_abs = 0;
            for (int i = 0; i < n_out; ++i) {
                float ref = cpu_row(W, x, i, n_in, side, perspective);
                double d = fabs((double)y_gpu[i] - (double)ref);
                if (d > max_abs) max_abs = d;
            }
            if (perspective) {
                if (max_abs > worst_view) worst_view = max_abs;
                if (v == 0) memcpy(y_view0, y_gpu, (size_t)n_out * sizeof(float));
            } else if (max_abs > worst_exact) {
                worst_exact = max_abs;
            }
            if (!perspective && v == 0) memcpy(y_exact0, y_gpu, (size_t)n_out * sizeof(float));
            printf("вектор %d %s max_abs=%.3e\n", v, perspective ? "перспектива" : "точный", max_abs);
        }
    }

    float *Wback = malloc(nw * sizeof(float));
    clEnqueueReadBuffer(q, bW, CL_TRUE, 0, nw * sizeof(float), Wback, 0, NULL, NULL);
    if (memcmp(W, Wback, nw * sizeof(float)) != 0) world_bits = 0;
    unsigned long after = checksum(Wback, nw * sizeof(float));

    cl_event ev;
    int zero = 0;
    const float *x0 = X;
    clEnqueueWriteBuffer(q, bX, CL_TRUE, 0, (size_t)n_in * sizeof(float), x0, 0, NULL, NULL);
    clSetKernelArg(kernel, 0, sizeof bW, &bW);
    clSetKernelArg(kernel, 1, sizeof bX, &bX);
    clSetKernelArg(kernel, 2, sizeof bY, &bY);
    clSetKernelArg(kernel, 3, sizeof(int), &n_in);
    clSetKernelArg(kernel, 4, sizeof(int), &side);
    clSetKernelArg(kernel, 5, sizeof(int), &zero);
    size_t global = (size_t)n_out;
    size_t local = 64;
    double ms = 0;
    int reps = 30;
    for (int r = 0; r < reps; ++r) {
        clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        clWaitForEvents(1, &ev);
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
        ms += (double)(t1 - t0) / 1e6;
        clReleaseEvent(ev);
    }

    printf("точный_совпал_побитово: %d\n", exact_bits);
    printf("перспектива_совпала_побитово: %d\n", view_bits);
    printf("веса_не_изменились: %d\n", world_bits);
    printf("контрольная_сумма_до: %lu\n", before);
    printf("контрольная_сумма_после: %lu\n", after);
    double view_gap = 0;
    for (int i = 0; i < n_out; ++i) {
        double d = fabs((double)y_exact0[i] - (double)y_view0[i]);
        if (d > view_gap) view_gap = d;
    }
    int scale_ok = 1;
    for (int j = 0; j < n_in; ++j) {
        int iz = j / (side * side);
        float scale = (float)side / ((float)iz + (float)side);
        if (!(scale > 0.f)) scale_ok = 0;
    }
    printf("худший_точный: %.3e\n", worst_exact);
    printf("кадры_различаются: %.3e\n", view_gap);
    printf("масштаб_перспективы_обратим: %d\n", scale_ok);
    printf("худшая_перспектива: %.3e\n", worst_view);
    printf("мс_точный_кадр: %.4f\n", ms / reps);
    printf("байт_весов: %zu\n", nw * sizeof(float));
    return world_bits && exact_bits && view_bits ? 0 : 1;
}
