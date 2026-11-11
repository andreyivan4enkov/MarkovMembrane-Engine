/* Перспектива как загрузка в карту.

Полная модель остаётся в обычной памяти.
В видеопамять попадает только то, что лежит на двумерной площадке.
Далёкий видимый объект — одно число. Близкий забирает много ячеек.
Сумма ячеек не больше 4/10 модели: остальные при приближении одного отдают место.
Перемножение, которое остаётся, — это произведение двух видимых плоскостей.
*/

#include <OpenCL/opencl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NX = 16, NY = 16, NZ = 16, N = NX * NY * NZ, D = 8192 };

static const char *SRC =
    "__kernel void chast(\n"
    "    __global const float *a,\n"
    "    __global float *part,\n"
    "    int n)\n"
    "{\n"
    "    int id = get_global_id(0);\n"
    "    int chunks = get_global_size(0);\n"
    "    long begin = ((long)n * id) / chunks;\n"
    "    long end = ((long)n * (id + 1)) / chunks;\n"
    "    float s = 0.0f;\n"
    "    for (long i = begin; i < end; ++i) s = s + a[i];\n"
    "    part[id] = s;\n"
    "}\n";

static void die(const char *what, cl_int err) {
    fprintf(stderr, "%s (%d)\n", what, err);
    exit(1);
}

static void pose_ranks(const float *px, const float *py, const float *pz,
                       float camx, float camy, float camz,
                       int *rank, float *depth) {
    float min_depth = 1e9f, max_depth = 0.f;
    int in[N];
    for (int i = 0; i < N; ++i) {
        float dx = px[i] - camx;
        float dy = py[i] - camy;
        float dz = pz[i] - camz;
        float lateral = sqrtf(dx * dx + dy * dy);
        depth[i] = dz;
        in[i] = (dz > 0.3f && lateral < dz * 0.75f);
        if (in[i] && dz < min_depth) min_depth = dz;
        if (in[i] && dz > max_depth) max_depth = dz;
    }
    long sum = 0;
    float span = max_depth - min_depth;
    if (span < 1e-4f) span = 1e-4f;
    for (int i = 0; i < N; ++i) {
        if (!in[i]) {
            rank[i] = 0;
            continue;
        }
        float far = (depth[i] - min_depth) / span;
        float detail = (1.f - far) * (1.f - far);
        int cells = 1 + (int)((D - 1) * detail);
        if (cells > D) cells = D;
        if (cells < 1) cells = 1;
        rank[i] = cells;
        sum += cells;
    }
    const long budget = (long)N * D * 4 / 10;
    while (sum > budget) {
        int victim = -1;
        float far = -1.f;
        int floor = 1;
        for (int pass = 0; pass < 2 && victim < 0; ++pass) {
            floor = pass == 0 ? 1 : 0;
            for (int i = 0; i < N; ++i) {
                if (rank[i] > floor && depth[i] > far) {
                    far = depth[i];
                    victim = i;
                }
            }
        }
        if (victim < 0) break;
        int cut = rank[victim] / 2;
        if (cut < 1) cut = 1;
        if (rank[victim] - cut < floor) cut = rank[victim] - floor;
        rank[victim] -= cut;
        sum -= cut;
    }
}

static long sum_ranks(const int *rank) {
    long s = 0;
    for (int i = 0; i < N; ++i) s += rank[i];
    return s;
}

static void report_pose(const char *name, const int *rank) {
    int scalars = 0, full = 0, gone = 0, maxr = 0, visible = 0;
    for (int i = 0; i < N; ++i) {
        if (rank[i] == 0) gone++;
        else visible++;
        if (rank[i] == 1) scalars++;
        if (rank[i] == D) full++;
        if (rank[i] > maxr) maxr = rank[i];
    }
    long cells = sum_ranks(rank);
    printf("%s видимых %d точек %d полных %d выгружено %d макс_ячеек %d байт %ld\n",
           name, visible, scalars, full, gone, maxr, cells * (long)sizeof(float));
}

int main(void) {
    float *px = malloc(sizeof(float) * N);
    float *py = malloc(sizeof(float) * N);
    float *pz = malloc(sizeof(float) * N);
    int i = 0;
    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x, ++i) {
                px[i] = ((x + 0.5f) / NX - 0.5f) * 4.f;
                py[i] = ((y + 0.5f) / NY - 0.5f) * 4.f;
                pz[i] = 2.f + z * 0.8f;
            }
    /* Модель на хосте: ячейка (объект, k) опознаётся. На карту её не кладём целиком. */
    float *model = malloc((size_t)N * D * sizeof(float));
    if (!model) die("нет памяти под модель", -1);
    for (int o = 0; o < N; ++o)
        for (int k = 0; k < D; ++k)
            model[(size_t)o * D + k] = (float)(o + 1) * 0.001f + (float)k * 1.0e-6f;

    int *rank_a = calloc(N, sizeof(int));
    int *rank_b = calloc(N, sizeof(int));
    float *depth = malloc(sizeof(float) * N);
    pose_ranks(px, py, pz, 0, 0, 0, rank_a, depth);

    /* Подходим к заднему центральному объекту. Прежние ближние остаются за спиной. */
    int back = (NZ - 1) * NX * NY + (NY / 2) * NX + NX / 2;
    pose_ranks(px, py, pz, px[back], py[back], pz[back] - 0.5f, rank_b, depth);

    long full_cells = (long)N * D;
    long budget = full_cells * 4 / 10;
    long cells_a = sum_ranks(rank_a);
    long cells_b = sum_ranks(rank_b);
    printf("объектов: %d\n", N);
    printf("ячеек_в_объекте: %d\n", D);
    printf("полная_модель_байт: %ld\n", full_cells * (long)sizeof(float));
    printf("бюджет_карты_байт: %ld\n", budget * (long)sizeof(float));
    printf("поза_А_байт: %ld\n", cells_a * (long)sizeof(float));
    printf("поза_Б_байт: %ld\n", cells_b * (long)sizeof(float));
    report_pose("А", rank_a);
    report_pose("Б", rank_b);
    int front = (NY / 2) * NX + NX / 2;
    printf("передний_объект: %d ячеек_А %d ячеек_Б %d\n", front, rank_a[front], rank_b[front]);
    printf("задний_объект: %d ячеек_А %d ячеек_Б %d\n", back, rank_a[back], rank_b[back]);
    printf("произведение_плоскостей_А: %ld\n", cells_a);
    printf("полное_перемножение: %ld\n", full_cells);
    printf("укладывается_А: %d\n", cells_a <= budget);
    printf("укладывается_Б: %d\n", cells_b <= budget);

    float *pack = malloc((size_t)cells_a * sizeof(float));
    long cursor = 0;
    for (int o = 0; o < N; ++o)
        for (int k = 0; k < rank_a[o]; ++k)
            pack[cursor++] = model[(size_t)o * D + k];

    char name[256];
    cl_uint np = 0;
    clGetPlatformIDs(0, NULL, &np);
    cl_platform_id ps[4];
    clGetPlatformIDs(np, ps, NULL);
    cl_device_id device = NULL;
    for (cl_uint p = 0; p < np && !device; ++p) {
        cl_uint nd = 0;
        clGetDeviceIDs(ps[p], CL_DEVICE_TYPE_GPU, 0, NULL, &nd);
        cl_device_id ds[8];
        clGetDeviceIDs(ps[p], CL_DEVICE_TYPE_GPU, nd, ds, NULL);
        for (cl_uint j = 0; j < nd; ++j) {
            clGetDeviceInfo(ds[j], CL_DEVICE_NAME, sizeof name, name, NULL);
            if (strstr(name, "Radeon")) device = ds[j];
        }
    }
    if (!device) die("Radeon не найден", -1);
    printf("устройство: %s\n", name);

    cl_int err = 0;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
    err = clBuildProgram(prog, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err) {
        char log[4096];
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "%s\n", log);
        die("build", err);
    }
    cl_kernel kernel = clCreateKernel(prog, "chast", &err);
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                (size_t)cells_a * sizeof(float), pack, &err);
    if (err) die("буфер", err);
    const int chunks = 256;
    cl_mem part = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, chunks * sizeof(float), NULL, &err);
    int npack = (int)cells_a;
    clSetKernelArg(kernel, 0, sizeof buf, &buf);
    clSetKernelArg(kernel, 1, sizeof part, &part);
    clSetKernelArg(kernel, 2, sizeof npack, &npack);
    size_t global = chunks;
    err = clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err) die("kernel", err);
    clFinish(q);
    float parts[256];
    clEnqueueReadBuffer(q, part, CL_TRUE, 0, sizeof parts, parts, 0, NULL, NULL);
    double gpu = 0, cpu = 0;
    for (int c = 0; c < chunks; ++c) gpu += parts[c];
    for (long t = 0; t < cells_a; ++t) cpu += pack[t];
    printf("буфер_карты_байт: %ld\n", cells_a * (long)sizeof(float));
    printf("буфер_меньше_модели: %d\n", cells_a < full_cells);
    printf("сумма_карты: %.6f\n", gpu);
    printf("сумма_видимого: %.6f\n", cpu);
    printf("суммы_совпали: %d\n", fabs(gpu - cpu) / (fabs(cpu) + 1.0) < 1e-5);
    return 0;
}
