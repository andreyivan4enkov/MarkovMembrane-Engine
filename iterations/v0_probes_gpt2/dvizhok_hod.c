/* Произведение двух плоскостей становится следующим произведением.

Плоскость весов не переписывается.
Плоскость активации после касания другая, и камера сдвигается.
Поэтому следующий шаг перемножает уже другие числа.
Это и есть ход коллизии активации с весами.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NX = 8, NY = 8, NZ = 8, N = NX * NY * NZ, D = 512, M = 32, STEPS = 8 };

static void ranks_of(const float *pz, float camz, int *rank, float *depth) {
    float min_d = 1e9f, max_d = 0.f;
    int in[N];
    for (int i = 0; i < N; ++i) {
        depth[i] = pz[i] - camz;
        in[i] = depth[i] > 0.3f;
        if (in[i] && depth[i] < min_d) min_d = depth[i];
        if (in[i] && depth[i] > max_d) max_d = depth[i];
    }
    float span = max_d - min_d;
    if (span < 1e-4f) span = 1e-4f;
    long sum = 0;
    for (int i = 0; i < N; ++i) {
        if (!in[i]) {
            rank[i] = 0;
            continue;
        }
        float far = (depth[i] - min_d) / span;
        float detail = (1.f - far) * (1.f - far);
        int cells = 1 + (int)((D - 1) * detail);
        if (cells < 1) cells = 1;
        if (cells > D) cells = D;
        rank[i] = cells;
        sum += cells;
    }
    long budget = (long)N * D * 4 / 10;
    while (sum > budget) {
        int victim = -1;
        float far = -1.f;
        for (int i = 0; i < N; ++i)
            if (rank[i] > 1 && depth[i] > far) {
                far = depth[i];
                victim = i;
            }
        if (victim < 0) break;
        int cut = rank[victim] / 2;
        if (cut < 1) cut = 1;
        if (rank[victim] - cut < 1) cut = rank[victim] - 1;
        rank[victim] -= cut;
        sum -= cut;
    }
}

static void one_run(const float *model, const float *pz, double *contacts, int *fronts, int *backs) {
    float act[M];
    for (int k = 0; k < M; ++k) act[k] = 0.01f * (float)(k + 1);
    float camz = 0.f;
    int rank[N];
    float depth[N];
    int front = (NY / 2) * NX + NX / 2;
    int back = (NZ - 1) * NX * NY + front;
    for (int s = 0; s < STEPS; ++s) {
        ranks_of(pz, camz, rank, depth);
        double contact = 0;
        double touch[M];
        for (int k = 0; k < M; ++k) touch[k] = 0;
        long seen = 0;
        for (int o = 0; o < N; ++o) {
            int r = rank[o];
            for (int k = 0; k < r; ++k) {
                float w = model[(size_t)o * D + k];
                float a = act[k % M];
                contact += (double)a * (double)w;
                touch[k % M] += w;
                seen++;
            }
        }
        contacts[s] = contact;
        fronts[s] = rank[front];
        backs[s] = rank[back];
        for (int k = 0; k < M; ++k) {
            float mean = seen ? (float)(touch[k] / (double)(seen / M + 1)) : 0.f;
            act[k] = 0.85f * act[k] + 0.15f * mean;
        }
        camz += 0.45f + 0.00001f * (float)contact;
    }
}

int main(void) {
    float *pz = malloc(sizeof(float) * N);
    float *model = malloc((size_t)N * D * sizeof(float));
    int i = 0;
    for (int z = 0; z < NZ; ++z)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x, ++i)
                pz[i] = 2.f + z * 0.7f;
    for (int o = 0; o < N; ++o)
        for (int k = 0; k < D; ++k)
            model[(size_t)o * D + k] = sinf(0.017f * (float)o + 0.003f * (float)k);
    unsigned long sum0 = 0;
    unsigned char *bytes = (unsigned char *)model;
    for (size_t b = 0; b < (size_t)N * D * sizeof(float); ++b) {
        sum0 ^= bytes[b];
        sum0 *= 1099511628211ul;
    }
    double a[STEPS], b[STEPS];
    int fa[STEPS], fb[STEPS], ba[STEPS], bb[STEPS];
    one_run(model, pz, a, fa, ba);
    one_run(model, pz, b, fb, bb);
    int same = memcmp(a, b, sizeof a) == 0;
    unsigned long sum1 = 0;
    for (size_t t = 0; t < (size_t)N * D * sizeof(float); ++t) {
        sum1 ^= bytes[t];
        sum1 *= 1099511628211ul;
    }
    printf("шаг контакт передний задний\n");
    for (int s = 0; s < STEPS; ++s)
        printf("%d %.6f %d %d\n", s + 1, a[s], fa[s], ba[s]);
    int moved = 0;
    for (int s = 1; s < STEPS; ++s)
        if (a[s] != a[0]) moved = 1;
    printf("контакт_меняется: %d\n", moved);
    printf("повтор_совпал: %d\n", same);
    printf("веса_не_изменились: %d\n", sum0 == sum1);
    printf("передний_с_1_по_8: %d -> %d\n", fa[0], fa[STEPS - 1]);
    printf("задний_с_1_по_8: %d -> %d\n", ba[0], ba[STEPS - 1]);
    return same && moved && sum0 == sum1 ? 0 : 1;
}
