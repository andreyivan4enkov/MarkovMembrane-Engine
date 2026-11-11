/* ===================================================================================== *
 * MARKOVMEMBRANE ENGINE (v11 — Native Metal 3 & MSL Shaders on AMD Radeon Pro 5500M)
 * Полный отказ от PyTorch и от побайтового дробления!
 *
 * Что делает этот движок:
 * 1. Читает реальный англоязычный корпус WikiText-2 и разбирает его на ЦЕЛЫЕ СЛОВА (понятия).
 * 2. Исполняет все обучение и физику 100% на шейдерах дискретной видеокарты AMD Radeon Pro 5500M (4 ГБ):
 *    - Упаковка естественного языка на уровне слов: Субъект (S) — Модулятор (M, 3D-поворот Родрига SO(3)) — Объект (O).
 *    - В центре 3D-комнаты (X, Y, Z) находится замкнутая полигональная фигура (сетка 12x12 = 144 вершины / 288 треугольников).
 *    - Голографический принцип (A ~ V^(2/3)): объем шара V(t) и площадь полигонов dA_poly растут от активной внутренней сложности
 *      и двусторонней интерференции между внутренним ожиданием W_int и внешним словом контекста W_ext.
 *    - На каждом полигоне рельефа работают 4 канала света float4(R, G, B, L) (процедурные цвета RGB + Яркость L).
 *    - Вокруг фигуры в 3D-комнате стоят N_OBJ = 128 узлов памяти и N_ROOM = 32 кристалла внешних навыков (0 байт KV-кэша!).
 * 3. Прямо во время реального обучения шейдер рендеринга на GPU (`render_3d_and_2d_frame`) рисует каждый шаг
 *    в буфер кадра 1280x720:
 *    - СЛЕВА (Перспективный 3D-вид комнаты под углом): видны оси X, Y, Z, границы 3D-комнаты,
 *      полигональная рельефная фигура с горами и впадинами в цветах RGBL, 3D-узлы памяти, внешние кристаллы
 *      и реальные лучи коллизий в момент перезаписи весов!
 *    - СПРАВА (2D-проекция оболочки и осей): развертка 2D-мембраны коллизии (u, v), карта высот рельефа,
 *      шкала осей, активные грани и живой график обучения (падение ошибки и рост кристаллизованной памяти).
 * ===================================================================================== */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define VOCAB_WORDS 2048      /* Словарь из 2048 самых частых целых слов WikiText-2 */
#define SIDE 8                /* 8x8 = 64 полигональных сектора на 2D-оболочке фигуры */
#define D_MODEL (SIDE * SIDE) /* 64 измерения = 64 полигона оболочки */
#define N_OBJ 128             /* 128 3D-узлов в комнате (X, Y, Z) */
#define N_ROOM 32             /* 32 внешних кристалла навыков в комнате (без KV-кэша) */
#define BATCH_WORDS 128       /* Пачка из 128 целых слов на кадр обучения */

#define IMG_W 1280
#define IMG_H 720

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

/* Простой хеш-словарь для токенизации текста на ЦЕЛЫЕ СЛОВА на чистом Си */
typedef struct {
    char word[32];
    int count;
    int id;
} WordEntry;

#define HASH_SIZE 65536
static WordEntry g_hash[HASH_SIZE];
static char g_id_to_word[VOCAB_WORDS][32];

static uint32_t hash_str(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h ^= (uint8_t)(*s++);
        h *= 16777619u;
    }
    return h;
}

static int cmp_entries(const void *a, const void *b) {
    const WordEntry *ea = (const WordEntry *)a;
    const WordEntry *eb = (const WordEntry *)b;
    return eb->count - ea->count;
}

static int *tokenize_corpus_to_words(const char *raw_text, long len, int *out_n_words) {
    memset(g_hash, 0, sizeof(g_hash));
    char buf[32];
    int blen = 0;

    /* Проход 1: подсчет частот целых слов */
    for (long i = 0; i < len; ++i) {
        char c = raw_text[i];
        if (isalnum((unsigned char)c)) {
            if (blen < 30) buf[blen++] = (char)tolower((unsigned char)c);
        } else if (blen > 0) {
            buf[blen] = '\0';
            uint32_t idx = hash_str(buf) & (HASH_SIZE - 1);
            while (g_hash[idx].count > 0 && strcmp(g_hash[idx].word, buf) != 0) {
                idx = (idx + 1) & (HASH_SIZE - 1);
            }
            if (g_hash[idx].count == 0) strcpy(g_hash[idx].word, buf);
            g_hash[idx].count++;
            blen = 0;
        }
    }

    /* Отбираем VOCAB_WORDS - 1 самых частых слов (0 = <unk>) */
    WordEntry *sorted = (WordEntry *)malloc(sizeof(WordEntry) * HASH_SIZE);
    int unique = 0;
    for (int i = 0; i < HASH_SIZE; ++i) {
        if (g_hash[i].count > 0) sorted[unique++] = g_hash[i];
    }
    qsort(sorted, (size_t)unique, sizeof(WordEntry), cmp_entries);

    strcpy(g_id_to_word[0], "<unk>");
    int limit = (unique < VOCAB_WORDS - 1) ? unique : (VOCAB_WORDS - 1);
    for (int i = 0; i < limit; ++i) {
        strcpy(g_id_to_word[i + 1], sorted[i].word);
        uint32_t idx = hash_str(sorted[i].word) & (HASH_SIZE - 1);
        while (strcmp(g_hash[idx].word, sorted[i].word) != 0) idx = (idx + 1) & (HASH_SIZE - 1);
        g_hash[idx].id = i + 1;
    }
    free(sorted);

    /* Проход 2: перевод текста в поток ID целых слов */
    int cap = (int)(len / 4 + 1024);
    int *tokens = (int *)malloc(sizeof(int) * (size_t)cap);
    int n_tok = 0;
    blen = 0;
    for (long i = 0; i < len; ++i) {
        char c = raw_text[i];
        if (isalnum((unsigned char)c)) {
            if (blen < 30) buf[blen++] = (char)tolower((unsigned char)c);
        } else if (blen > 0) {
            buf[blen] = '\0';
            uint32_t idx = hash_str(buf) & (HASH_SIZE - 1);
            int wid = 0;
            while (g_hash[idx].count > 0) {
                if (strcmp(g_hash[idx].word, buf) == 0) {
                    wid = g_hash[idx].id;
                    break;
                }
                idx = (idx + 1) & (HASH_SIZE - 1);
            }
            tokens[n_tok++] = wid;
            blen = 0;
        }
    }
    *out_n_words = n_tok;
    printf("Корпус разобран на ЦЕЛЫЕ СЛОВА: всего слов = %d, уникальных = %d, активный словарь = %d слов\n",
           n_tok, unique, VOCAB_WORDS);
    printf("Топ-10 слов словаря: ");
    for (int i = 1; i <= 10; ++i) printf("'%s' ", g_id_to_word[i]);
    printf("\n");
    return tokens;
}

/* ===================================================================================== *
 * НАТИВНЫЙ METAL SHADING LANGUAGE (MSL) КОД ДЛЯ AMD RADEON PRO 5500M                    *
 * Включает:                                                                             *
 *   1. pass1_svo_holographic_forward:                                                   *
 *      - Ролевая упаковка слов (Субъект S — Модулятор M — Объект O) + 3D-поворот Родрига *
 *      - Полигональная 2D-оболочка в центре 3D-комнаты + Голография V^(2/3)             *
 *      - 4 канала света float4(R, G, B, L) + Внешняя 3D-комната кристаллов навыков      *
 *   2. pass2_word_head_and_grad: Softmax и градиент по целому словарю слов              *
 *   3. pass3_update_3d_objects_and_pos: бэкпроп по полигональным граням и 3D-позициям   *
 *   4. pass4_update_word_embeddings: обновление векторов слов и ролей SVO               *
 *   5. render_3d_room_and_2d_projection:                                                *
 *      Настоящий 3D Raycast / Wireframe рендерер сцены 1280x720 прямо на GPU!           *
 * ===================================================================================== */
static const char *METAL_SHADER_SRC = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define VOCAB_WORDS 2048
#define SIDE 8
#define D_MODEL 64
#define N_OBJ 128
#define N_ROOM 32
#define IMG_W 1280
#define IMG_H 720

/* Формула Родрига: 3D-поворот вектора v вокруг оси углового момента omega */
inline float3 rotate_rodrigues(float3 v, float3 omega) {
    float theta = length(omega);
    if (theta < 1e-5f) return v;
    float3 k = omega / theta;
    float ct = cos(theta);
    float st = sin(theta);
    return v * ct + cross(k, v) * st + k * dot(k, v) * (1.0f - ct);
}

/* Pass 1: Полный физический шаг на уровне целых слов (SVO + Полигоны + Голография V^(2/3) + RGBL + Экзокортекс) */
kernel void pass1_svo_holographic_forward(
    device const int *words          [[buffer(0)]],
    device const float *word_embed   [[buffer(1)]],
    device const float *u_dir        [[buffer(2)]],
    device const float *base_area    [[buffer(3)]],
    device const float *obj_pos      [[buffer(4)]],
    device const float *obj_z_open   [[buffer(5)]],
    device const float4 *obj_rgbl    [[buffer(6)]],
    device const float *obj_in       [[buffer(7)]],
    device const float *obj_out      [[buffer(8)]],
    device const float *room_anchors [[buffer(9)]],
    device const float *room_z_need  [[buffer(10)]],
    device float *room_facts         [[buffer(11)]],
    device float *room_skills        [[buffer(12)]],
    device float *flux_out           [[buffer(13)]],
    device float *lum_out            [[buffer(14)]],
    device float *sig_out            [[buffer(15)]],
    device float *mu_post_out        [[buffer(16)]],
    device float *snap_r_vert        [[buffer(17)]], /* (64): радиусы вершин фигуры для 3D-рендера */
    device float4 *snap_rgbl         [[buffer(18)]], /* (64): цвета и яркость RGBL полигонов для рендера */
    device float *snap_obj_energy    [[buffer(19)]], /* (128): энергия вспышки 3D-узлов для рендера */
    device float *snap_room_energy   [[buffer(20)]], /* (32): активность кристаллов комнаты для рендера */
    device float3 *snap_ridge_ray    [[buffer(21)]], /* [0]=ridge_pos, [1]=ridge_norm, [2]=omega_rot */
    device int *active_facets_cnt    [[buffer(22)]],
    device int *recalled_slots_cnt   [[buffer(23)]],
    constant int &n_words            [[buffer(24)]],
    uint tid                         [[thread_position_in_grid]])
{
    if ((int)tid >= n_words) return;

    int w_cur = words[tid];
    int w_prev1 = (tid > 0) ? words[tid - 1] : w_cur;
    int w_prev2 = (tid > 1) ? words[tid - 2] : w_prev1;

    /* 1. УПАКОВКА СМЫСЛОВ ЕСТЕСТВЕННОГО ЯЗЫКА: СУБЪЕКТ (S) — МОДУЛЯТОР (M) — ОБЪЕКТ (O) */
    float s_vec[D_MODEL], m_vec[D_MODEL], o_vec[D_MODEL];
    float w_ext[D_MODEL], w_int[D_MODEL], eps[D_MODEL], c_int[D_MODEL], R_vert[D_MODEL];

    for (int d = 0; d < D_MODEL; ++d) {
        s_vec[d] = word_embed[w_prev2 * D_MODEL + d]; /* Субъект (опорное ядро фразы) */
        m_vec[d] = word_embed[w_prev1 * D_MODEL + d]; /* Модулятор (оператор воздействия/глагол) */
        o_vec[d] = word_embed[w_cur * D_MODEL + d];   /* Текущий Объект на входе */

        /* Билинейная модуляция Субъекта оператором Модулятора (S * tanh(M) -> ожидание Объекта) */
        float mod_op = tanh(m_vec[d]);
        w_int[d] = 0.55f * s_vec[d] * (1.0f + mod_op) + 0.45f * m_vec[d];
        w_ext[d] = o_vec[d] + 0.30f * m_vec[d];
        eps[d] = w_ext[d] - w_int[d];
        c_int[d] = tanh(fabs(w_int[d]) * 1.6f);
    }

    /* Модулятор задает 3D-вектор углового момента omega_M и поворачивает 3D-оболочку по Родригу! */
    float3 omega_rot = float3(
        tanh(m_vec[0] - s_vec[0]) * 0.65f,
        tanh(m_vec[1] - s_vec[1]) * 0.65f,
        tanh(m_vec[2] - s_vec[2]) * 0.65f
    );

    float3 p_surf[D_MODEL];
    float3 rot_udir[D_MODEL];
    for (int d = 0; d < D_MODEL; ++d) {
        /* Голографический принцип: объем V растет от внутренней сложности c_int и энергии интерференции */
        float vol = 0.85f + c_int[d] + 0.55f * tanh(0.5f * eps[d] * eps[d]);
        float r_val = pow(vol, 0.333333f) * (1.0f + 0.35f * tanh(w_int[d]) + 0.45f * tanh(eps[d]));
        R_vert[d] = r_val;
        float3 u0 = float3(u_dir[d * 3 + 0], u_dir[d * 3 + 1], u_dir[d * 3 + 2]);
        float3 u_rot = rotate_rodrigues(u0, omega_rot);
        rot_udir[d] = u_rot;
        p_surf[d] = r_val * u_rot;
    }

    /* 2. ПОЛИГОНАЛЬНЫЙ 2D-РЕЛЬЕФ (ЧЕРЕЗ ВЕКТОРНОЕ ПРОИЗВЕДЕНИЕ 3D-РЕБЕР) И 4 КАНАЛА СВЕТА FLOAT4(R, G, B, L) */
    float4 proc_rgbl[D_MODEL];
    float rad_flux[D_MODEL];
    float3 active_pos = float3(0.0f);
    float3 active_norm = float3(0.0f);
    float w_sum = 1e-6f;

    for (int u = 0; u < SIDE; ++u) {
        int u_next = (u < SIDE - 1) ? (u + 1) : (SIDE - 2);
        for (int v = 0; v < SIDE; ++v) {
            int v_next = (v + 1) & (SIDE - 1);
            int v_prev = (v + SIDE - 1) & (SIDE - 1);
            int d = u * SIDE + v;

            float3 edge_u = p_surf[u_next * SIDE + v] - p_surf[d];
            float3 edge_v = p_surf[u * SIDE + v_next] - p_surf[d];
            float3 poly_cross = cross(edge_u, edge_v);
            float area_3d = length(poly_cross) + 1e-6f;
            float3 n_poly = poly_cross / area_3d;
            if (dot(n_poly, rot_udir[d]) < 0.0f) n_poly = -n_poly;

            float crease = R_vert[u * SIDE + v_next] + R_vert[u * SIDE + v_prev] - 2.0f * R_vert[d];
            float dA_poly = (area_3d / (base_area[d] + 1e-5f)) * sqrt(1.0f + 0.5f * crease * crease);

            float eps_u = tanh(eps[d]);
            float tilt = 1.0f - fabs(dot(n_poly, rot_udir[d]));
            float col_R = eps_u * cos(1.4f * R_vert[d]);
            float col_G = eps_u * tilt * cos(2.8f * R_vert[d]) + tanh(crease) * sin(2.8f * R_vert[d]);
            float col_B = tanh(crease) * cos(5.2f * eps_u) + 0.5f * c_int[d] * eps_u;
            float lum_L = 1.0f + 0.70f * tanh(0.5f * eps[d] * eps[d] + c_int[d] * fabs(eps[d]));

            proc_rgbl[d] = float4(col_R * lum_L, col_G * lum_L, col_B * lum_L, lum_L);
            float flux = lum_L * dA_poly;
            rad_flux[d] = flux;
            flux_out[tid * D_MODEL + d] = dA_poly;
            lum_out[tid * D_MODEL + d] = lum_L;
            sig_out[tid * D_MODEL + d] = eps[d] + 0.25f * (col_R + col_G + col_B) * lum_L;

            if ((int)tid == n_words - 1) {
                snap_r_vert[d] = R_vert[d];
                snap_rgbl[d] = float4(col_R, col_G, col_B, lum_L);
            }

            float fw = fabs(eps[d]) * flux;
            active_pos += p_surf[d] * fw;
            active_norm += n_poly * fw;
            w_sum += fw;
        }
    }
    active_pos /= w_sum;
    active_norm = normalize(active_norm + float3(1e-6f, 0.0f, 0.0f));
    if ((int)tid == n_words - 1) {
        snap_ridge_ray[0] = active_pos;
        snap_ridge_ray[1] = active_norm;
        snap_ridge_ray[2] = omega_rot;
    }

    /* 3. ВНУТРЕННИЙ КАСКАД ПО 3D-ОБЪЕКТАМ КОМНАТЫ */
    float corr[D_MODEL];
    for (int d = 0; d < D_MODEL; ++d) corr[d] = 0.0f;
    int opened = 0;

    for (int j = 0; j < N_OBJ; ++j) {
        float3 opos = float3(obj_pos[j * 3 + 0], obj_pos[j * 3 + 1], obj_pos[j * 3 + 2]);
        float3 delta = opos - active_pos;
        float dist = length(delta) + 1e-6f;
        float cos_v = max(0.0f, dot(active_norm, delta / dist));
        float persp = (0.35f + 0.65f * cos_v) * exp(-dist * 0.65f);
        float z_thresh = obj_z_open[j];

        float4 flt = obj_rgbl[j];
        int off = j * D_MODEL;
        float dot_acc = 0.0f;
        int obj_open = 0;
        if (persp * 3.5f > z_thresh) {
            for (int d = 0; d < D_MODEL; ++d) {
                float sec_e = persp * rad_flux[d];
                if (sec_e > z_thresh) {
                    float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);
                    float rgbl_sig = eps[d] + dot(proc_rgbl[d], flt);
                    dot_acc += rgbl_sig * obj_in[off + d] * gate;
                    obj_open++;
                }
            }
            if (obj_open > 0) {
                opened += obj_open;
                float resp = tanh(dot_acc);
                for (int d = 0; d < D_MODEL; ++d) {
                    float sec_e = persp * rad_flux[d];
                    if (sec_e > z_thresh) {
                        float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);
                        corr[d] += resp * obj_out[off + d] * gate;
                    }
                }
            }
        }
        if ((int)tid == n_words - 1) {
            snap_obj_energy[j] = (float)obj_open / (float)D_MODEL;
        }
    }

    /* 4. ВНЕШНИЙ 3D-ЭКЗОКОРТЕКС НАВЫКОВ В КОМНАТЕ (БЕЗ KV-КЭША, ПО НЕОБХОДИМОСТИ) */
    float unsolved_norm2 = 0.0f;
    for (int d = 0; d < D_MODEL; ++d) {
        float u_err = eps[d] - corr[d];
        unsolved_norm2 += u_err * u_err;
    }
    float need_amp = sqrt(unsolved_norm2 + 1e-6f);
    int recalled = 0;
    for (int m = 0; m < N_ROOM; ++m) {
        float3 rpos = float3(room_anchors[m * 3 + 0], room_anchors[m * 3 + 1], room_anchors[m * 3 + 2]);
        float3 to_s = rpos - active_pos;
        float d_s = length(to_s) + 1e-6f;
        float cos_b = max(0.0f, dot(active_norm, to_s / d_s));
        float q_energy = need_amp * (cos_b * cos_b * cos_b);
        float rg = 0.0f;
        if (q_energy > room_z_need[m]) {
            recalled++;
            rg = (q_energy - room_z_need[m]) / (q_energy - room_z_need[m] + 1.0f);
            int m_off = m * D_MODEL;
            for (int d = 0; d < D_MODEL; ++d) {
                corr[d] += 0.15f * rg * (room_facts[m_off + d] + room_skills[m_off + d]);
                if ((int)tid == n_words - 1) {
                    room_facts[m_off + d] = 0.92f * room_facts[m_off + d] + 0.08f * m_vec[d];
                    room_skills[m_off + d] = 0.92f * room_skills[m_off + d] + 0.08f * (w_ext[d] - m_vec[d]);
                }
            }
        }
        if ((int)tid == n_words - 1) snap_room_energy[m] = rg;
    }

    active_facets_cnt[tid] = opened;
    recalled_slots_cnt[tid] = recalled;
    for (int d = 0; d < D_MODEL; ++d) {
        mu_post_out[tid * D_MODEL + d] = w_ext[d] + corr[d];
    }
}

/* Pass 2: Вычисление логитов по словарю ЦЕЛЫХ СЛОВ (VOCAB_WORDS = 2048) и градиента на GPU */
kernel void pass2_word_head_and_grad(
    device const float *mu_post_out [[buffer(0)]],
    device const int *target_words  [[buffer(1)]],
    device const float *head_w      [[buffer(2)]],
    device float *d_logits_out      [[buffer(3)]],
    device float *d_hidden          [[buffer(4)]],
    device float *word_losses       [[buffer(5)]],
    constant int &n_words           [[buffer(6)]],
    uint tid                        [[thread_position_in_grid]])
{
    if ((int)tid >= n_words) return;
    float h[D_MODEL];
    for (int d = 0; d < D_MODEL; ++d) h[d] = mu_post_out[tid * D_MODEL + d];

    float max_l = -1e30f;
    for (int v = 0; v < VOCAB_WORDS; ++v) {
        float s = 0.0f;
        int off = v * D_MODEL;
        for (int d = 0; d < D_MODEL; ++d) s += h[d] * head_w[off + d];
        d_logits_out[tid * VOCAB_WORDS + v] = s;
        if (s > max_l) max_l = s;
    }
    float sum_e = 0.0f;
    for (int v = 0; v < VOCAB_WORDS; ++v) {
        float e = exp(d_logits_out[tid * VOCAB_WORDS + v] - max_l);
        d_logits_out[tid * VOCAB_WORDS + v] = e;
        sum_e += e;
    }
    int tgt = target_words[tid];
    float prob = max(d_logits_out[tid * VOCAB_WORDS + tgt] / sum_e, 1e-12f);
    word_losses[tid] = -log(prob);

    float dh[D_MODEL];
    for (int d = 0; d < D_MODEL; ++d) dh[d] = 0.0f;
    float inv_n = 1.0f / (float)n_words;
    for (int v = 0; v < VOCAB_WORDS; ++v) {
        float dl = (d_logits_out[tid * VOCAB_WORDS + v] / sum_e - ((v == tgt) ? 1.0f : 0.0f)) * inv_n;
        d_logits_out[tid * VOCAB_WORDS + v] = dl;
        int off = v * D_MODEL;
        for (int d = 0; d < D_MODEL; ++d) dh[d] += dl * head_w[off + d];
    }
    for (int d = 0; d < D_MODEL; ++d) d_hidden[tid * D_MODEL + d] = dh[d];
}

/* Pass 3: Бэкпроп по полигональным граням и 3D-координатам объектов в комнате (X, Y, Z) */
kernel void pass3_update_3d_objects_and_pos(
    device const float *sig_out    [[buffer(0)]],
    device const float *flux_out   [[buffer(1)]],
    device const float *lum_out    [[buffer(2)]],
    device const float *d_hidden   [[buffer(3)]],
    device const float *obj_z_open [[buffer(4)]],
    device float *obj_in           [[buffer(5)]],
    device float *obj_out          [[buffer(6)]],
    device float *obj_pos          [[buffer(7)]],
    constant int &n_words          [[buffer(8)]],
    constant float &lr             [[buffer(9)]],
    uint j                         [[thread_position_in_grid]])
{
    if ((int)j >= N_OBJ) return;
    int off = j * D_MODEL;
    float z_thresh = obj_z_open[j];
    float gin[D_MODEL], gout[D_MODEL];
    for (int d = 0; d < D_MODEL; ++d) { gin[d] = 0.0f; gout[d] = 0.0f; }
    float3 gpos = float3(0.0f);

    for (int tid = 0; tid < n_words; ++tid) {
        for (int d = 0; d < D_MODEL; ++d) {
            float sec_e = 0.35f * flux_out[tid * D_MODEL + d] * lum_out[tid * D_MODEL + d];
            if (sec_e <= z_thresh) continue;
            float gate = (sec_e - z_thresh) / (sec_e - z_thresh + 0.5f);
            float dh = d_hidden[tid * D_MODEL + d] * gate;
            float s_val = sig_out[tid * D_MODEL + d];
            gout[d] += dh * tanh(s_val * obj_in[off + d]);
            gin[d]  += dh * obj_out[off + d] * s_val;
            if (d < 3) gpos[d] += dh * s_val * 0.05f;
        }
    }
    for (int d = 0; d < D_MODEL; ++d) {
        obj_in[off + d]  -= lr * clamp(gin[d], -1.0f, 1.0f);
        obj_out[off + d] -= lr * clamp(gout[d], -1.0f, 1.0f);
    }
    /* 3D-узлы в комнате физически сдвигаются по градиенту в процессе обучения! */
    obj_pos[j * 3 + 0] = clamp(obj_pos[j * 3 + 0] - lr * clamp(gpos.x, -0.2f, 0.2f), -1.8f, 1.8f);
    obj_pos[j * 3 + 1] = clamp(obj_pos[j * 3 + 1] - lr * clamp(gpos.y, -0.2f, 0.2f), -1.8f, 1.8f);
    obj_pos[j * 3 + 2] = clamp(obj_pos[j * 3 + 2] - lr * clamp(gpos.z, -0.2f, 0.2f), -1.8f, 1.8f);
}

/* Pass 4: Обновление смысловых векторов ЦЕЛЫХ СЛОВ */
kernel void pass4_update_word_embeddings(
    device const int *words         [[buffer(0)]],
    device const float *mu_post_out [[buffer(1)]],
    device const float *d_logits    [[buffer(2)]],
    device const float *d_hidden    [[buffer(3)]],
    device float *head_w            [[buffer(4)]],
    device float *head_mom          [[buffer(5)]],
    device float *word_embed        [[buffer(6)]],
    device float *embed_mom         [[buffer(7)]],
    constant int &n_words           [[buffer(8)]],
    constant float &lr              [[buffer(9)]],
    uint v                          [[thread_position_in_grid]])
{
    if ((int)v >= VOCAB_WORDS) return;
    float g_head[D_MODEL], g_emb[D_MODEL];
    for (int d = 0; d < D_MODEL; ++d) { g_head[d] = 0.0f; g_emb[d] = 0.0f; }

    for (int tid = 0; tid < n_words; ++tid) {
        float dl = d_logits[tid * VOCAB_WORDS + v];
        for (int d = 0; d < D_MODEL; ++d) g_head[d] += dl * mu_post_out[tid * D_MODEL + d];
        int w_cur = words[tid];
        int w_p1 = (tid > 0) ? words[tid - 1] : w_cur;
        int w_p2 = (tid > 1) ? words[tid - 2] : w_p1;
        float we = ((w_cur == (int)v) ? 1.0f : 0.0f) + ((w_p1 == (int)v) ? 0.55f : 0.0f) + ((w_p2 == (int)v) ? 0.35f : 0.0f);
        if (we > 0.0f) {
            for (int d = 0; d < D_MODEL; ++d) g_emb[d] += we * d_hidden[tid * D_MODEL + d];
        }
    }
    int off = v * D_MODEL;
    for (int d = 0; d < D_MODEL; ++d) {
        float mh = 0.9f * head_mom[off + d] + g_head[d];
        head_mom[off + d] = mh;
        head_w[off + d] -= lr * clamp(mh, -1.5f, 1.5f);
        float me = 0.9f * embed_mom[off + d] + g_emb[d];
        embed_mom[off + d] = me;
        word_embed[off + d] -= lr * clamp(me, -1.5f, 1.5f);
    }
}

/* Вспомогательная функция расстояния от 2D-точки p до отрезка (a, b) на экране */
inline float dist_to_seg2d(float2 p, float2 a, float2 b) {
    float2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / (dot(ba, ba) + 1e-6f), 0.0f, 1.0f);
    return length(pa - ba * h);
}

/* Перспективная проекция 3D-точки мира (X, Y, Z) на левый видовой экран 3D-комнаты */
inline float2 project_3d_point(float3 pt, float cam_yaw, float cam_pitch, float2 center, float scale, thread float &out_z) {
    float cy = cos(cam_yaw), sy = sin(cam_yaw);
    float cp = cos(cam_pitch), sp = sin(cam_pitch);
    /* Поворот вокруг вертикальной оси Z и наклон по Pitch */
    float x1 =  pt.x * cy + pt.y * sy;
    float y1 = -pt.x * sy + pt.y * cy;
    float z1 =  pt.z;
    float y2 =  y1 * cp - z1 * sp;
    float z2 =  y1 * sp + z1 * cp;
    float depth = 5.2f + y2;
    out_z = depth;
    float inv_z = 1.0f / max(depth, 0.5f);
    return float2(center.x + x1 * scale * inv_z, center.y - z2 * scale * inv_z);
}

/* Pass 5: НАТИВНЫЙ GPU-РЕНДЕРЕР 3D-ПРОСТРАНСТВА, ОСЕЙ, РЕЛЬЕФНОЙ ФИГУРЫ И 2D-ПРОЕКЦИИ В РЕАЛЬНОМ ВРЕМЕНИ (1280x720) */
kernel void render_3d_room_and_2d_projection(
    device const float *u_dir            [[buffer(0)]],
    device const float *snap_r_vert      [[buffer(1)]],
    device const float4 *snap_rgbl       [[buffer(2)]],
    device const float *obj_pos          [[buffer(3)]],
    device const float *snap_obj_energy  [[buffer(4)]],
    device const float *room_anchors     [[buffer(5)]],
    device const float *snap_room_energy [[buffer(6)]],
    device const float3 *snap_ridge_ray  [[buffer(7)]],
    device const float *loss_history     [[buffer(8)]], /* история ошибок по кадрам */
    device const float *area_history     [[buffer(9)]], /* история площади рельефа */
    device uchar4 *frame_rgba            [[buffer(10)]],
    constant int &cur_frame              [[buffer(11)]],
    constant int &total_frames           [[buffer(12)]],
    uint2 gid                            [[thread_position_in_grid]])
{
    if (gid.x >= IMG_W || gid.y >= IMG_H) return;
    float2 pix = float2((float)gid.x, (float)gid.y);
    float3 col = float3(0.04f, 0.05f, 0.08f); /* глубокий темный фон 3D-сцены */

    /* Разделительная линия между 3D-перспективой слева (0..780) и 2D-проекциями справа (780..1280) */
    if (abs(pix.x - 780.0f) < 1.5f) {
        frame_rgba[gid.y * IMG_W + gid.x] = uchar4(90, 110, 140, 255);
        return;
    }

    float3 omega_rot = snap_ridge_ray[2];
    float cam_yaw = 0.55f + 0.0045f * (float)cur_frame;
    float cam_pitch = 0.42f;

    /* ================================================================================= *
     * ЛЕВЫЙ ЭКРАН (0..780 x 0..720): ПЕРСПЕКТИВНЫЙ 3D-ВИД КОМНАТЫ, ОСЕЙ X,Y,Z И ФИГУРЫ   *
     * ================================================================================= */
    if (pix.x < 780.0f) {
        float2 center = float2(390.0f, 365.0f);
        float scale = 820.0f;
        float z_dummy = 1.0f;

        /* 1. Рисуем координатную сетку пола 3D-комнаты Z = -2.0 и куб границ [-2, 2]^3 */
        for (int g = -2; g <= 2; ++g) {
            float gf = (float)g;
            float2 a1 = project_3d_point(float3(gf, -2.0f, -2.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
            float2 b1 = project_3d_point(float3(gf,  2.0f, -2.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
            float d1 = dist_to_seg2d(pix, a1, b1);
            if (d1 < 1.2f) col = max(col, float3(0.12f, 0.16f, 0.24f) * (1.2f - d1));

            float2 a2 = project_3d_point(float3(-2.0f, gf, -2.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
            float2 b2 = project_3d_point(float3( 2.0f, gf, -2.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
            float d2 = dist_to_seg2d(pix, a2, b2);
            if (d2 < 1.2f) col = max(col, float3(0.12f, 0.16f, 0.24f) * (1.2f - d2));
        }

        /* 2. ГЛАВНЫЕ ПРОСТРАНСТВЕННЫЕ ОСИ X (Красная), Y (Зеленая), Z (Голубая) из центра (0,0,0) */
        float2 orig = project_3d_point(float3(0.0f, 0.0f, 0.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
        float2 ax_x = project_3d_point(float3(2.4f, 0.0f, 0.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
        float2 ax_y = project_3d_point(float3(0.0f, 2.4f, 0.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
        float2 ax_z = project_3d_point(float3(0.0f, 0.0f, 2.4f), cam_yaw, cam_pitch, center, scale, z_dummy);

        float dx = dist_to_seg2d(pix, orig, ax_x);
        if (dx < 2.2f) col = mix(col, float3(1.0f, 0.22f, 0.22f), clamp(2.2f - dx, 0.0f, 1.0f));
        float dy = dist_to_seg2d(pix, orig, ax_y);
        if (dy < 2.2f) col = mix(col, float3(0.22f, 1.0f, 0.35f), clamp(2.2f - dy, 0.0f, 1.0f));
        float dz = dist_to_seg2d(pix, orig, ax_z);
        if (dz < 2.2f) col = mix(col, float3(0.25f, 0.65f, 1.0f), clamp(2.2f - dz, 0.0f, 1.0f));

        /* Засечки единиц вдоль осей X, Y, Z */
        for (int tick = 1; tick <= 2; ++tick) {
            float tf = (float)tick;
            float2 tx = project_3d_point(float3(tf, 0.0f, 0.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
            float2 ty = project_3d_point(float3(0.0f, tf, 0.0f), cam_yaw, cam_pitch, center, scale, z_dummy);
            float2 tz = project_3d_point(float3(0.0f, 0.0f, tf), cam_yaw, cam_pitch, center, scale, z_dummy);
            if (length(pix - tx) < 4.0f) col = float3(1.0f, 0.4f, 0.4f);
            if (length(pix - ty) < 4.0f) col = float3(0.4f, 1.0f, 0.5f);
            if (length(pix - tz) < 4.0f) col = float3(0.4f, 0.8f, 1.0f);
        }

        /* 3. ВНЕШНИЕ КРИСТАЛЛЫ ПАМЯТИ НАВЫКОВ В 3D-КОМНАТЕ (32 кристалла на внешней орбите) */
        float3 ridge_pos = snap_ridge_ray[0];
        float2 ridge_2d = project_3d_point(ridge_pos, cam_yaw, cam_pitch, center, scale, z_dummy);

        for (int m = 0; m < N_ROOM; ++m) {
            float3 rpos = float3(room_anchors[m * 3 + 0], room_anchors[m * 3 + 1], room_anchors[m * 3 + 2]);
            float2 rp2d = project_3d_point(rpos, cam_yaw, cam_pitch, center, scale, z_dummy);
            float r_en = snap_room_energy[m];
            float d_node = length(pix - rp2d);
            float rad_p = 3.5f + 5.0f * r_en;
            if (d_node < rad_p) {
                float3 c_cryst = mix(float3(0.35f, 0.45f, 0.70f), float3(1.0f, 0.85f, 0.25f), clamp(r_en * 2.5f, 0.0f, 1.0f));
                col = mix(col, c_cryst, clamp(rad_p - d_node, 0.0f, 1.0f));
            }
            /* Если кристалл пробит лучом необходимости — рисуем золотой луч обращения к внешней памяти! */
            if (r_en > 0.02f) {
                float d_beam = dist_to_seg2d(pix, ridge_2d, rp2d);
                if (d_beam < 1.5f) {
                    col += float3(1.0f, 0.80f, 0.20f) * (1.5f - d_beam) * clamp(r_en * 1.8f, 0.15f, 0.85f);
                }
            }
        }

        /* 4. ВНУТРЕННИЕ И ОКРУЖАЮЩИЕ 3D-УЗЛЫ ПАМЯТИ (128 узлов в комнате) + ЛУЧИ КОЛЛИЗИЙ */
        for (int j = 0; j < N_OBJ; ++j) {
            float3 opos = float3(obj_pos[j * 3 + 0], obj_pos[j * 3 + 1], obj_pos[j * 3 + 2]);
            float2 op2d = project_3d_point(opos, cam_yaw, cam_pitch, center, scale, z_dummy);
            float o_en = snap_obj_energy[j];
            float d_o = length(pix - op2d);
            float r_o = 2.2f + 4.5f * o_en;
            if (d_o < r_o) {
                float3 node_rgb = clamp(float3(0.45f + 0.35f * fabs(opos.x), 0.45f + 0.35f * fabs(opos.y), 0.55f + 0.35f * fabs(opos.z)), 0.25f, 1.0f);
                col = mix(col, node_rgb * (0.4f + 1.4f * o_en), clamp(r_o - d_o, 0.0f, 1.0f));
            }
            if (o_en > 0.12f) {
                float d_ray = dist_to_seg2d(pix, ridge_2d, op2d);
                if (d_ray < 1.1f) {
                    col += float3(0.20f, 0.85f, 1.0f) * (1.1f - d_ray) * o_en * 0.6f;
                }
            }
        }

        /* 5. РЕЛЬЕФНАЯ ПОЛИГОНАЛЬНАЯ ФИГУРА В ЦЕНТРЕ 3D-КОМНАТЫ (ГОРЫ, ВПАДИНЫ, ПОЛИГОНЫ И ЦВЕТА RGBL) */
        for (int u = 0; u < SIDE; ++u) {
            int u_next = (u < SIDE - 1) ? (u + 1) : u;
            for (int v = 0; v < SIDE; ++v) {
                int v_next = (v + 1) & (SIDE - 1);
                int d = u * SIDE + v;
                int d_u = u_next * SIDE + v;
                int d_v = u * SIDE + v_next;

                float3 u0 = rotate_rodrigues(float3(u_dir[d * 3 + 0], u_dir[d * 3 + 1], u_dir[d * 3 + 2]), omega_rot);
                float3 uu = rotate_rodrigues(float3(u_dir[d_u * 3 + 0], u_dir[d_u * 3 + 1], u_dir[d_u * 3 + 2]), omega_rot);
                float3 uv = rotate_rodrigues(float3(u_dir[d_v * 3 + 0], u_dir[d_v * 3 + 1], u_dir[d_v * 3 + 2]), omega_rot);

                float r0 = snap_r_vert[d] * 0.78f;
                float ru = snap_r_vert[d_u] * 0.78f;
                float rv = snap_r_vert[d_v] * 0.78f;

                float z0 = 1.0f, zu = 1.0f, zv = 1.0f;
                float2 p0 = project_3d_point(r0 * u0, cam_yaw, cam_pitch, center, scale, z0);
                float2 pu = project_3d_point(ru * uu, cam_yaw, cam_pitch, center, scale, zu);
                float2 pv = project_3d_point(rv * uv, cam_yaw, cam_pitch, center, scale, zv);

                float4 rgbl = snap_rgbl[d];
                float lum = rgbl.w;
                float3 poly_rgb = clamp(float3(0.45f + 0.7f * fabs(rgbl.x), 0.35f + 0.7f * fabs(rgbl.y), 0.50f + 0.7f * fabs(rgbl.z)) * (0.65f * lum), 0.15f, 1.0f);

                /* Ребра полигона рельефа + диагональ треугольника */
                float de1 = dist_to_seg2d(pix, p0, pu);
                float de2 = dist_to_seg2d(pix, p0, pv);
                float de3 = dist_to_seg2d(pix, pu, pv);
                float d_edge = min(de1, min(de2, de3));
                if (d_edge < 1.6f) {
                    col = mix(col, poly_rgb, clamp(1.6f - d_edge, 0.0f, 1.0f));
                }
                /* Вершина горы полигона светится пропорционально яркости L */
                float dv = length(pix - p0);
                float v_rad = 2.5f + 2.5f * (lum - 1.0f);
                if (dv < v_rad) {
                    col = mix(col, min(poly_rgb * 1.4f, float3(1.0f)), clamp(v_rad - dv, 0.0f, 1.0f));
                }
            }
        }
    }
    /* ================================================================================= *
     * ПРАВЫЙ ЭКРАН (780..1280 x 0..720):                                                *
     * 1) Верх (y: 30..270): 2D-развертка полигональной оболочки (u, v) с рельефом и RGBL*
     * 2) Середина (y: 295..495): 2D-проекции пространства по осям XY и XZ               *
     * 3) Низ (y: 520..695): Живая кривая реального обучения (падение ошибки по кадрам)  *
     * ================================================================================= */
    else {
        /* 1. 2D-РАЗВЕРТКА РЕЛЬЕФНОЙ ОБОЛОЧКИ ШАРА (theta_u x phi_v, 8x8 полигонов) */
        float2 map_tl = float2(815.0f, 35.0f);
        float2 map_sz = float2(430.0f, 225.0f);
        if (pix.x >= map_tl.x && pix.x < map_tl.x + map_sz.x && pix.y >= map_tl.y && pix.y < map_tl.y + map_sz.y) {
            float fx = (pix.x - map_tl.x) / map_sz.x * (float)SIDE;
            float fy = (pix.y - map_tl.y) / map_sz.y * (float)SIDE;
            int v = clamp((int)fx, 0, SIDE - 1);
            int u = clamp((int)fy, 0, SIDE - 1);
            int d = u * SIDE + v;

            float r_h = snap_r_vert[d];
            float4 rgbl = snap_rgbl[d];
            float lum = rgbl.w;
            float3 cell_col = float3(
                0.25f + 0.85f * fabs(rgbl.x) * lum,
                0.20f + 0.85f * fabs(rgbl.y) * lum,
                0.30f + 0.85f * fabs(rgbl.z) * lum
            );
            /* Изолинии высоты горы рельефа R(u, v) внутри каждого сектора 2D-проекции */
            float2 local_uv = float2(fx - floor(fx), fy - floor(fy)) - float2(0.5f, 0.5f);
            float ring = abs(length(local_uv) - clamp((r_h - 0.7f) * 0.45f, 0.08f, 0.45f));
            if (ring < 0.035f) cell_col = mix(cell_col, float3(1.0f, 0.95f, 0.75f), 0.75f);
            if (abs(local_uv.x) > 0.47f || abs(local_uv.y) > 0.47f) cell_col *= 0.35f;
            col = clamp(cell_col, 0.0f, 1.0f);
        }

        /* 2. ДВЕ ОРТОГОНАЛЬНЫЕ 2D-ПРОЕКЦИИ ОСЕЙ ПРОСТРАНСТВА: ПЛОСКОСТЬ XY (слева) и ПЛОСКОСТЬ XZ (справа) */
        for (int pane = 0; pane < 2; ++pane) {
            float2 p_tl = float2((pane == 0) ? 815.0f : 1040.0f, 290.0f);
            float2 p_sz = float2(205.0f, 195.0f);
            if (pix.x >= p_tl.x && pix.x < p_tl.x + p_sz.x && pix.y >= p_tl.y && pix.y < p_tl.y + p_sz.y) {
                float2 pc = p_tl + 0.5f * p_sz;
                col = float3(0.06f, 0.08f, 0.12f);
                /* Оси 2D-проекции: в окне 0 это X (красная) и Y (зеленая), в окне 1 это X (красная) и Z (голубая) */
                if (abs(pix.y - pc.y) < 1.2f) col = float3(0.95f, 0.25f, 0.25f);
                if (abs(pix.x - pc.x) < 1.2f) col = (pane == 0) ? float3(0.25f, 0.95f, 0.35f) : float3(0.25f, 0.65f, 1.0f);

                /* 2D-проекция контура деформированной фигуры на плоскость XY или XZ */
                float2 w_coord = (pix - pc) / 48.0f;
                float r_pix = length(w_coord);
                float ang = atan2(w_coord.y, w_coord.x);
                int v_idx = ((int)rint((ang + 3.14159f) / (2.0f * 3.14159f) * SIDE)) & (SIDE - 1);
                int u_idx = (pane == 0) ? (SIDE / 2) : clamp((int)((w_coord.y + 1.5f) / 3.0f * SIDE), 0, SIDE - 1);
                float r_bound = snap_r_vert[u_idx * SIDE + v_idx] * 0.78f;
                if (abs(r_pix - r_bound) < 0.07f) {
                    col = float3(0.95f, 0.75f, 0.25f);
                } else if (r_pix < r_bound) {
                    col = mix(col, float3(0.20f, 0.40f, 0.65f), 0.35f);
                }

                /* Проекция всех 128 3D-узлов комнаты на плоскость XY или XZ */
                for (int j = 0; j < N_OBJ; ++j) {
                    float ox = obj_pos[j * 3 + 0];
                    float oy = (pane == 0) ? obj_pos[j * 3 + 1] : obj_pos[j * 3 + 2];
                    float2 np = pc + float2(ox, -oy) * 48.0f;
                    float dn = length(pix - np);
                    float o_en = snap_obj_energy[j];
                    if (dn < 2.5f + 2.5f * o_en) {
                        col = mix(float3(0.35f, 0.55f, 0.85f), float3(1.0f, 0.45f, 0.15f), clamp(o_en * 2.0f, 0.0f, 1.0f));
                    }
                }
            }
        }

        /* 3. ЖИВОЙ ГРАФИК РЕАЛЬНОГО ОБУЧЕНИЯ НА GPU (y: 515..695) */
        float2 g_tl = float2(815.0f, 515.0f);
        float2 g_sz = float2(430.0f, 175.0f);
        if (pix.x >= g_tl.x && pix.x < g_tl.x + g_sz.x && pix.y >= g_tl.y && pix.y < g_tl.y + g_sz.y) {
            col = float3(0.05f, 0.07f, 0.10f);
            float gx = (pix.x - g_tl.x) / g_sz.x;
            float gy = 1.0f - (pix.y - g_tl.y) / g_sz.y; /* 0 внизу, 1 вверху */
            if (abs(gy - 0.25f) < 0.006f || abs(gy - 0.50f) < 0.006f || abs(gy - 0.75f) < 0.006f) {
                col = float3(0.14f, 0.18f, 0.24f);
            }
            int f_idx = clamp((int)(gx * (float)total_frames), 0, total_frames - 1);
            if (f_idx <= cur_frame) {
                /* Кривая 1 (ярко-бирюзовая): падение ошибки обучения слов (Loss от ~7.6 до ~4.5) */
                float l_val = loss_history[f_idx];
                float l_norm = clamp((l_val - 4.0f) / 4.2f, 0.05f, 0.95f);
                if (abs(gy - l_norm) < 0.022f) {
                    col = float3(0.15f, 0.95f, 0.85f);
                } else if (gy < l_norm) {
                    col = mix(col, float3(0.08f, 0.35f, 0.40f), 0.35f);
                }
                /* Кривая 2 (оранжевая): площадь полигонального рельефа dA_poly (~1.0x .. 2.5x) */
                float a_val = area_history[f_idx];
                float a_norm = clamp((a_val - 0.8f) / 1.5f, 0.05f, 0.95f);
                if (abs(gy - a_norm) < 0.018f) {
                    col = float3(1.0f, 0.60f, 0.15f);
                }
            }
        }
    }

    uchar4 out_px = uchar4(
        (uchar)clamp(col.x * 255.0f, 0.0f, 255.0f),
        (uchar)clamp(col.y * 255.0f, 0.0f, 255.0f),
        (uchar)clamp(col.z * 255.0f, 0.0f, 255.0f),
        255
    );
    frame_rgba[gid.y * IMG_W + gid.x] = out_px;
}
)MSL";

static float frand_sym(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return ((float)(*state & 0xFFFFu) / 32768.0f) - 1.0f;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *corpus_path = (argc > 1) ? argv[1] : "../lumen_nanite_3d_llm/wikitext_train_bytes.bin";
        FILE *fp = fopen(corpus_path, "rb");
        if (!fp) {
            fprintf(stderr, "Не найден корпус %s\n", corpus_path);
            return 1;
        }
        fseek(fp, 0, SEEK_END);
        long fsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        char *raw_text = (char *)malloc((size_t)fsize + 1);
        fread(raw_text, 1, (size_t)fsize, fp);
        raw_text[fsize] = '\0';
        fclose(fp);

        int total_corpus_words = 0;
        int *word_stream = tokenize_corpus_to_words(raw_text, fsize, &total_corpus_words);
        free(raw_text);

        /* Выбираем дискретную видеокарту AMD Radeon Pro 5500M через нативный Metal 3 */
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        id<MTLDevice> gpu = nil;
        for (id<MTLDevice> d in devices) {
            if (![d isLowPower]) {
                gpu = d;
                break;
            }
        }
        if (!gpu && [devices count] > 0) gpu = devices[0];
        printf("=== ЗАПУСК НАТИВНОГО METAL 3 ДВИЖКА НА %s (VRAM: %llu МБ, 0 СТРОЧЕК PYTORCH!) ===\n",
               [[gpu name] UTF8String], (unsigned long long)[gpu recommendedMaxWorkingSetSize] / (1024 * 1024));

        NSError *err = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> lib = [gpu newLibraryWithSource:[NSString stringWithUTF8String:METAL_SHADER_SRC]
                                               options:opts
                                                 error:&err];
        if (!lib) {
            fprintf(stderr, "Ошибка компиляции шейдера Metal:\n%s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        id<MTLComputePipelineState> pso_p1 = [gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pass1_svo_holographic_forward"] error:&err];
        id<MTLComputePipelineState> pso_p2 = [gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pass2_word_head_and_grad"] error:&err];
        id<MTLComputePipelineState> pso_p3 = [gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pass3_update_3d_objects_and_pos"] error:&err];
        id<MTLComputePipelineState> pso_p4 = [gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"pass4_update_word_embeddings"] error:&err];
        id<MTLComputePipelineState> pso_ren = [gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"render_3d_room_and_2d_projection"] error:&err];
        id<MTLCommandQueue> queue = [gpu newCommandQueue];

        /* Выделяем буферы в разделяемой/управляемой памяти Metal для прямого доступа GPU */
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
            h_ranch[m * 3 + 0] = 1.95f * sinf(phi) * cosf(th);
            h_ranch[m * 3 + 1] = 1.95f * sinf(phi) * sinf(th);
            h_ranch[m * 3 + 2] = 1.95f * cosf(phi);
            h_rzneed[m] = 0.20f + 0.65f * ((float)m / N_ROOM);
        }

        id<MTLBuffer> b_words   = [gpu newBufferWithLength:sizeof(int) * BATCH_WORDS options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_targets = [gpu newBufferWithLength:sizeof(int) * BATCH_WORDS options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_embed   = [gpu newBufferWithLength:sizeof(float) * VOCAB_WORDS * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_emb_m   = [gpu newBufferWithLength:sizeof(float) * VOCAB_WORDS * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_head    = [gpu newBufferWithLength:sizeof(float) * VOCAB_WORDS * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_head_m  = [gpu newBufferWithLength:sizeof(float) * VOCAB_WORDS * D_MODEL options:MTLResourceStorageModeShared];

        id<MTLBuffer> b_udir    = [gpu newBufferWithBytes:h_udir length:sizeof(h_udir) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_barea   = [gpu newBufferWithBytes:h_base_area length:sizeof(h_base_area) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_pos     = [gpu newBufferWithLength:sizeof(float) * N_OBJ * 3 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_zopen   = [gpu newBufferWithLength:sizeof(float) * N_OBJ options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_rgbl    = [gpu newBufferWithLength:sizeof(float) * N_OBJ * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_in      = [gpu newBufferWithLength:sizeof(float) * N_OBJ * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_out     = [gpu newBufferWithLength:sizeof(float) * N_OBJ * D_MODEL options:MTLResourceStorageModeShared];

        id<MTLBuffer> b_ranch   = [gpu newBufferWithBytes:h_ranch length:sizeof(h_ranch) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_rzneed  = [gpu newBufferWithBytes:h_rzneed length:sizeof(h_rzneed) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_rfacts  = [gpu newBufferWithLength:sizeof(float) * N_ROOM * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_rskills = [gpu newBufferWithLength:sizeof(float) * N_ROOM * D_MODEL options:MTLResourceStorageModeShared];

        id<MTLBuffer> b_flux    = [gpu newBufferWithLength:sizeof(float) * BATCH_WORDS * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_lum     = [gpu newBufferWithLength:sizeof(float) * BATCH_WORDS * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_sig     = [gpu newBufferWithLength:sizeof(float) * BATCH_WORDS * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_mupost  = [gpu newBufferWithLength:sizeof(float) * BATCH_WORDS * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_dlog    = [gpu newBufferWithLength:sizeof(float) * BATCH_WORDS * VOCAB_WORDS options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_dhid    = [gpu newBufferWithLength:sizeof(float) * BATCH_WORDS * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_losses  = [gpu newBufferWithLength:sizeof(float) * BATCH_WORDS options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_act_cnt = [gpu newBufferWithLength:sizeof(int) * BATCH_WORDS options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_rec_cnt = [gpu newBufferWithLength:sizeof(int) * BATCH_WORDS options:MTLResourceStorageModeShared];

        int total_video_frames = 360; /* 360 кадров реального обучения (12 секунд при 30 FPS) */
        id<MTLBuffer> b_snap_r   = [gpu newBufferWithLength:sizeof(float) * D_MODEL options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_snap_col = [gpu newBufferWithLength:sizeof(float) * D_MODEL * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_snap_oe  = [gpu newBufferWithLength:sizeof(float) * N_OBJ options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_snap_re  = [gpu newBufferWithLength:sizeof(float) * N_ROOM options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_snap_ray = [gpu newBufferWithLength:sizeof(float) * 16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_loss_h   = [gpu newBufferWithLength:sizeof(float) * total_video_frames options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_area_h   = [gpu newBufferWithLength:sizeof(float) * total_video_frames options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_frame_px = [gpu newBufferWithLength:IMG_W * IMG_H * 4 options:MTLResourceStorageModeShared];

        /* Инициализация весов */
        uint32_t rng = 42;
        float *p_emb = (float *)[b_embed contents];
        float *p_hd  = (float *)[b_head contents];
        memset([b_emb_m contents], 0, sizeof(float) * VOCAB_WORDS * D_MODEL);
        memset([b_head_m contents], 0, sizeof(float) * VOCAB_WORDS * D_MODEL);
        memset([b_rfacts contents], 0, sizeof(float) * N_ROOM * D_MODEL);
        memset([b_rskills contents], 0, sizeof(float) * N_ROOM * D_MODEL);
        for (int i = 0; i < VOCAB_WORDS * D_MODEL; ++i) {
            p_emb[i] = frand_sym(&rng) * 0.15f;
            p_hd[i]  = frand_sym(&rng) * 0.15f;
        }
        float *p_pos = (float *)[b_pos contents];
        float *p_zop = (float *)[b_zopen contents];
        float *p_rgb = (float *)[b_rgbl contents];
        float *p_in  = (float *)[b_in contents];
        float *p_out = (float *)[b_out contents];
        for (int j = 0; j < N_OBJ; ++j) {
            p_pos[j * 3 + 0] = frand_sym(&rng) * 1.45f;
            p_pos[j * 3 + 1] = frand_sym(&rng) * 1.45f;
            p_pos[j * 3 + 2] = frand_sym(&rng) * 1.45f;
            p_zop[j] = 0.42f + 0.35f * fabsf(frand_sym(&rng));
            for (int c = 0; c < 4; ++c) p_rgb[j * 4 + c] = frand_sym(&rng) * 0.25f;
        }
        for (int i = 0; i < N_OBJ * D_MODEL; ++i) {
            p_in[i]  = frand_sym(&rng) * 0.12f;
            p_out[i] = frand_sym(&rng) * 0.12f;
        }

        /* Открываем прямой видео-пайп в ffmpeg для записи реального MP4 (H.264, 1280x720, 30 fps) */
        const char *ffmpeg_cmd = "/usr/local/bin/ffmpeg -y -f rawvideo -pix_fmt rgba -s 1280x720 -r 30 -i - "
                                 "-c:v libx264 -pix_fmt yuv420p -preset fast -crf 18 training_3d_real_gpu_v11.mp4 2>/dev/null";
        FILE *ff_pipe = popen(ffmpeg_cmd, "w");
        if (!ff_pipe) {
            fprintf(stderr, "Не удалось запустить ffmpeg!\n");
            return 1;
        }

        int n_words = BATCH_WORDS;
        float lr = 0.12f;
        long cursor = 0;
        double t0 = now_sec();
        float *loss_hist = (float *)[b_loss_h contents];
        float *area_hist = (float *)[b_area_h contents];
        float smooth_loss = 7.6f;

        for (int frame = 0; frame < total_video_frames; ++frame) {
            /* В каждом видео-кадре выполняем 4 шага обучения на словах + рендер кадра на GPU */
            for (int sub = 0; sub < 4; ++sub) {
                if (cursor + BATCH_WORDS + 1 >= total_corpus_words) cursor = 0;
                int *pw = (int *)[b_words contents];
                int *pt = (int *)[b_targets contents];
                for (int i = 0; i < BATCH_WORDS; ++i) {
                    pw[i] = word_stream[cursor + i];
                    pt[i] = word_stream[cursor + i + 1];
                }
                cursor += BATCH_WORDS;

                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                [enc setComputePipelineState:pso_p1];
                [enc setBuffer:b_words offset:0 atIndex:0];
                [enc setBuffer:b_embed offset:0 atIndex:1];
                [enc setBuffer:b_udir offset:0 atIndex:2];
                [enc setBuffer:b_barea offset:0 atIndex:3];
                [enc setBuffer:b_pos offset:0 atIndex:4];
                [enc setBuffer:b_zopen offset:0 atIndex:5];
                [enc setBuffer:b_rgbl offset:0 atIndex:6];
                [enc setBuffer:b_in offset:0 atIndex:7];
                [enc setBuffer:b_out offset:0 atIndex:8];
                [enc setBuffer:b_ranch offset:0 atIndex:9];
                [enc setBuffer:b_rzneed offset:0 atIndex:10];
                [enc setBuffer:b_rfacts offset:0 atIndex:11];
                [enc setBuffer:b_rskills offset:0 atIndex:12];
                [enc setBuffer:b_flux offset:0 atIndex:13];
                [enc setBuffer:b_lum offset:0 atIndex:14];
                [enc setBuffer:b_sig offset:0 atIndex:15];
                [enc setBuffer:b_mupost offset:0 atIndex:16];
                [enc setBuffer:b_snap_r offset:0 atIndex:17];
                [enc setBuffer:b_snap_col offset:0 atIndex:18];
                [enc setBuffer:b_snap_oe offset:0 atIndex:19];
                [enc setBuffer:b_snap_re offset:0 atIndex:20];
                [enc setBuffer:b_snap_ray offset:0 atIndex:21];
                [enc setBuffer:b_act_cnt offset:0 atIndex:22];
                [enc setBuffer:b_rec_cnt offset:0 atIndex:23];
                [enc setBytes:&n_words length:sizeof(int) atIndex:24];
                [enc dispatchThreads:MTLSizeMake(BATCH_WORDS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                [enc setComputePipelineState:pso_p2];
                [enc setBuffer:b_mupost offset:0 atIndex:0];
                [enc setBuffer:b_targets offset:0 atIndex:1];
                [enc setBuffer:b_head offset:0 atIndex:2];
                [enc setBuffer:b_dlog offset:0 atIndex:3];
                [enc setBuffer:b_dhid offset:0 atIndex:4];
                [enc setBuffer:b_losses offset:0 atIndex:5];
                [enc setBytes:&n_words length:sizeof(int) atIndex:6];
                [enc dispatchThreads:MTLSizeMake(BATCH_WORDS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                [enc setComputePipelineState:pso_p3];
                [enc setBuffer:b_sig offset:0 atIndex:0];
                [enc setBuffer:b_flux offset:0 atIndex:1];
                [enc setBuffer:b_lum offset:0 atIndex:2];
                [enc setBuffer:b_dhid offset:0 atIndex:3];
                [enc setBuffer:b_zopen offset:0 atIndex:4];
                [enc setBuffer:b_in offset:0 atIndex:5];
                [enc setBuffer:b_out offset:0 atIndex:6];
                [enc setBuffer:b_pos offset:0 atIndex:7];
                [enc setBytes:&n_words length:sizeof(int) atIndex:8];
                [enc setBytes:&lr length:sizeof(float) atIndex:9];
                [enc dispatchThreads:MTLSizeMake(N_OBJ, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                [enc setComputePipelineState:pso_p4];
                [enc setBuffer:b_words offset:0 atIndex:0];
                [enc setBuffer:b_mupost offset:0 atIndex:1];
                [enc setBuffer:b_dlog offset:0 atIndex:2];
                [enc setBuffer:b_dhid offset:0 atIndex:3];
                [enc setBuffer:b_head offset:0 atIndex:4];
                [enc setBuffer:b_head_m offset:0 atIndex:5];
                [enc setBuffer:b_embed offset:0 atIndex:6];
                [enc setBuffer:b_emb_m offset:0 atIndex:7];
                [enc setBytes:&n_words length:sizeof(int) atIndex:8];
                [enc setBytes:&lr length:sizeof(float) atIndex:9];
                [enc dispatchThreads:MTLSizeMake(VOCAB_WORDS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }

            /* Считаем статистику шага и запускаем GPU-шейдер рендеринга 3D-комнаты и 2D-проекций */
            float *pl = (float *)[b_losses contents];
            float *pfl = (float *)[b_flux contents];
            int *pac = (int *)[b_act_cnt contents];
            int *prc = (int *)[b_rec_cnt contents];
            double lsum = 0.0, asum = 0.0, fsum = 0.0, rsum = 0.0;
            for (int i = 0; i < BATCH_WORDS; ++i) {
                lsum += pl[i];
                fsum += pac[i];
                rsum += prc[i];
            }
            for (int i = 0; i < BATCH_WORDS * D_MODEL; ++i) asum += pfl[i];
            float avg_loss = (float)(lsum / BATCH_WORDS);
            smooth_loss = (frame == 0) ? avg_loss : (0.85f * smooth_loss + 0.15f * avg_loss);
            float avg_area = (float)(asum / (BATCH_WORDS * D_MODEL));
            loss_hist[frame] = smooth_loss;
            area_hist[frame] = avg_area;

            id<MTLCommandBuffer> rcmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> renc = [rcmd computeCommandEncoder];
            [renc setComputePipelineState:pso_ren];
            [renc setBuffer:b_udir offset:0 atIndex:0];
            [renc setBuffer:b_snap_r offset:0 atIndex:1];
            [renc setBuffer:b_snap_col offset:0 atIndex:2];
            [renc setBuffer:b_pos offset:0 atIndex:3];
            [renc setBuffer:b_snap_oe offset:0 atIndex:4];
            [renc setBuffer:b_ranch offset:0 atIndex:5];
            [renc setBuffer:b_snap_re offset:0 atIndex:6];
            [renc setBuffer:b_snap_ray offset:0 atIndex:7];
            [renc setBuffer:b_loss_h offset:0 atIndex:8];
            [renc setBuffer:b_area_h offset:0 atIndex:9];
            [renc setBuffer:b_frame_px offset:0 atIndex:10];
            [renc setBytes:&frame length:sizeof(int) atIndex:11];
            [renc setBytes:&total_video_frames length:sizeof(int) atIndex:12];
            [renc dispatchThreads:MTLSizeMake(IMG_W, IMG_H, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [renc endEncoding];
            [rcmd commit];
            [rcmd waitUntilCompleted];

            fwrite([b_frame_px contents], 1, IMG_W * IMG_H * 4, ff_pipe);

            if (frame == 0 || (frame + 1) % 60 == 0) {
                int w0 = ((int *)[b_words contents])[BATCH_WORDS - 3];
                int w1 = ((int *)[b_words contents])[BATCH_WORDS - 2];
                int w2 = ((int *)[b_words contents])[BATCH_WORDS - 1];
                printf("Кадр %3d/%d (Шаг %4d) | Loss слов=%.3f (Перплексия=%.1f) | Площадь полигонов=%.2fx | "
                       "Граней=%.1f%% | Внеш.память=%.1f%% | Триада SVO: [%s -> %s -> %s]\n",
                       frame + 1, total_video_frames, (frame + 1) * 4,
                       smooth_loss, exp(smooth_loss), avg_area,
                       (fsum / (BATCH_WORDS * N_OBJ * D_MODEL)) * 100.0,
                       (rsum / (BATCH_WORDS * N_ROOM)) * 100.0,
                       g_id_to_word[w0], g_id_to_word[w1], g_id_to_word[w2]);
            }
        }
        pclose(ff_pipe);
        double elapsed = now_sec() - t0;
        printf("=== ЗАПИСЬ ВИДЕО И ОБУЧЕНИЕ НА METAL 3 GPU ЗАВЕРШЕНЫ ЗА %.2f с! ===\n", elapsed);
        printf("Видео сохранено: training_3d_real_gpu_v11.mp4 (1280x720, 360 кадров, 1440 шагов обучения на целых словах)\n");
        free(word_stream);
    }
    return 0;
}

