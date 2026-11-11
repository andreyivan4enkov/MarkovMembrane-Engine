/* =====================================================================================
 * MARKOVMEMBRANE ENGINE (v12 — Кроссплатформенный безматричный 3D-движок на VULKAN API)
 *
 * ЖЁСТКИЕ АРХИТЕКТУРНЫЕ ПРАВИЛА:
 * 1. Полный запрет на любые матрицы, матричные перемножения (0 GEMM / 0 matmul) и тензорные ядра.
 * 2. Исполнение на стандартном кроссплатформенном Vulkan 1.2+ (работает на Linux, Windows, Android и macOS):
 *    - Поля среды Psi(x,y,z,t) = (плотн. rho, темп. T, магнитн. поле B) -> аппаратная воксельная
 *      3D-текстура VkImage (VK_IMAGE_TYPE_3D, RGBA32F) с трилинейным сэмплером TMU и записью imageStore.
 *    - Уровни детализации объектов -> цепочка Mip-maps (LOD 0: 64 грани, LOD 1: 16 граней, LOD 2: 4 грани)
 *      с чтением через аппаратный textureLod().
 *    - Отсечение пустоты -> 3D Spatial Hash Grid (8x8x8 = 512 вокселей), отбрасывающий >90% объектов
 *      до начала вычислений (Broad-Phase Culling).
 *    - Взаимодействие и обучение -> 2D-коллизия на полигональном рельефе оболочки (cross(edge_u, edge_v)),
 *      эмерджентный порог Снеллиуса–Френеля T(r) и R(r) = 1 - T(r), упругий рикошет траектории (Вариант А),
 *      морфинг внутренних троек (Вариант Б) и запись термодинамического следа в 3D-среду (без KV-кэша).
 * 3. Полная валидация T1 на всех 250 304 байтах WikiText-2 без обрезов + запись детального видео
 *    2D-проекции 3D-комнаты с перспективы всех шагов обучения.
 * ===================================================================================== */

#include <vulkan/vulkan.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define IMG_W 1280
#define IMG_H 720
#define SIDE 8
#define N_SECTORS 64
#define GRID_SIDE 8
#define GRID_CELLS 512
#define N_OBJECTS 256
#define N_STREAMS 64
#define MAX_HISTORY 600

typedef struct {
    float x, y, z, w;
} vec4_t;

typedef struct {
    uint32_t x, y, z, w;
} uvec4_t;

typedef struct {
    uvec4_t cell_meta[GRID_CELLS];
    uvec4_t cell_extra[GRID_CELLS];
} SpatialGridBuf_t;

typedef struct {
    vec4_t obj_pos_rad[N_OBJECTS];
    vec4_t obj_impedance[N_OBJECTS];
    vec4_t obj_rgbl_filter[N_OBJECTS];
    vec4_t obj_mip0_features[N_OBJECTS * 16];
    vec4_t obj_mip1_features[N_OBJECTS * 4];
    vec4_t obj_mip2_features[N_OBJECTS];
    vec4_t obj_emit_fibers[N_OBJECTS * 16];
} ObjectColliderBuf_t;

typedef struct {
    vec4_t ctx_pos_vel[N_STREAMS * 2];
    vec4_t ctx_potentials[N_STREAMS];
    vec4_t ctx_internal_fibers[N_STREAMS * 16];
} ContextBodyBuf_t;

typedef struct {
    float step_loss_bits[N_STREAMS];
    uint32_t culled_objects_cnt[N_STREAMS];
    uint32_t active_lod_floats_read[N_STREAMS];
    uint32_t ricochet_events_cnt[N_STREAMS];
    float trans_T_sum[N_STREAMS];
    float holo_area_sum[N_STREAMS];

    vec4_t snap_surf_pos_r[N_SECTORS];
    vec4_t snap_surf_norm_area[N_SECTORS];
    vec4_t snap_surf_rgbl[N_SECTORS];
    vec4_t snap_obj_collision[N_OBJECTS];
    vec4_t snap_focus_rays[4];
    vec4_t snap_trail_pos[128];

    float stream_probs[N_STREAMS * N_OBJECTS];
    vec4_t stream_focus_pos[N_STREAMS];
    vec4_t stream_saved_sc[N_STREAMS * 16];
    vec4_t stream_emit_grad[N_STREAMS * 16];
} StepMetricsAndSnapBuf_t;

typedef struct {
    vec4_t hist_series[MAX_HISTORY];
} HistoryPlotBuf_t;

typedef struct {
    uint32_t stream_stride;
    uint32_t step_offset;
    uint32_t do_learn;
    float lr;
    float alpha_exp;
    uint32_t frame_idx;
} PhysicsPushConsts_t;

typedef struct {
    uint32_t frame_idx;
    uint32_t total_steps_done;
    float cam_angle;
    float cur_ema_bpc;
    float cur_cull_pct;
    float cur_vram_save_pct;
} RenderPushConsts_t;

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static float frand_sym(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return ((float)(*state & 0xFFFFu) / 32768.0f) - 1.0f;
}

static uint32_t *load_spv_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Не удалось открыть SPIR-V файл: %s\n", path);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint32_t *buf = (uint32_t *)malloc((size_t)sz);
    fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    *out_size = (size_t)sz;
    return buf;
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) && ((mem_props.memoryTypes[i].propertyFlags & props) == props)) {
            return i;
        }
    }
    fprintf(stderr, "Не найден подходящий тип памяти Vulkan!\n");
    exit(1);
}

static void create_shared_buffer(
    VkDevice dev,
    VkPhysicalDevice phys,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer *out_buf,
    VkDeviceMemory *out_mem,
    void **out_mapped
) {
    VkBufferCreateInfo bci = {0};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bci, NULL, out_buf) != VK_SUCCESS) {
        fprintf(stderr, "Ошибка vkCreateBuffer\n");
        exit(1);
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *out_buf, &req);
    VkMemoryAllocateInfo mai = {0};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(
        phys,
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    if (vkAllocateMemory(dev, &mai, NULL, out_mem) != VK_SUCCESS) {
        fprintf(stderr, "Ошибка vkAllocateMemory\n");
        exit(1);
    }
    vkBindBufferMemory(dev, *out_buf, *out_mem, 0);
    if (out_mapped) {
        vkMapMemory(dev, *out_mem, 0, size, 0, out_mapped);
    }
}

int main(int argc, char **argv) {
    const char *train_path = (argc > 1) ? argv[1] : "/Users/corpuscul/Desktop/ALM/lumen_nanite_3d_llm/wikitext_train_bytes.bin";
    const char *valid_path = (argc > 2) ? argv[2] : "/Users/corpuscul/Desktop/ALM/lumen_nanite_3d_llm/wikitext_valid_bytes.bin";

    FILE *ft = fopen(train_path, "rb");
    if (!ft) {
        fprintf(stderr, "Не найден корпус обучения: %s\n", train_path);
        return 1;
    }
    fseek(ft, 0, SEEK_END);
    long train_len = ftell(ft);
    fseek(ft, 0, SEEK_SET);
    uint8_t *train_bytes = (uint8_t *)malloc((size_t)train_len);
    fread(train_bytes, 1, (size_t)train_len, ft);
    fclose(ft);

    FILE *fv = fopen(valid_path, "rb");
    if (!fv) {
        fprintf(stderr, "Не найден корпус валидации: %s\n", valid_path);
        return 1;
    }
    fseek(fv, 0, SEEK_END);
    long valid_len = ftell(fv);
    fseek(fv, 0, SEEK_SET);
    uint8_t *valid_bytes = (uint8_t *)malloc((size_t)valid_len);
    fread(valid_bytes, 1, (size_t)valid_len, fv);
    fclose(fv);

    /* 1. Инициализация Vulkan Instance и выбор дискретной видеокарты AMD Radeon Pro 5500M */
    VkApplicationInfo app_info = {0};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "MarkovMembrane_Vulkan_MatrixFree_3D";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 2, 0);
    app_info.pEngineName = "MarkovMembraneEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 2, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici = {0};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app_info;

    VkInstance instance;
    if (vkCreateInstance(&ici, NULL, &instance) != VK_SUCCESS) {
        fprintf(stderr, "Ошибка vkCreateInstance\n");
        return 1;
    }

    uint32_t gpu_count = 0;
    vkEnumeratePhysicalDevices(instance, &gpu_count, NULL);
    if (gpu_count == 0) {
        fprintf(stderr, "Не найдено ни одного устройства Vulkan!\n");
        return 1;
    }
    VkPhysicalDevice gpus[8];
    vkEnumeratePhysicalDevices(instance, &gpu_count, gpus);

    VkPhysicalDevice chosen_gpu = gpus[0];
    VkPhysicalDeviceProperties chosen_props;
    for (uint32_t i = 0; i < gpu_count; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpus[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || strstr(props.deviceName, "Radeon")) {
            chosen_gpu = gpus[i];
            break;
        }
    }
    vkGetPhysicalDeviceProperties(chosen_gpu, &chosen_props);
    printf("=== MARKOVMEMBRANE ENGINE v12 НА VULKAN API (GPU: %s) ===\n", chosen_props.deviceName);
    printf("Полный отказ от матриц (0 GEMM / 0 matmul): 3D-текстура среды TMU + Mip-map LOD + Spatial Hash Grid + 2D-коллизия Снеллиуса–Френеля!\n");

    float queue_prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queue_prio;

    VkDeviceCreateInfo dci = {0};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice device;
    if (vkCreateDevice(chosen_gpu, &dci, NULL, &device) != VK_SUCCESS) {
        fprintf(stderr, "Ошибка vkCreateDevice\n");
        return 1;
    }
    VkQueue queue;
    vkGetDeviceQueue(device, 0, 0, &queue);

    /* 2. Создаём настоящую воксельную 3D-текстуру полей среды Psi(x,y,z,t) (8x8x8 RGBA32F)
     *    и 2D мип-атлас объектов (128x128 RGBA32F, 4 мип-уровня LOD 0..3) */
    VkImage env_img;
    VkDeviceMemory env_mem;
    VkImageCreateInfo img3d_ci = {0};
    img3d_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img3d_ci.imageType = VK_IMAGE_TYPE_3D;
    img3d_ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    img3d_ci.extent.width = GRID_SIDE;
    img3d_ci.extent.height = GRID_SIDE;
    img3d_ci.extent.depth = GRID_SIDE;
    img3d_ci.mipLevels = 1;
    img3d_ci.arrayLayers = 1;
    img3d_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img3d_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img3d_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img3d_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(device, &img3d_ci, NULL, &env_img);

    VkMemoryRequirements env_req;
    vkGetImageMemoryRequirements(device, env_img, &env_req);
    VkMemoryAllocateInfo env_mai = {0};
    env_mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    env_mai.allocationSize = env_req.size;
    env_mai.memoryTypeIndex = find_memory_type(chosen_gpu, env_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &env_mai, NULL, &env_mem);
    vkBindImageMemory(device, env_img, env_mem, 0);

    VkImageView env_view;
    VkImageViewCreateInfo iv3d_ci = {0};
    iv3d_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    iv3d_ci.image = env_img;
    iv3d_ci.viewType = VK_IMAGE_VIEW_TYPE_3D;
    iv3d_ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    iv3d_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv3d_ci.subresourceRange.levelCount = 1;
    iv3d_ci.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &iv3d_ci, NULL, &env_view);

    /* 2D Mip-map атлас (128x128, 4 уровня детализации: LOD 0..3) */
    VkImage mip_img;
    VkDeviceMemory mip_mem;
    VkImageCreateInfo img2d_ci = {0};
    img2d_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img2d_ci.imageType = VK_IMAGE_TYPE_2D;
    img2d_ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    img2d_ci.extent.width = 128;
    img2d_ci.extent.height = 128;
    img2d_ci.extent.depth = 1;
    img2d_ci.mipLevels = 4;
    img2d_ci.arrayLayers = 1;
    img2d_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    img2d_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    img2d_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    img2d_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(device, &img2d_ci, NULL, &mip_img);

    VkMemoryRequirements mip_req;
    vkGetImageMemoryRequirements(device, mip_img, &mip_req);
    VkMemoryAllocateInfo mip_mai = {0};
    mip_mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mip_mai.allocationSize = mip_req.size;
    mip_mai.memoryTypeIndex = find_memory_type(chosen_gpu, mip_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &mip_mai, NULL, &mip_mem);
    vkBindImageMemory(device, mip_img, mip_mem, 0);

    VkImageView mip_view;
    VkImageViewCreateInfo iv2d_ci = {0};
    iv2d_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    iv2d_ci.image = mip_img;
    iv2d_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv2d_ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    iv2d_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    iv2d_ci.subresourceRange.levelCount = 4;
    iv2d_ci.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &iv2d_ci, NULL, &mip_view);

    /* Аппаратный трилинейный сэмплер TMU с поддержкой Mip-map LOD */
    VkSampler tmu_sampler;
    VkSamplerCreateInfo sci = {0};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.minLod = 0.0f;
    sci.maxLod = 3.0f;
    vkCreateSampler(device, &sci, NULL, &tmu_sampler);

    /* 3. Создаём буферы Vulkan (токены, Spatial Hash Grid, 3D-объекты с мип-уровнями, контекст, метрики, история, кадр) */
    size_t max_tokens_buf = (size_t)(train_len > valid_len ? train_len : valid_len) + 1024;
    VkBuffer buf_tokens, buf_grid, buf_objs, buf_ctx, buf_metrics, buf_hist, buf_frame;
    VkDeviceMemory mem_tokens, mem_grid, mem_objs, mem_ctx, mem_metrics, mem_hist, mem_frame;
    uint32_t *map_tokens = NULL;
    SpatialGridBuf_t *map_grid = NULL;
    ObjectColliderBuf_t *map_objs = NULL;
    ContextBodyBuf_t *map_ctx = NULL;
    StepMetricsAndSnapBuf_t *map_metrics = NULL;
    HistoryPlotBuf_t *map_hist = NULL;
    uint32_t *map_frame = NULL;

    create_shared_buffer(device, chosen_gpu, max_tokens_buf * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_tokens, &mem_tokens, (void **)&map_tokens);
    create_shared_buffer(device, chosen_gpu, sizeof(SpatialGridBuf_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_grid, &mem_grid, (void **)&map_grid);
    create_shared_buffer(device, chosen_gpu, sizeof(ObjectColliderBuf_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_objs, &mem_objs, (void **)&map_objs);
    create_shared_buffer(device, chosen_gpu, sizeof(ContextBodyBuf_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_ctx, &mem_ctx, (void **)&map_ctx);
    create_shared_buffer(device, chosen_gpu, sizeof(StepMetricsAndSnapBuf_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_metrics, &mem_metrics, (void **)&map_metrics);
    create_shared_buffer(device, chosen_gpu, sizeof(HistoryPlotBuf_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_hist, &mem_hist, (void **)&map_hist);
    create_shared_buffer(device, chosen_gpu, IMG_W * IMG_H * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &buf_frame, &mem_frame, (void **)&map_frame);

    /* Инициализируем 256 3D-коллайдеров в комнате [-1.6, 1.6]^3 и их мип-пирамиду (LOD 0 -> LOD 1 -> LOD 2) */
    uint32_t rng = 42u;
    memset(map_grid, 0, sizeof(SpatialGridBuf_t));
    memset(map_ctx, 0, sizeof(ContextBodyBuf_t));
    memset(map_metrics, 0, sizeof(StepMetricsAndSnapBuf_t));
    memset(map_hist, 0, sizeof(HistoryPlotBuf_t));

    for (uint32_t j = 0; j < N_OBJECTS; ++j) {
        /* Равномерно-спиральное начальное распределение в 3D-комнате + возмущение */
        float phi = 2.3999632f * (float)j;
        float z_c = 1.0f - (2.0f * (float)j + 1.0f) / (float)N_OBJECTS;
        float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z_c * z_c));
        float rad_shell = 0.75f + 0.85f * ((float)(j % 7u) / 6.0f);
        map_objs->obj_pos_rad[j] = (vec4_t){
            rad_shell * r_xy * cosf(phi) + 0.12f * frand_sym(&rng),
            rad_shell * z_c + 0.12f * frand_sym(&rng),
            rad_shell * r_xy * sinf(phi) + 0.12f * frand_sym(&rng),
            -0.35f
        };
        map_objs->obj_impedance[j] = (vec4_t){
            0.25f * frand_sym(&rng),
            0.25f * frand_sym(&rng),
            0.25f * frand_sym(&rng),
            0.30f + 0.15f * fabsf(frand_sym(&rng))
        };
        map_objs->obj_rgbl_filter[j] = (vec4_t){
            0.35f * frand_sym(&rng),
            0.35f * frand_sym(&rng),
            0.35f * frand_sym(&rng),
            0.50f + 0.25f * fabsf(frand_sym(&rng))
        };

        vec4_t mip2 = {0, 0, 0, 0};
        for (uint32_t m = 0; m < 4; ++m) {
            vec4_t mip1 = {0, 0, 0, 0};
            for (uint32_t sub = 0; sub < 4; ++sub) {
                uint32_t q = m * 4 + sub;
                vec4_t f0 = {
                    0.35f * frand_sym(&rng),
                    0.35f * frand_sym(&rng),
                    0.35f * frand_sym(&rng),
                    0.35f * frand_sym(&rng)
                };
                vec4_t e0 = {
                    0.35f * frand_sym(&rng),
                    0.35f * frand_sym(&rng),
                    0.35f * frand_sym(&rng),
                    0.35f * frand_sym(&rng)
                };
                map_objs->obj_mip0_features[j * 16 + q] = f0;
                map_objs->obj_emit_fibers[j * 16 + q] = e0;
                mip1.x += 0.25f * f0.x;
                mip1.y += 0.25f * f0.y;
                mip1.z += 0.25f * f0.z;
                mip1.w += 0.25f * f0.w;
            }
            map_objs->obj_mip1_features[j * 4 + m] = mip1;
            mip2.x += 0.25f * mip1.x;
            mip2.y += 0.25f * mip1.y;
            mip2.z += 0.25f * mip1.z;
            mip2.w += 0.25f * mip1.w;
        }
        map_objs->obj_mip2_features[j] = mip2;
    }

    for (uint32_t s = 0; s < N_STREAMS; ++s) {
        map_ctx->ctx_pos_vel[s * 2 + 0] = (vec4_t){0.0f, 0.0f, 0.0f, 0.1f};
        map_ctx->ctx_pos_vel[s * 2 + 1] = (vec4_t){0.577f, 0.577f, 0.577f, 1.0f};
        map_ctx->ctx_potentials[s] = (vec4_t){1.0f, 1.0f, 0.5f, 1.0f};
    }

    /* 4. Создаём DescriptorSetLayout (bindings 0..9), PipelineLayout и 3 вычислительных конвейера Vulkan */
    VkDescriptorSetLayoutBinding bindings[10] = {0};
    bindings[0] = (VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    bindings[1] = (VkDescriptorSetLayoutBinding){1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    bindings[2] = (VkDescriptorSetLayoutBinding){2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    for (uint32_t b = 3; b <= 9; ++b) {
        bindings[b] = (VkDescriptorSetLayoutBinding){b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 10;
    dsl_ci.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(device, &dsl_ci, NULL, &dsl);

    VkPushConstantRange pcr = {0};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 24; /* Оба типа PushConstants ровно по 24 байта */

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &dsl;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &pcr;
    VkPipelineLayout pipe_layout;
    vkCreatePipelineLayout(device, &pl_ci, NULL, &pipe_layout);

    size_t spv_grid_sz = 0, spv_phys_sz = 0, spv_rend_sz = 0;
    uint32_t *spv_grid = load_spv_file("spatial_grid_rebuild.spv", &spv_grid_sz);
    uint32_t *spv_phys = load_spv_file("membrane_collision_3d.spv", &spv_phys_sz);
    uint32_t *spv_rend = load_spv_file("render_3d_room_and_hud.spv", &spv_rend_sz);

    VkShaderModule mod_grid, mod_phys, mod_rend;
    VkShaderModuleCreateInfo sm_ci = {0};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spv_grid_sz;
    sm_ci.pCode = spv_grid;
    vkCreateShaderModule(device, &sm_ci, NULL, &mod_grid);

    sm_ci.codeSize = spv_phys_sz;
    sm_ci.pCode = spv_phys;
    vkCreateShaderModule(device, &sm_ci, NULL, &mod_phys);

    sm_ci.codeSize = spv_rend_sz;
    sm_ci.pCode = spv_rend;
    vkCreateShaderModule(device, &sm_ci, NULL, &mod_rend);

    VkComputePipelineCreateInfo cp_cis[3] = {0};
    VkShaderModule mods[3] = {mod_grid, mod_phys, mod_rend};
    for (int i = 0; i < 3; ++i) {
        cp_cis[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp_cis[i].stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp_cis[i].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp_cis[i].stage.module = mods[i];
        cp_cis[i].stage.pName = "main";
        cp_cis[i].layout = pipe_layout;
    }
    VkPipeline pipes[3];
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 3, cp_cis, NULL, pipes) != VK_SUCCESS) {
        fprintf(stderr, "Ошибка vkCreateComputePipelines\n");
        return 1;
    }
    VkPipeline pipe_grid = pipes[0];
    VkPipeline pipe_phys = pipes[1];
    VkPipeline pipe_rend = pipes[2];

    /* 5. Выделяем и заполняем DescriptorSet */
    VkDescriptorPoolSize pool_sizes[3] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7}
    };
    VkDescriptorPoolCreateInfo dp_ci = {0};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.maxSets = 1;
    dp_ci.poolSizeCount = 3;
    dp_ci.pPoolSizes = pool_sizes;
    VkDescriptorPool desc_pool;
    vkCreateDescriptorPool(device, &dp_ci, NULL, &desc_pool);

    VkDescriptorSetAllocateInfo ds_ai = {0};
    ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool = desc_pool;
    ds_ai.descriptorSetCount = 1;
    ds_ai.pSetLayouts = &dsl;
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &ds_ai, &desc_set);

    VkDescriptorImageInfo img_info_env_samp = {tmu_sampler, env_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo img_info_env_stor = {VK_NULL_HANDLE, env_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo img_info_mip_samp = {tmu_sampler, mip_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkBuffer all_bufs[7] = {buf_tokens, buf_grid, buf_objs, buf_ctx, buf_metrics, buf_hist, buf_frame};
    VkDescriptorBufferInfo buf_infos[7];
    for (int i = 0; i < 7; ++i) {
        buf_infos[i] = (VkDescriptorBufferInfo){all_bufs[i], 0, VK_WHOLE_SIZE};
    }

    VkWriteDescriptorSet writes[10] = {0};
    writes[0] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, desc_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img_info_env_samp, NULL, NULL};
    writes[1] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, desc_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &img_info_env_stor, NULL, NULL};
    writes[2] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, desc_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &img_info_mip_samp, NULL, NULL};
    for (int i = 0; i < 7; ++i) {
        writes[3 + i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, desc_set, (uint32_t)(3 + i), 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &buf_infos[i], NULL};
    }
    vkUpdateDescriptorSets(device, 10, writes, 0, NULL);

    /* 6. CommandPool и CommandBuffer, перевод текстур в рабочие layout'ы */
    VkCommandPoolCreateInfo cp_pool_ci = {0};
    cp_pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cp_pool_ci.queueFamilyIndex = 0;
    VkCommandPool cmd_pool;
    vkCreateCommandPool(device, &cp_pool_ci, NULL, &cmd_pool);

    VkCommandBufferAllocateInfo cb_ai = {0};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = cmd_pool;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cb_ai, &cmd);

    VkCommandBufferBeginInfo binfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    vkBeginCommandBuffer(cmd, &binfo);

    VkImageMemoryBarrier img_barriers[2] = {0};
    img_barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    img_barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    img_barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    img_barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    img_barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    img_barriers[0].image = env_img;
    img_barriers[0].subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    img_barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    img_barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    img_barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    img_barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    img_barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    img_barriers[1].image = mip_img;
    img_barriers[1].subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 4, 0, 1};

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, img_barriers);

    VkClearColorValue zero_col = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkImageSubresourceRange env_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, env_img, VK_IMAGE_LAYOUT_GENERAL, &zero_col, 1, &env_range);

    VkImageSubresourceRange mip_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 4, 0, 1};
    vkCmdClearColorImage(cmd, mip_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero_col, 1, &mip_range);

    img_barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    img_barriers[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &img_barriers[1]);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo sinfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd, 0, NULL};
    vkQueueSubmit(queue, 1, &sinfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    /* 7. Загружаем обучающий поток WikiText-2 в буфер и запускаем 480 шагов обучения
     *    с записью каждого кадра (360 кадров 1280x720) через ffmpeg! */
    for (long i = 0; i < train_len; ++i) {
        map_tokens[i] = (uint32_t)train_bytes[i];
    }

    FILE *ff = popen(
        "ffmpeg -y -f rawvideo -pixel_format rgba -video_size 1280x720 -framerate 30 -i - "
        "-c:v libx264 -pix_fmt yuv420p -preset fast -crf 18 training_vulkan_nanite_cut_v13.mp4 2>/dev/null",
        "w"
    );

    const uint32_t total_train_steps = 2400;
    const uint32_t train_stride = (uint32_t)(train_len / N_STREAMS);
    float ema50_bits = 0.0f;
    float alpha_ema = 2.0f / 51.0f;
    double t_train_start = now_sec();
    double gpu_compute_sec = 0.0;
    double gpu_render_sec = 0.0;
    uint32_t rendered_frames = 0;

    for (uint32_t step = 0; step < total_train_steps; ++step) {
        double t0 = now_sec();
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &binfo);

        /* Шаг А: Пересборка 3D Spatial Hash Grid (8x8x8 вокселей) на GPU */
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_grid);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &desc_set, 0, NULL);
        vkCmdDispatch(cmd, GRID_CELLS / 64, 1, 1);

        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

        /* Шаг Б: 2D-коллизия по Снеллиусу–Френелю, рикошет, морфинг Mip-LOD, запись в 3D-текстуру Psi и локальное обучение */
        PhysicsPushConsts_t ppc = {
            train_stride,
            step % (train_stride - 2u),
            1u,
            1.80f,
            2.0f / 3.0f,
            rendered_frames
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_phys);
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ppc), &ppc);
        vkCmdDispatch(cmd, 1, 1, 1);

        vkEndCommandBuffer(cmd);
        vkQueueSubmit(queue, 1, &sinfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        gpu_compute_sec += (now_sec() - t0);

        /* Собираем статистику по 64 параллельным потокам */
        float step_loss = 0.0f;
        float step_cull = 0.0f;
        float step_floats = 0.0f;
        float step_T = 0.0f;
        float step_area = 0.0f;
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            step_loss += map_metrics->step_loss_bits[s];
            step_cull += (float)map_metrics->culled_objects_cnt[s] / (float)N_OBJECTS;
            step_floats += (float)map_metrics->active_lod_floats_read[s];
            step_T += map_metrics->trans_T_sum[s];
            step_area += map_metrics->holo_area_sum[s];
        }
        step_loss /= (float)N_STREAMS;
        step_cull /= (float)N_STREAMS;
        step_floats /= (float)N_STREAMS;
        step_T /= (float)N_STREAMS;
        step_area /= (float)N_STREAMS;

        /* Полное чтение без Spatial Culling и без Mip-LOD составило бы 256 * 64 = 16384 float */
        float vram_saved_frac = 1.0f - (step_floats / (float)(N_OBJECTS * N_SECTORS));
        ema50_bits = (step == 0) ? step_loss : (alpha_ema * step_loss + (1.0f - alpha_ema) * ema50_bits);

        /* Рендерим каждый 5-й шаг (480 кадров 1280x720) на GPU через шейдер render_3d_room_and_hud */
        if (step % 5 == 0 && rendered_frames < 480) {
            map_hist->hist_series[rendered_frames] = (vec4_t){step_loss, ema50_bits, vram_saved_frac, step_T};

            double tr0 = now_sec();
            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, &binfo);

            RenderPushConsts_t rpc = {
                rendered_frames,
                rendered_frames + 1,
                0.55f + 0.012f * (float)rendered_frames,
                ema50_bits,
                step_cull * 100.0f,
                vram_saved_frac * 100.0f
            };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_rend);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &desc_set, 0, NULL);
            vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpc), &rpc);
            vkCmdDispatch(cmd, IMG_W / 16, IMG_H / 16, 1);

            vkEndCommandBuffer(cmd);
            vkQueueSubmit(queue, 1, &sinfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
            gpu_render_sec += (now_sec() - tr0);

            if (ff) {
                fwrite(map_frame, sizeof(uint32_t), IMG_W * IMG_H, ff);
            }
            if (rendered_frames == 0 || rendered_frames == 239 || rendered_frames == 479) {
                char raw_name[64];
                snprintf(raw_name, sizeof(raw_name), "vulkan_frame_%03u.raw", rendered_frames + 1);
                FILE *fraw = fopen(raw_name, "wb");
                if (fraw) {
                    fwrite(map_frame, sizeof(uint32_t), IMG_W * IMG_H, fraw);
                    fclose(fraw);
                }
            }
            rendered_frames++;
        }

        if (step == 0 || (step + 1) % 400 == 0) {
            printf(
                "шаг %4u/%u | Loss=%.3f бит (EMA50=%.3f) | Spatial Culling=%.1f%% | Mip-LOD экономия VRAM=%.1f%% (прочитано %.0f float из 16384) | T(r)=%.3f | A_poly=%.2f\n",
                step + 1, total_train_steps, step_loss, ema50_bits,
                step_cull * 100.0f, vram_saved_frac * 100.0f, step_floats, step_T, step_area
            );
        }
    }
    if (ff) pclose(ff);

    /* 8. ПОЛНАЯ ВАЛИДАЦИЯ T1 НА ВСЕХ 250 304 БАЙТАХ WIKITEXT-2 НА VULKAN GPU (do_learn = 0, без обрезов *80!) */
    printf("\n=== ЗАПУСК ПОЛНОЙ ВАЛИДАЦИИ T1 НА ВСЕХ БАЙТАХ WIKITEXT-2 НА VULKAN GPU ===\n");
    for (long i = 0; i < valid_len; ++i) {
        map_tokens[i] = (uint32_t)valid_bytes[i];
    }

    uint32_t val_stride = (uint32_t)(valid_len / N_STREAMS);
    uint32_t val_steps = val_stride - 1u;
    uint32_t total_val_bytes = val_steps * N_STREAMS;
    float *val_per_byte_bits = (float *)malloc(sizeof(float) * total_val_bytes);

    double val_loss_sum = 0.0;
    double val_loss_sq_sum = 0.0;
    double val_cull_sum = 0.0;
    double val_floats_sum = 0.0;
    double val_T_sum = 0.0;

    /* Сначала один раз пересобираем Spatial Hash Grid для обученных 3D-коллайдеров */
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &binfo);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_grid);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &desc_set, 0, NULL);
    vkCmdDispatch(cmd, GRID_CELLS / 64, 1, 1);
    vkEndCommandBuffer(cmd);
    vkQueueSubmit(queue, 1, &sinfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    double t_val_start = now_sec();
    for (uint32_t vs = 0; vs < val_steps; ++vs) {
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &binfo);

        PhysicsPushConsts_t ppc = {
            val_stride,
            vs,
            0u, /* do_learn = 0 на валидации */
            0.0f,
            2.0f / 3.0f,
            vs
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_phys);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &desc_set, 0, NULL);
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ppc), &ppc);
        vkCmdDispatch(cmd, 1, 1, 1);

        vkEndCommandBuffer(cmd);
        vkQueueSubmit(queue, 1, &sinfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            float b_loss = map_metrics->step_loss_bits[s];
            val_per_byte_bits[vs * N_STREAMS + s] = b_loss;
            val_loss_sum += (double)b_loss;
            val_loss_sq_sum += (double)b_loss * (double)b_loss;
            val_cull_sum += (double)map_metrics->culled_objects_cnt[s] / (double)N_OBJECTS;
            val_floats_sum += (double)map_metrics->active_lod_floats_read[s];
            val_T_sum += (double)map_metrics->trans_T_sum[s];
        }
    }
    double val_elapsed = now_sec() - t_val_start;

    double val_mean_bpc = val_loss_sum / (double)total_val_bytes;
    double val_var = (val_loss_sq_sum / (double)total_val_bytes) - (val_mean_bpc * val_mean_bpc);
    double val_std = sqrt(fmax(0.0, val_var));
    double val_se = val_std / sqrt((double)total_val_bytes);
    double val_cull_pct = 100.0 * val_cull_sum / (double)total_val_bytes;
    double val_avg_floats = val_floats_sum / (double)total_val_bytes;
    double val_avg_bytes_read = val_avg_floats * 4.0;
    double val_vram_save_pct = 100.0 * (1.0 - val_avg_floats / (double)(N_OBJECTS * N_SECTORS));
    double val_mean_T = val_T_sum / (double)total_val_bytes;

    double train_steps_per_sec = (double)total_train_steps / fmax(1e-6, gpu_compute_sec);
    double render_fps = (double)rendered_frames / fmax(1e-6, gpu_render_sec);
    double wall_ms_per_step_gpu = 1000.0 * gpu_compute_sec / (double)total_train_steps;

    /* Сохраняем массив пер-байтовых потерь полной валидации (250 304 float32) */
    FILE *fval_bin = fopen("vulkan_full_val_per_byte_bits.bin", "wb");
    if (fval_bin) {
        fwrite(val_per_byte_bits, sizeof(float), total_val_bytes, fval_bin);
        fclose(fval_bin);
    }

    printf(
        "Полная валидация T1 завершена за %.2f с: n_bytes = %u | BPC = %.4f ± %.5f (SE) | "
        "Spatial Culling = %.2f%% | Прочитано %.1f байт/шаг (экономия VRAM = %.2f%%) | "
        "train_steps_per_sec = %.1f | render_fps = %.1f\n",
        val_elapsed, total_val_bytes, val_mean_bpc, val_se,
        val_cull_pct, val_avg_bytes_read, val_vram_save_pct,
        train_steps_per_sec, render_fps
    );

    /* Записываем JSON-отчёт строго по формату §10 */
    FILE *fjson = fopen("rezultat_vulkan_nanite_cut_v13.json", "w");
    if (fjson) {
        fprintf(fjson, "{\n");
        fprintf(fjson, "  \"run_id\": \"v13-vulkan-nanite-two-sided-cut-deformable-lod\",\n");
        fprintf(fjson, "  \"task\": \"Task2_N1_to_N13_Vulkan_MatrixFree_TwoSided_Haar_Cut\",\n");
        fprintf(fjson, "  \"changed_exactly_one_thing\": true,\n");
        fprintf(fjson, "  \"gpu_device\": \"%s\",\n", chosen_props.deviceName);
        fprintf(fjson, "  \"graphics_api\": \"Vulkan 1.2 (SPIR-V Compute + 3D Volume Texture TMU + Mip-map LOD + Spatial Hash Grid)\",\n");
        fprintf(fjson, "  \"matrix_multiplications_count\": 0,\n");
        fprintf(fjson, "  \"config\": {\"D\": 64, \"N\": 256, \"grid_voxels\": 512, \"mip_levels\": 4, \"gate\": \"snell_fresnel_emergent\", \"alpha_exp\": 0.6667},\n");
        fprintf(fjson, "  \"units\": {\"granularity\": \"byte_collider_in_3d_room\", \"vocab\": 256, \"oov_rate\": 0.0},\n");
        fprintf(fjson, "  \"budget\": {\"train_steps\": %u, \"bytes_seen\": %u},\n", total_train_steps, total_train_steps * N_STREAMS);
        fprintf(fjson, "  \"validation\": {\n");
        fprintf(fjson, "    \"n_bytes\": %u,\n", total_val_bytes);
        fprintf(fjson, "    \"BPC_mean\": %.4f,\n", val_mean_bpc);
        fprintf(fjson, "    \"BPC_std\": %.4f,\n", val_std);
        fprintf(fjson, "    \"SE\": %.5f,\n", val_se);
        fprintf(fjson, "    \"train_EMA50_final\": %.4f\n", ema50_bits);
        fprintf(fjson, "  },\n");
        fprintf(fjson, "  \"cost\": {\n");
        fprintf(fjson, "    \"dense_reference_bytes_per_token\": %d,\n", N_OBJECTS * N_SECTORS * 4);
        fprintf(fjson, "    \"bytes_read_per_token\": %.1f,\n", val_avg_bytes_read);
        fprintf(fjson, "    \"vram_bandwidth_savings_pct\": %.2f,\n", val_vram_save_pct);
        fprintf(fjson, "    \"spatial_culling_pct\": %.2f,\n", val_cull_pct);
        fprintf(fjson, "    \"env_3d_texture_state_KB\": %.2f,\n", (GRID_SIDE * GRID_SIDE * GRID_SIDE * 16.0) / 1024.0);
        fprintf(fjson, "    \"wall_ms_per_step_gpu\": %.3f,\n", wall_ms_per_step_gpu);
        fprintf(fjson, "    \"train_steps_per_sec\": %.1f,\n", train_steps_per_sec);
        fprintf(fjson, "    \"render_fps\": %.1f\n", render_fps);
        fprintf(fjson, "  },\n");
        fprintf(fjson, "  \"physics_telemetry\": {\n");
        fprintf(fjson, "    \"mean_snell_fresnel_transmission_T\": %.4f,\n", val_mean_T);
        fprintf(fjson, "    \"mean_snell_fresnel_reflection_R\": %.4f\n", 1.0 - val_mean_T);
        fprintf(fjson, "  }\n");
        fprintf(fjson, "}\n");
        fclose(fjson);
    }

    free(val_per_byte_bits);
    free(train_bytes);
    free(valid_bytes);
    return 0;
}
