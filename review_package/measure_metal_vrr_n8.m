#import <Metal/Metal.h>
#include <stdio.h>
#include <math.h>

int main(void) {
    NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
    id<MTLDevice> radeon = nil;
    for (id<MTLDevice> d in devs) {
        if (![d isLowPower]) { radeon = d; break; }
    }
    if (!radeon) radeon = MTLCreateSystemDefaultDevice();

    BOOL supported = [radeon supportsRasterizationRateMapWithLayerCount:1];
    printf("N8 Metal VRR Check on %s: supportsRasterizationRateMapWithLayerCount(1) = %s\n",
           [[radeon name] UTF8String], supported ? "true" : "false");

    if (!supported) return 0;

    /* Экран мембраны 512x512 (8x8 секторов по 64x64 пикселя), частоты зон берутся из уровней keep N2:
     * уровень m=64 -> rate 1.0; m=16..32 -> rate 0.5; m=1..8 -> rate 0.25 */
    MTLSize screenSize = MTLSizeMake(512, 512, 1);
    MTLRasterizationRateMapDescriptor *desc = [MTLRasterizationRateMapDescriptor rasterizationRateMapDescriptorWithScreenSize:screenSize];
    desc.label = @"MembraneHaarLOD_VRR";

    MTLRasterizationRateLayerDescriptor *layer = [[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:MTLSizeMake(8, 8, 1)];
    float rates_8[8] = {0.25f, 0.25f, 0.50f, 1.00f, 1.00f, 0.50f, 0.25f, 0.25f};
    for (int i = 0; i < 8; ++i) {
        layer.horizontalSampleStorage[i] = rates_8[i];
        layer.verticalSampleStorage[i] = rates_8[i];
    }
    [desc setLayer:layer atIndex:0];

    id<MTLRasterizationRateMap> rateMap = [radeon newRasterizationRateMapWithDescriptor:desc];
    MTLSize physSize = [rateMap physicalSizeForLayer:0];
    MTLSize gran = [rateMap physicalGranularity];

    long full_pixels = (long)screenSize.width * (long)screenSize.height;
    long phys_pixels = (long)physSize.width * (long)physSize.height;
    long saved_calls = full_pixels - phys_pixels;
    double saved_pct = 100.0 * (double)saved_calls / (double)full_pixels;

    printf("screenSize = %lux%lu (%ld fragment calls, %ld bytes RGBA32F)\n",
           screenSize.width, screenSize.height, full_pixels, full_pixels * 16L);
    printf("physicalSize = %lux%lu (granularity %lux%lu) (%ld fragment calls, %ld bytes RGBA32F)\n",
           physSize.width, physSize.height, gran.width, gran.height, phys_pixels, phys_pixels * 16L);
    printf("fragment_calls_saved = %ld (%.2f%% reduction)\n", saved_calls, saved_pct);

    FILE *fp = fopen("/Users/corpuscul/Desktop/ALM/MarkovMembrane-Engine/review_package/metal_vrr_n8_result.json", "w");
    if (fp) {
        fprintf(fp, "{\n");
        fprintf(fp, "  \"gpu_name\": \"%s\",\n", [[radeon name] UTF8String]);
        fprintf(fp, "  \"metal_vrr_supported\": %s,\n", supported ? "true" : "false");
        fprintf(fp, "  \"screen_size\": [%lu, %lu],\n", screenSize.width, screenSize.height);
        fprintf(fp, "  \"physical_size\": [%lu, %lu],\n", physSize.width, physSize.height);
        fprintf(fp, "  \"physical_granularity\": [%lu, %lu],\n", gran.width, gran.height);
        fprintf(fp, "  \"full_rate_fragment_calls\": %ld,\n", full_pixels);
        fprintf(fp, "  \"vrr_physical_fragment_calls\": %ld,\n", phys_pixels);
        fprintf(fp, "  \"metal_vrr_fragment_calls_saved\": %ld,\n", saved_calls);
        fprintf(fp, "  \"metal_vrr_savings_pct\": %.2f,\n", saved_pct);
        fprintf(fp, "  \"intermediate_texture_bytes_full\": %ld,\n", full_pixels * 16L);
        fprintf(fp, "  \"intermediate_texture_bytes_vrr\": %ld\n", phys_pixels * 16L);
        fprintf(fp, "}\n");
        fclose(fp);
    }
    return 0;
}
