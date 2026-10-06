#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#import <mach/mach_time.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static mach_timebase_info_data_t g_timebase;

static inline uint64_t now_ns(void) {
    uint64_t t = mach_continuous_time();
    return t * g_timebase.numer / g_timebase.denom;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static double median_ns(uint64_t *v, int n) {
    qsort(v, (size_t)n, sizeof(uint64_t), cmp_u64);
    if (n & 1) return (double)v[n / 2];
    return ((double)v[n / 2 - 1] + (double)v[n / 2]) / 2.0;
}

static double percentile_ns(uint64_t *v, int n, double p) {
    qsort(v, (size_t)n, sizeof(uint64_t), cmp_u64);
    int idx = (int)((n - 1) * p);
    return (double)v[idx];
}

static CVPixelBufferRef make_nv12(size_t width, size_t height) {
    NSDictionary *attrs = @{
        (NSString *)kCVPixelBufferMetalCompatibilityKey: @YES,
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };
    CVPixelBufferRef pb = NULL;
    CVReturn r = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        (__bridge CFDictionaryRef)attrs,
        &pb);
    if (r != kCVReturnSuccess || !pb) {
        fprintf(stderr, "CVPixelBufferCreate failed: %d\n", (int)r);
        exit(2);
    }

    CVPixelBufferLockBaseAddress(pb, 0);
    for (size_t plane = 0; plane < CVPixelBufferGetPlaneCount(pb); plane++) {
        uint8_t *base = CVPixelBufferGetBaseAddressOfPlane(pb, plane);
        size_t bytes = CVPixelBufferGetBytesPerRowOfPlane(pb, plane) *
                       CVPixelBufferGetHeightOfPlane(pb, plane);
        memset(base, (int)(17 + plane * 101), bytes);
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return pb;
}

static void copy_nv12_packed(CVPixelBufferRef pb, uint8_t *dst,
                             size_t width, size_t height) {
    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);

    uint8_t *out = dst;
    const uint8_t *y = CVPixelBufferGetBaseAddressOfPlane(pb, 0);
    size_t y_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
    for (size_t row = 0; row < height; row++) {
        memcpy(out, y + row * y_stride, width);
        out += width;
    }

    const uint8_t *uv = CVPixelBufferGetBaseAddressOfPlane(pb, 1);
    size_t uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
    for (size_t row = 0; row < height / 2; row++) {
        memcpy(out, uv + row * uv_stride, width);
        out += width;
    }

    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
}

static double bench_copy_reuse(CVPixelBufferRef pb, size_t w, size_t h,
                               int iters, double *p95_out) {
    size_t bytes = w * h * 3 / 2;
    uint8_t *dst = malloc(bytes);
    uint64_t *times = malloc((size_t)iters * sizeof(uint64_t));
    volatile uint64_t sink = 0;

    for (int i = 0; i < 10; i++) copy_nv12_packed(pb, dst, w, h);

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = now_ns();
        copy_nv12_packed(pb, dst, w, h);
        uint64_t t1 = now_ns();
        sink += dst[0] + dst[bytes - 1];
        times[i] = t1 - t0;
    }

    double med = median_ns(times, iters);
    *p95_out = percentile_ns(times, iters, 0.95);
    if (sink == 0xdeadbeef) fprintf(stderr, "%llu\n", sink);
    free(times);
    free(dst);
    return med;
}

static double bench_copy_alloc(CVPixelBufferRef pb, size_t w, size_t h,
                               int iters, double *p95_out) {
    size_t bytes = w * h * 3 / 2;
    uint64_t *times = malloc((size_t)iters * sizeof(uint64_t));
    volatile uint64_t sink = 0;

    for (int i = 0; i < 5; i++) {
        uint8_t *dst = malloc(bytes);
        copy_nv12_packed(pb, dst, w, h);
        sink += dst[0];
        free(dst);
    }

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = now_ns();
        uint8_t *dst = malloc(bytes);
        copy_nv12_packed(pb, dst, w, h);
        sink += dst[0] + dst[bytes - 1];
        free(dst);
        uint64_t t1 = now_ns();
        times[i] = t1 - t0;
    }

    double med = median_ns(times, iters);
    *p95_out = percentile_ns(times, iters, 0.95);
    if (sink == 0xdeadbeef) fprintf(stderr, "%llu\n", sink);
    free(times);
    return med;
}

static double bench_metal_map(CVPixelBufferRef pb, CVMetalTextureCacheRef cache,
                              size_t w, size_t h, int iters,
                              double *p95_out) {
    uint64_t *times = malloc((size_t)iters * sizeof(uint64_t));
    volatile NSUInteger sink = 0;

    for (int i = 0; i < 20; i++) {
        CVMetalTextureRef y_ref = NULL;
        CVMetalTextureRef uv_ref = NULL;
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pb, NULL,
            MTLPixelFormatR8Unorm, w, h, 0, &y_ref);
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pb, NULL,
            MTLPixelFormatRG8Unorm, w / 2, h / 2, 1, &uv_ref);
        id<MTLTexture> y = CVMetalTextureGetTexture(y_ref);
        id<MTLTexture> uv = CVMetalTextureGetTexture(uv_ref);
        sink += y.width + uv.width;
        if (y_ref) CFRelease(y_ref);
        if (uv_ref) CFRelease(uv_ref);
    }

    for (int i = 0; i < iters; i++) {
        uint64_t t0 = now_ns();
        CVMetalTextureRef y_ref = NULL;
        CVMetalTextureRef uv_ref = NULL;
        CVReturn yr = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pb, NULL,
            MTLPixelFormatR8Unorm, w, h, 0, &y_ref);
        CVReturn uvr = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pb, NULL,
            MTLPixelFormatRG8Unorm, w / 2, h / 2, 1, &uv_ref);
        if (yr != kCVReturnSuccess || uvr != kCVReturnSuccess ||
            !y_ref || !uv_ref) {
            fprintf(stderr, "Metal map failed: %d %d\n", (int)yr, (int)uvr);
            exit(3);
        }
        id<MTLTexture> y = CVMetalTextureGetTexture(y_ref);
        id<MTLTexture> uv = CVMetalTextureGetTexture(uv_ref);
        sink += y.width + uv.width;
        CFRelease(y_ref);
        CFRelease(uv_ref);
        uint64_t t1 = now_ns();
        times[i] = t1 - t0;
    }

    double med = median_ns(times, iters);
    *p95_out = percentile_ns(times, iters, 0.95);
    if (sink == 0xdeadbeef) fprintf(stderr, "%lu\n", (unsigned long)sink);
    free(times);
    return med;
}

static void run_case(id<MTLDevice> device, CVMetalTextureCacheRef cache,
                     size_t w, size_t h, int iters, bool comma) {
    CVPixelBufferRef pb = make_nv12(w, h);
    size_t bytes = w * h * 3 / 2;

    double copy_reuse_p95 = 0;
    double copy_alloc_p95 = 0;
    double map_p95 = 0;
    double copy_reuse = bench_copy_reuse(pb, w, h, iters, &copy_reuse_p95);
    double copy_alloc = bench_copy_alloc(pb, w, h, iters, &copy_alloc_p95);
    double map = bench_metal_map(pb, cache, w, h, iters, &map_p95);

    double gb = (double)bytes / 1e9;
    double reuse_gbps = gb / (copy_reuse / 1e9);

    printf("%s{\n", comma ? "," : "");
    printf("  \"width\": %zu, \"height\": %zu, \"bytes\": %zu,\n", w, h, bytes);
    printf("  \"copy_reuse_median_us\": %.3f, \"copy_reuse_p95_us\": %.3f,\n",
           copy_reuse / 1000.0, copy_reuse_p95 / 1000.0);
    printf("  \"copy_alloc_median_us\": %.3f, \"copy_alloc_p95_us\": %.3f,\n",
           copy_alloc / 1000.0, copy_alloc_p95 / 1000.0);
    printf("  \"metal_map_median_us\": %.3f, \"metal_map_p95_us\": %.3f,\n",
           map / 1000.0, map_p95 / 1000.0);
    printf("  \"copy_reuse_over_map\": %.2f,\n", copy_reuse / map);
    printf("  \"copy_alloc_over_map\": %.2f,\n", copy_alloc / map);
    printf("  \"copy_reuse_GBps\": %.2f\n", reuse_gbps);
    printf("}\n");

    CVPixelBufferRelease(pb);
}

int main(void) {
    @autoreleasepool {
        mach_timebase_info(&g_timebase);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "No Metal device\n");
            return 2;
        }

        CVMetalTextureCacheRef cache = NULL;
        CVReturn cr = CVMetalTextureCacheCreate(
            kCFAllocatorDefault, NULL, device, NULL, &cache);
        if (cr != kCVReturnSuccess || !cache) {
            fprintf(stderr, "No CVMetalTextureCache: %d\n", (int)cr);
            return 3;
        }

        printf("{\n");
        printf("\"metal_device\": \"%s\",\n", [[device name] UTF8String]);
        printf("\"cases\": [\n");
        run_case(device, cache, 1920, 1080, 1000, false);
        run_case(device, cache, 3840, 2160, 300, true);
        run_case(device, cache, 7680, 4320, 80, true);
        printf("]\n}\n");

        CFRelease(cache);
        return 0;
    }
}
