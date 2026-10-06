#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreMedia/CoreMedia.h>

static void print_bool(const char *name, bool value) {
    printf("%s=%s\n", name, value ? "true" : "false");
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        const char *device_name = device ? [[device name] UTF8String] : "NONE";
        printf("metal_device=%s\n", device_name);

        bool h264_decode = VTIsHardwareDecodeSupported(kCMVideoCodecType_H264);
        bool hevc_decode = VTIsHardwareDecodeSupported(kCMVideoCodecType_HEVC);
        print_bool("videotoolbox_h264_decode", h264_decode);
        print_bool("videotoolbox_hevc_decode", hevc_decode);

        CVMetalTextureCacheRef cache = NULL;
        CVReturn cache_status = kCVReturnError;
        if (device) {
            cache_status = CVMetalTextureCacheCreate(
                kCFAllocatorDefault, NULL, device, NULL, &cache);
        }
        printf("cvmetal_texture_cache_status=%d\n", (int)cache_status);

        NSDictionary *attrs = @{
            (NSString *)kCVPixelBufferMetalCompatibilityKey: @YES,
            (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{}
        };

        CVPixelBufferRef pixel_buffer = NULL;
        CVReturn pb_status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            3840,
            2160,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            (__bridge CFDictionaryRef)attrs,
            &pixel_buffer);
        printf("cvpixelbuffer_status=%d\n", (int)pb_status);

        CVReturn y_status = kCVReturnError;
        CVReturn uv_status = kCVReturnError;
        CVMetalTextureRef y_tex = NULL;
        CVMetalTextureRef uv_tex = NULL;

        if (cache && pixel_buffer) {
            y_status = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault,
                cache,
                pixel_buffer,
                NULL,
                MTLPixelFormatR8Unorm,
                3840,
                2160,
                0,
                &y_tex);

            uv_status = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault,
                cache,
                pixel_buffer,
                NULL,
                MTLPixelFormatRG8Unorm,
                1920,
                1080,
                1,
                &uv_tex);
        }

        printf("metal_y_plane_status=%d\n", (int)y_status);
        printf("metal_uv_plane_status=%d\n", (int)uv_status);

        bool y_ok = y_tex && CVMetalTextureGetTexture(y_tex) != nil;
        bool uv_ok = uv_tex && CVMetalTextureGetTexture(uv_tex) != nil;
        print_bool("metal_y_plane_visible", y_ok);
        print_bool("metal_uv_plane_visible", uv_ok);

        if (y_tex) CFRelease(y_tex);
        if (uv_tex) CFRelease(uv_tex);
        if (pixel_buffer) CVPixelBufferRelease(pixel_buffer);
        if (cache) CFRelease(cache);

        bool pass = device && h264_decode && cache_status == kCVReturnSuccess &&
                    pb_status == kCVReturnSuccess && y_ok && uv_ok;
        print_bool("puma_t0_capability_pass", pass);
        return pass ? 0 : 2;
    }
}
