#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_sample_pipeline = nil;
static CVMetalTextureCacheRef g_cache = NULL;
static CVMetalBufferCacheRef g_buffer_cache = NULL;

static PyObject *map_once(PyObject *self, PyObject *arg) {
    unsigned long long raw = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        return NULL;
    }

    CVPixelBufferRef pb = (CVPixelBufferRef)(uintptr_t)raw;
    if (pb == NULL) {
        PyErr_SetString(PyExc_ValueError, "null CVPixelBufferRef");
        return NULL;
    }

    size_t planes = CVPixelBufferGetPlaneCount(pb);
    if (planes < 2) {
        PyErr_SetString(PyExc_ValueError, "expected bi-planar VideoToolbox frame");
        return NULL;
    }

    size_t width = CVPixelBufferGetWidth(pb);
    size_t height = CVPixelBufferGetHeight(pb);

    CVMetalTextureRef y_ref = NULL;
    CVMetalTextureRef uv_ref = NULL;

    CVReturn yr = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        g_cache,
        pb,
        NULL,
        MTLPixelFormatR8Unorm,
        CVPixelBufferGetWidthOfPlane(pb, 0),
        CVPixelBufferGetHeightOfPlane(pb, 0),
        0,
        &y_ref);

    CVReturn uvr = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        g_cache,
        pb,
        NULL,
        MTLPixelFormatRG8Unorm,
        CVPixelBufferGetWidthOfPlane(pb, 1),
        CVPixelBufferGetHeightOfPlane(pb, 1),
        1,
        &uv_ref);

    if (yr != kCVReturnSuccess || uvr != kCVReturnSuccess ||
        y_ref == NULL || uv_ref == NULL) {
        if (y_ref) CFRelease(y_ref);
        if (uv_ref) CFRelease(uv_ref);
        PyErr_Format(
            PyExc_RuntimeError,
            "CVMetalTextureCacheCreateTextureFromImage failed: y=%d uv=%d",
            (int)yr,
            (int)uvr);
        return NULL;
    }

    id<MTLTexture> y = CVMetalTextureGetTexture(y_ref);
    id<MTLTexture> uv = CVMetalTextureGetTexture(uv_ref);
    if (y == nil || uv == nil) {
        CFRelease(y_ref);
        CFRelease(uv_ref);
        PyErr_SetString(PyExc_RuntimeError, "Metal texture view missing");
        return NULL;
    }

    NSUInteger yw = y.width;
    NSUInteger yh = y.height;
    NSUInteger uvw = uv.width;
    NSUInteger uvh = uv.height;

    CFRelease(y_ref);
    CFRelease(uv_ref);

    return Py_BuildValue(
        "(KKKKKK)",
        (unsigned long long)width,
        (unsigned long long)height,
        (unsigned long long)yw,
        (unsigned long long)yh,
        (unsigned long long)uvw,
        (unsigned long long)uvh);
}


typedef struct {
    int device_type;
    int device_id;
} PumaDLDevice;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} PumaDLDataType;

typedef struct {
    void *data;
    PumaDLDevice device;
    int32_t ndim;
    PumaDLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
} PumaDLTensor;

typedef struct PumaDLManagedTensor {
    PumaDLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(struct PumaDLManagedTensor *self);
} PumaDLManagedTensor;

typedef struct {
    PumaDLManagedTensor managed;
    CVMetalBufferRef cvbuf;
    int64_t shape[3];
    int64_t strides[3];
} PumaMetalDLContext;

enum {
    PUMA_KDL_UINT = 1,
    PUMA_KDL_METAL = 8,
};

static void puma_dlpack_deleter(PumaDLManagedTensor *managed) {
    if (managed == NULL || managed->manager_ctx == NULL) {
        return;
    }
    PumaMetalDLContext *ctx = (PumaMetalDLContext *)managed->manager_ctx;
    if (ctx->cvbuf != NULL) {
        CFRelease(ctx->cvbuf);
        ctx->cvbuf = NULL;
    }
    free(ctx);
}

static void puma_dlpack_capsule_destructor(PyObject *capsule) {
    const char *name = PyCapsule_GetName(capsule);
    if (name == NULL) {
        PyErr_Clear();
        return;
    }
    if (strcmp(name, "dltensor") != 0) {
        return;
    }
    PumaDLManagedTensor *managed =
        (PumaDLManagedTensor *)PyCapsule_GetPointer(capsule, "dltensor");
    if (managed == NULL) {
        PyErr_Clear();
        return;
    }
    if (managed->deleter != NULL) {
        managed->deleter(managed);
    }
}


static int puma_plane_offset(
    IOSurfaceRef surface,
    size_t plane,
    uint64_t *offset_out)
{
    CFTypeRef value = IOSurfaceCopyValue(surface, kIOSurfacePlaneInfo);
    if (value != NULL && CFGetTypeID(value) == CFArrayGetTypeID()) {
        CFArrayRef planes = (CFArrayRef)value;
        if (plane < (size_t)CFArrayGetCount(planes)) {
            CFTypeRef entry = CFArrayGetValueAtIndex(planes, (CFIndex)plane);
            if (entry != NULL &&
                CFGetTypeID(entry) == CFDictionaryGetTypeID()) {
                CFTypeRef base = CFDictionaryGetValue(
                    (CFDictionaryRef)entry, kIOSurfacePlaneBase);
                if (base != NULL &&
                    CFGetTypeID(base) == CFNumberGetTypeID()) {
                    int64_t offset = 0;
                    if (CFNumberGetValue(
                            (CFNumberRef)base,
                            kCFNumberSInt64Type,
                            &offset) &&
                        offset >= 0) {
                        *offset_out = (uint64_t)offset;
                        CFRelease(value);
                        return 0;
                    }
                }
            }
        }
        CFRelease(value);
    }
    else if (value != NULL) {
        CFRelease(value);
    }

    /* Fallback for IOSurfaces whose plane dictionary omits kIOSurfacePlaneBase.
       We only use this to derive metadata; no pixel bytes are copied. */
    uint32_t seed = 0;
    kern_return_t kr = IOSurfaceLock(
        surface, kIOSurfaceLockReadOnly, &seed);
    if (kr != KERN_SUCCESS) {
        return -1;
    }
    void *base = IOSurfaceGetBaseAddress(surface);
    void *plane_base = IOSurfaceGetBaseAddressOfPlane(surface, plane);
    int ok = 0;
    if (base != NULL && plane_base != NULL &&
        (uintptr_t)plane_base >= (uintptr_t)base) {
        *offset_out =
            (uint64_t)((uintptr_t)plane_base - (uintptr_t)base);
        ok = 1;
    }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, &seed);
    return ok ? 0 : -1;
}

static PyObject *plane_layout(PyObject *self, PyObject *arg) {
    unsigned long long raw = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        return NULL;
    }
    CVPixelBufferRef pb = (CVPixelBufferRef)(uintptr_t)raw;
    if (pb == NULL) {
        PyErr_SetString(PyExc_ValueError, "null CVPixelBufferRef");
        return NULL;
    }
    IOSurfaceRef surface = CVPixelBufferGetIOSurface(pb);
    if (surface == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "CVPixelBuffer has no IOSurface");
        return NULL;
    }

    size_t count = CVPixelBufferGetPlaneCount(pb);
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (list == NULL) {
        return NULL;
    }

    for (size_t plane = 0; plane < count; ++plane) {
        uint64_t offset = 0;
        if (puma_plane_offset(surface, plane, &offset) < 0) {
            Py_DECREF(list);
            PyErr_Format(
                PyExc_RuntimeError,
                "cannot determine IOSurface offset for plane %zu",
                plane);
            return NULL;
        }
        size_t width = CVPixelBufferGetWidthOfPlane(pb, plane);
        size_t height = CVPixelBufferGetHeightOfPlane(pb, plane);
        size_t bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, plane);
        size_t bpe = IOSurfaceGetBytesPerElementOfPlane(surface, plane);
        if (bpe == 0) {
            bpe = 1;
        }
        PyObject *entry = Py_BuildValue(
            "(KKKKK)",
            (unsigned long long)offset,
            (unsigned long long)width,
            (unsigned long long)height,
            (unsigned long long)bpr,
            (unsigned long long)bpe);
        if (entry == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, (Py_ssize_t)plane, entry);
    }
    return list;
}

static PyObject *make_dlpack_plane_capsule(
    PyObject *self, PyObject *args)
{
    unsigned long long raw = 0;
    Py_ssize_t plane_index = 0;
    if (!PyArg_ParseTuple(args, "Kn", &raw, &plane_index)) {
        return NULL;
    }

    CVPixelBufferRef pb = (CVPixelBufferRef)(uintptr_t)raw;
    if (pb == NULL) {
        PyErr_SetString(PyExc_ValueError, "null CVPixelBufferRef");
        return NULL;
    }
    if (plane_index < 0 ||
        (size_t)plane_index >= CVPixelBufferGetPlaneCount(pb)) {
        PyErr_SetString(PyExc_IndexError, "invalid CVPixelBuffer plane");
        return NULL;
    }
    if (g_buffer_cache == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "CVMetalBufferCache unavailable");
        return NULL;
    }

    IOSurfaceRef surface = CVPixelBufferGetIOSurface(pb);
    if (surface == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "CVPixelBuffer has no IOSurface");
        return NULL;
    }

    CVMetalBufferRef cvbuf = NULL;
    CVReturn r = CVMetalBufferCacheCreateBufferFromImage(
        kCFAllocatorDefault, g_buffer_cache, pb, &cvbuf);
    if (r != kCVReturnSuccess || cvbuf == NULL) {
        if (cvbuf != NULL) {
            CFRelease(cvbuf);
        }
        PyErr_Format(
            PyExc_RuntimeError,
            "CVMetalBufferCacheCreateBufferFromImage failed: %d",
            (int)r);
        return NULL;
    }

    id<MTLBuffer> buffer = CVMetalBufferGetBuffer(cvbuf);
    if (buffer == nil) {
        CFRelease(cvbuf);
        PyErr_SetString(PyExc_RuntimeError, "CVMetalBuffer has no MTLBuffer");
        return NULL;
    }

    uint64_t offset = 0;
    if (puma_plane_offset(surface, (size_t)plane_index, &offset) < 0) {
        CFRelease(cvbuf);
        PyErr_SetString(
            PyExc_RuntimeError,
            "cannot determine IOSurface plane offset");
        return NULL;
    }

    size_t width = CVPixelBufferGetWidthOfPlane(pb, (size_t)plane_index);
    size_t height = CVPixelBufferGetHeightOfPlane(pb, (size_t)plane_index);
    size_t bpr =
        CVPixelBufferGetBytesPerRowOfPlane(pb, (size_t)plane_index);
    size_t bpe =
        IOSurfaceGetBytesPerElementOfPlane(surface, (size_t)plane_index);
    if (bpe == 0) {
        bpe = 1;
    }

    uint64_t active_end =
        offset +
        (height > 0 ? (uint64_t)(height - 1) * bpr : 0) +
        (uint64_t)width * bpe;
    if (active_end > (uint64_t)buffer.length) {
        CFRelease(cvbuf);
        PyErr_Format(
            PyExc_RuntimeError,
            "plane %zd exceeds Metal buffer: end=%llu length=%llu",
            plane_index,
            (unsigned long long)active_end,
            (unsigned long long)buffer.length);
        return NULL;
    }

    PumaMetalDLContext *ctx =
        (PumaMetalDLContext *)calloc(1, sizeof(PumaMetalDLContext));
    if (ctx == NULL) {
        CFRelease(cvbuf);
        return PyErr_NoMemory();
    }

    ctx->cvbuf = cvbuf;
    if (bpe == 1) {
        ctx->shape[0] = (int64_t)height;
        ctx->shape[1] = (int64_t)width;
        ctx->strides[0] = (int64_t)bpr;
        ctx->strides[1] = 1;
        ctx->managed.dl_tensor.ndim = 2;
    }
    else {
        ctx->shape[0] = (int64_t)height;
        ctx->shape[1] = (int64_t)width;
        ctx->shape[2] = (int64_t)bpe;
        ctx->strides[0] = (int64_t)bpr;
        ctx->strides[1] = (int64_t)bpe;
        ctx->strides[2] = 1;
        ctx->managed.dl_tensor.ndim = 3;
    }

    ctx->managed.dl_tensor.data = (__bridge void *)buffer;
    ctx->managed.dl_tensor.device.device_type = PUMA_KDL_METAL;
    ctx->managed.dl_tensor.device.device_id = 0;
    ctx->managed.dl_tensor.dtype.code = PUMA_KDL_UINT;
    ctx->managed.dl_tensor.dtype.bits = 8;
    ctx->managed.dl_tensor.dtype.lanes = 1;
    ctx->managed.dl_tensor.shape = ctx->shape;
    ctx->managed.dl_tensor.strides = ctx->strides;
    ctx->managed.dl_tensor.byte_offset = offset;
    ctx->managed.manager_ctx = ctx;
    ctx->managed.deleter = puma_dlpack_deleter;

    PyObject *capsule = PyCapsule_New(
        &ctx->managed, "dltensor", puma_dlpack_capsule_destructor);
    if (capsule == NULL) {
        puma_dlpack_deleter(&ctx->managed);
        return NULL;
    }
    return capsule;
}

static PyObject *make_dlpack_capsule(PyObject *self, PyObject *arg) {
    unsigned long long raw = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        return NULL;
    }

    CVPixelBufferRef pb = (CVPixelBufferRef)(uintptr_t)raw;
    if (pb == NULL) {
        PyErr_SetString(PyExc_ValueError, "null CVPixelBufferRef");
        return NULL;
    }
    if (g_buffer_cache == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "CVMetalBufferCache unavailable");
        return NULL;
    }

    CVMetalBufferRef cvbuf = NULL;
    CVReturn r = CVMetalBufferCacheCreateBufferFromImage(
        kCFAllocatorDefault,
        g_buffer_cache,
        pb,
        &cvbuf);
    if (r != kCVReturnSuccess || cvbuf == NULL) {
        if (cvbuf != NULL) {
            CFRelease(cvbuf);
        }
        PyErr_Format(
            PyExc_RuntimeError,
            "CVMetalBufferCacheCreateBufferFromImage failed: %d",
            (int)r);
        return NULL;
    }

    id<MTLBuffer> buffer = CVMetalBufferGetBuffer(cvbuf);
    if (buffer == nil) {
        CFRelease(cvbuf);
        PyErr_SetString(PyExc_RuntimeError, "CVMetalBuffer has no MTLBuffer");
        return NULL;
    }

    PumaMetalDLContext *ctx =
        (PumaMetalDLContext *)calloc(1, sizeof(PumaMetalDLContext));
    if (ctx == NULL) {
        CFRelease(cvbuf);
        return PyErr_NoMemory();
    }

    ctx->cvbuf = cvbuf;  /* +1 from CreateBufferFromImage; owned by capsule. */
    ctx->shape[0] = (int64_t)buffer.length;
    ctx->strides[0] = 1;

    ctx->managed.dl_tensor.data = (__bridge void *)buffer;
    ctx->managed.dl_tensor.device.device_type = PUMA_KDL_METAL;
    ctx->managed.dl_tensor.device.device_id = 0;
    ctx->managed.dl_tensor.ndim = 1;
    ctx->managed.dl_tensor.dtype.code = PUMA_KDL_UINT;
    ctx->managed.dl_tensor.dtype.bits = 8;
    ctx->managed.dl_tensor.dtype.lanes = 1;
    ctx->managed.dl_tensor.shape = ctx->shape;
    ctx->managed.dl_tensor.strides = ctx->strides;
    ctx->managed.dl_tensor.byte_offset = 0;
    ctx->managed.manager_ctx = ctx;
    ctx->managed.deleter = puma_dlpack_deleter;

    PyObject *capsule = PyCapsule_New(
        &ctx->managed, "dltensor", puma_dlpack_capsule_destructor);
    if (capsule == NULL) {
        puma_dlpack_deleter(&ctx->managed);
        return NULL;
    }
    return capsule;
}

static PyObject *inspect_dlpack_capsule(PyObject *self, PyObject *arg) {
    PumaDLManagedTensor *managed =
        (PumaDLManagedTensor *)PyCapsule_GetPointer(arg, "dltensor");
    if (managed == NULL) {
        return NULL;
    }
    PumaDLTensor *t = &managed->dl_tensor;
    unsigned long long data_handle =
        (unsigned long long)(uintptr_t)t->data;
    unsigned long long shape0 =
        (t->ndim > 0 && t->shape != NULL)
            ? (unsigned long long)t->shape[0]
            : 0;
    return Py_BuildValue(
        "(iiKK)",
        t->device.device_type,
        t->device.device_id,
        data_handle,
        shape0);
}

static PyObject *map_buffer_once(PyObject *self, PyObject *arg) {
    unsigned long long raw = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        return NULL;
    }

    CVPixelBufferRef pb = (CVPixelBufferRef)(uintptr_t)raw;
    if (pb == NULL) {
        PyErr_SetString(PyExc_ValueError, "null CVPixelBufferRef");
        return NULL;
    }
    if (g_buffer_cache == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "CVMetalBufferCache unavailable");
        return NULL;
    }

    CVMetalBufferRef cvbuf = NULL;
    CVReturn r = CVMetalBufferCacheCreateBufferFromImage(
        kCFAllocatorDefault,
        g_buffer_cache,
        pb,
        &cvbuf);
    if (r != kCVReturnSuccess || cvbuf == NULL) {
        if (cvbuf) CFRelease(cvbuf);
        PyErr_Format(
            PyExc_RuntimeError,
            "CVMetalBufferCacheCreateBufferFromImage failed: %d",
            (int)r);
        return NULL;
    }

    id<MTLBuffer> buffer = CVMetalBufferGetBuffer(cvbuf);
    if (buffer == nil) {
        CFRelease(cvbuf);
        PyErr_SetString(PyExc_RuntimeError, "CVMetalBuffer has no MTLBuffer");
        return NULL;
    }

    NSUInteger length = buffer.length;
    NSUInteger storage_mode = buffer.storageMode;
    NSUInteger cpu_cache_mode = buffer.cpuCacheMode;
    void *contents = buffer.contents;

    CFRelease(cvbuf);

    return Py_BuildValue(
        "(KKKK)",
        (unsigned long long)length,
        (unsigned long long)storage_mode,
        (unsigned long long)cpu_cache_mode,
        (unsigned long long)(uintptr_t)contents);
}


static PyObject *sample_luma_gpu(PyObject *self, PyObject *arg) {
    unsigned long long raw = PyLong_AsUnsignedLongLong(arg);
    if (PyErr_Occurred()) {
        return NULL;
    }
    CVPixelBufferRef pb = (CVPixelBufferRef)(uintptr_t)raw;
    if (pb == NULL) {
        PyErr_SetString(PyExc_ValueError, "null CVPixelBufferRef");
        return NULL;
    }
    if (g_cache == NULL || g_queue == nil || g_sample_pipeline == nil) {
        PyErr_SetString(PyExc_RuntimeError, "Metal sampling pipeline unavailable");
        return NULL;
    }

    size_t planes = CVPixelBufferGetPlaneCount(pb);
    if (planes < 2) {
        PyErr_SetString(PyExc_ValueError, "expected bi-planar VideoToolbox frame");
        return NULL;
    }

    size_t width = CVPixelBufferGetWidthOfPlane(pb, 0);
    size_t height = CVPixelBufferGetHeightOfPlane(pb, 0);

    CVMetalTextureRef y_ref = NULL;
    CVReturn r = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        g_cache,
        pb,
        NULL,
        MTLPixelFormatR8Unorm,
        width,
        height,
        0,
        &y_ref);
    if (r != kCVReturnSuccess || y_ref == NULL) {
        if (y_ref) CFRelease(y_ref);
        PyErr_Format(
            PyExc_RuntimeError,
            "CVMetalTextureCacheCreateTextureFromImage failed: %d",
            (int)r);
        return NULL;
    }

    id<MTLTexture> texture = CVMetalTextureGetTexture(y_ref);
    if (texture == nil) {
        CFRelease(y_ref);
        PyErr_SetString(PyExc_RuntimeError, "Metal luma texture missing");
        return NULL;
    }

    const NSUInteger nx = 5;
    const NSUInteger ny = 4;
    const NSUInteger count = nx * ny;
    vector_uint2 coords[count];
    for (NSUInteger j = 0; j < ny; ++j) {
        NSUInteger y = (ny == 1) ? 0 : (j * (height - 1) / (ny - 1));
        for (NSUInteger i = 0; i < nx; ++i) {
            NSUInteger x = (nx == 1) ? 0 : (i * (width - 1) / (nx - 1));
            coords[j * nx + i] = (vector_uint2){(uint32_t)x, (uint32_t)y};
        }
    }

    id<MTLBuffer> coord_buffer =
        [g_device newBufferWithBytes:coords
                              length:sizeof(coords)
                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer =
        [g_device newBufferWithLength:count
                              options:MTLResourceStorageModeShared];
    if (coord_buffer == nil || out_buffer == nil) {
        CFRelease(y_ref);
        PyErr_SetString(PyExc_MemoryError, "failed to allocate Metal sample buffers");
        return NULL;
    }

    id<MTLCommandBuffer> command = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
        CFRelease(y_ref);
        PyErr_SetString(PyExc_RuntimeError, "failed to create Metal command encoder");
        return NULL;
    }

    [encoder setComputePipelineState:g_sample_pipeline];
    [encoder setTexture:texture atIndex:0];
    [encoder setBuffer:out_buffer offset:0 atIndex:0];
    [encoder setBuffer:coord_buffer offset:0 atIndex:1];

    NSUInteger tg = MIN((NSUInteger)32, g_sample_pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];

    if (command.status == MTLCommandBufferStatusError) {
        NSString *msg = command.error.localizedDescription ?: @"unknown Metal error";
        CFRelease(y_ref);
        PyErr_Format(PyExc_RuntimeError, "Metal sample kernel failed: %s", [msg UTF8String]);
        return NULL;
    }

    uint8_t *values = (uint8_t *)out_buffer.contents;
    if (values == NULL) {
        CFRelease(y_ref);
        PyErr_SetString(PyExc_RuntimeError, "Metal sample output has no CPU-visible contents");
        return NULL;
    }

    PyObject *result = PyList_New((Py_ssize_t)count);
    if (result == NULL) {
        CFRelease(y_ref);
        return NULL;
    }
    for (NSUInteger k = 0; k < count; ++k) {
        PyObject *entry = Py_BuildValue(
            "(KKi)",
            (unsigned long long)coords[k].x,
            (unsigned long long)coords[k].y,
            (int)values[k]);
        if (entry == NULL) {
            Py_DECREF(result);
            CFRelease(y_ref);
            return NULL;
        }
        PyList_SET_ITEM(result, (Py_ssize_t)k, entry);
    }

    CFRelease(y_ref);
    return result;
}

static PyObject *buffer_cache_available(
    PyObject *self, PyObject *Py_UNUSED(ignored)) {
    if (g_buffer_cache != NULL) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *device_name(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    if (g_device == nil) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString([[g_device name] UTF8String]);
}

static PyMethodDef methods[] = {
    {"map_once", (PyCFunction)map_once, METH_O,
     "Create transient Metal texture views for a borrowed CVPixelBufferRef."},
    {"map_buffer_once", (PyCFunction)map_buffer_once, METH_O,
     "Create a transient CVMetalBuffer/MTLBuffer live binding."},
    {"make_dlpack_capsule", (PyCFunction)make_dlpack_capsule, METH_O,
     "Export a CVPixelBuffer as a flat kDLMetal uint8 DLPack capsule."},
    {"make_dlpack_plane_capsule", (PyCFunction)make_dlpack_plane_capsule, METH_VARARGS,
     "Export one CVPixelBuffer plane as a strided kDLMetal DLPack capsule."},
    {"plane_layout", (PyCFunction)plane_layout, METH_O,
     "Return IOSurface-backed plane offset/shape/stride metadata."},
    {"sample_luma_gpu", (PyCFunction)sample_luma_gpu, METH_O,
     "Sample deterministic luma pixels through the zero-copy Metal texture view."},
    {"inspect_dlpack_capsule", (PyCFunction)inspect_dlpack_capsule, METH_O,
     "Inspect a legacy DLPack capsule without consuming it."},
    {"buffer_cache_available", (PyCFunction)buffer_cache_available, METH_NOARGS,
     "Return whether CVMetalBufferCache initialization succeeded."},
    {"device_name", (PyCFunction)device_name, METH_NOARGS,
     "Return the active Metal device name."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "puma_metal_bridge",
    "Research-only PyAV VideoToolbox to Metal bridge.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_puma_metal_bridge(void) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device == nil) {
            PyErr_SetString(PyExc_RuntimeError, "no Metal device");
            return NULL;
        }

        g_queue = [g_device newCommandQueue];
        if (g_queue == nil) {
            PyErr_SetString(PyExc_RuntimeError, "cannot create Metal command queue");
            return NULL;
        }

        NSString *source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void puma_sample_luma("
             "texture2d<float, access::read> tex [[texture(0)]],"
             "device uchar *out [[buffer(0)]],"
             "constant uint2 *coords [[buffer(1)]],"
             "uint tid [[thread_position_in_grid]]) {"
             "  float v = tex.read(coords[tid]).r;"
             "  out[tid] = (uchar)clamp(rint(v * 255.0f), 0.0f, 255.0f);"
             "}\n";
        NSError *library_error = nil;
        id<MTLLibrary> library =
            [g_device newLibraryWithSource:source options:nil error:&library_error];
        if (library == nil) {
            const char *msg = library_error.localizedDescription.UTF8String;
            PyErr_Format(PyExc_RuntimeError, "Metal library compile failed: %s", msg ?: "unknown");
            return NULL;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"puma_sample_luma"];
        if (fn == nil) {
            PyErr_SetString(PyExc_RuntimeError, "Metal sample function missing");
            return NULL;
        }
        NSError *pipeline_error = nil;
        g_sample_pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&pipeline_error];
        if (g_sample_pipeline == nil) {
            const char *msg = pipeline_error.localizedDescription.UTF8String;
            PyErr_Format(PyExc_RuntimeError, "Metal sample pipeline failed: %s", msg ?: "unknown");
            return NULL;
        }

        CVReturn r = CVMetalTextureCacheCreate(
            kCFAllocatorDefault, NULL, g_device, NULL, &g_cache);
        if (r != kCVReturnSuccess || g_cache == NULL) {
            PyErr_Format(PyExc_RuntimeError,
                         "CVMetalTextureCacheCreate failed: %d", (int)r);
            return NULL;
        }

        CVReturn br = CVMetalBufferCacheCreate(
            kCFAllocatorDefault, NULL, g_device, &g_buffer_cache);
        if (br != kCVReturnSuccess) {
            g_buffer_cache = NULL;
        }

        return PyModule_Create(&module);
    }
}
