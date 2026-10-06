#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>

static id<MTLDevice> g_device = nil;
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
    int64_t shape[1];
    int64_t strides[1];
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
     "Export a CVPixelBuffer as a kDLMetal uint8 DLPack capsule."},
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
