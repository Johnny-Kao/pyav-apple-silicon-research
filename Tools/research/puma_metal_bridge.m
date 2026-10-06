#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>

static id<MTLDevice> g_device = nil;
static CVMetalTextureCacheRef g_cache = NULL;

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

static PyObject *device_name(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    if (g_device == nil) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString([[g_device name] UTF8String]);
}

static PyMethodDef methods[] = {
    {"map_once", (PyCFunction)map_once, METH_O,
     "Create transient Metal texture views for a borrowed CVPixelBufferRef."},
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
        return PyModule_Create(&module);
    }
}
