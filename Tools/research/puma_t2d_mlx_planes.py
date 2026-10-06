#!/usr/bin/env python3
import argparse
import gc
import json
import statistics
import time

import av
import mlx.core as mx
import numpy as np
import puma_metal_bridge
from av.codec.hwaccel import HWAccel


class PlaneDLPack:
    def __init__(self, pixel_buffer_address, plane):
        self._capsule = puma_metal_bridge.make_dlpack_plane_capsule(
            pixel_buffer_address, plane
        )
        (
            self.device_type,
            self.device_id,
            self.data_handle,
            self.shape0,
        ) = puma_metal_bridge.inspect_dlpack_capsule(self._capsule)

    def __dlpack_device__(self):
        return (self.device_type, self.device_id)

    def __dlpack__(
        self,
        stream=None,
        max_version=None,
        dl_device=None,
        copy=None,
    ):
        if copy is True:
            raise BufferError("research provider only exposes zero-copy Metal")
        if self._capsule is None:
            raise RuntimeError("DLPack capsule already consumed")
        cap = self._capsule
        self._capsule = None
        return cap


def hwaccel():
    return HWAccel(
        device_type="videotoolbox",
        allow_software_fallback=False,
        is_hw_owned=True,
    )


def gpu_sum_u8(arr):
    value = mx.sum(arr.astype(mx.uint32))
    mx.eval(value)
    return int(value.item())


def inspect_export(arr):
    cap = arr.__dlpack__()
    return puma_metal_bridge.inspect_dlpack_capsule(cap)


def capability(path):
    with av.open(path, hwaccel=hwaccel()) as container:
        frame = next(container.decode(video=0))
        if frame.format.name != "videotoolbox_vld":
            raise RuntimeError(frame.format.name)

        h, w = frame.height, frame.width
        pixel = frame._research_videotoolbox_pixel_buffer_address()
        layout = puma_metal_bridge.plane_layout(pixel)
        if len(layout) != 2:
            raise RuntimeError(f"expected NV12 two-plane frame, got {layout}")

        y_provider = PlaneDLPack(pixel, 0)
        uv_provider = PlaneDLPack(pixel, 1)
        y_handle = y_provider.data_handle
        uv_handle = uv_provider.data_handle

        y = mx.from_dlpack(y_provider, copy=False)
        uv = mx.from_dlpack(uv_provider, copy=False)

        y_sum = gpu_sum_u8(y)
        uv_sum = gpu_sum_u8(uv)

        y_export = inspect_export(y)
        uv_export = inspect_export(uv)

        cpu = frame.to_ndarray(format="nv12")
        if cpu.ndim != 2 or cpu.shape[0] < h + h // 2 or cpu.shape[1] < w:
            raise RuntimeError(f"unexpected NV12 ndarray shape {cpu.shape}")
        cpu_y = cpu[:h, :w]
        cpu_uv = cpu[h : h + h // 2, :w]
        cpu_y_sum = int(cpu_y.sum(dtype=np.uint64))
        cpu_uv_sum = int(cpu_uv.sum(dtype=np.uint64))

        return {
            "frame": {
                "width": w,
                "height": h,
                "format": frame.format.name,
                "sw_format": frame.sw_format.name if frame.sw_format else None,
            },
            "plane_layout": [
                {
                    "offset": int(x[0]),
                    "width": int(x[1]),
                    "height": int(x[2]),
                    "bytes_per_row": int(x[3]),
                    "bytes_per_element": int(x[4]),
                }
                for x in layout
            ],
            "mlx_y_shape": list(y.shape),
            "mlx_uv_shape": list(uv.shape),
            "mlx_y_sum": y_sum,
            "mlx_uv_sum": uv_sum,
            "numpy_y_sum": cpu_y_sum,
            "numpy_uv_sum": cpu_uv_sum,
            "y_sum_equal": y_sum == cpu_y_sum,
            "uv_sum_equal": uv_sum == cpu_uv_sum,
            "y_same_metal_handle": int(y_export[2]) == int(y_handle),
            "uv_same_metal_handle": int(uv_export[2]) == int(uv_handle),
            "copy_false_succeeded": True,
        }


def run_mlx_luma(path, max_frames):
    timings = []
    checksums = 0
    count = 0
    cpu0 = time.process_time_ns()
    wall0 = time.perf_counter_ns()
    first = None

    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            if first is None:
                first = (frame.width, frame.height)
            pixel = frame._research_videotoolbox_pixel_buffer_address()

            t0 = time.perf_counter_ns()
            provider = PlaneDLPack(pixel, 0)
            y = mx.from_dlpack(provider, copy=False)
            value = mx.sum(y.astype(mx.uint32))
            mx.eval(value)
            checksums ^= int(value.item())
            timings.append(time.perf_counter_ns() - t0)

            count += 1
            if count >= max_frames:
                break

    cpu_ns = time.process_time_ns() - cpu0
    wall_ns = time.perf_counter_ns() - wall0
    gc.collect()
    trimmed = timings[5:] if len(timings) > 10 else timings
    return {
        "mode": "mlx_zero_copy_luma_sum",
        "frames": count,
        "width": first[0],
        "height": first[1],
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "operation_p95_us": sorted(trimmed)[int((len(trimmed)-1)*0.95)] / 1e3,
        "loop_wall_ms_per_frame": wall_ns / count / 1e6,
        "loop_cpu_ms_per_frame": cpu_ns / count / 1e6,
        "checksum": checksums,
    }


def run_numpy_luma(path, max_frames):
    timings = []
    checksums = 0
    count = 0
    cpu0 = time.process_time_ns()
    wall0 = time.perf_counter_ns()
    first = None

    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            if first is None:
                first = (frame.width, frame.height)
            h, w = frame.height, frame.width

            t0 = time.perf_counter_ns()
            arr = frame.to_ndarray(format="nv12")
            value = int(arr[:h, :w].sum(dtype=np.uint64))
            checksums ^= value
            timings.append(time.perf_counter_ns() - t0)

            count += 1
            if count >= max_frames:
                break

    cpu_ns = time.process_time_ns() - cpu0
    wall_ns = time.perf_counter_ns() - wall0
    gc.collect()
    trimmed = timings[5:] if len(timings) > 10 else timings
    return {
        "mode": "numpy_download_luma_sum",
        "frames": count,
        "width": first[0],
        "height": first[1],
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "operation_p95_us": sorted(trimmed)[int((len(trimmed)-1)*0.95)] / 1e3,
        "loop_wall_ms_per_frame": wall_ns / count / 1e6,
        "loop_cpu_ms_per_frame": cpu_ns / count / 1e6,
        "checksum": checksums,
    }


def benchmark(path, frames):
    cap = capability(path)

    # Warm Metal/MLX.
    with av.open(path, hwaccel=hwaccel()) as container:
        f = next(container.decode(video=0))
        p = PlaneDLPack(f._research_videotoolbox_pixel_buffer_address(), 0)
        y = mx.from_dlpack(p, copy=False)
        gpu_sum_u8(y)

    candidate = run_mlx_luma(path, frames)
    baseline = run_numpy_luma(path, frames)

    if candidate["checksum"] != baseline["checksum"]:
        raise RuntimeError(
            f"luma checksum mismatch: {candidate['checksum']} != "
            f"{baseline['checksum']}"
        )

    return {
        "path": path,
        "capability": cap,
        "candidate": candidate,
        "baseline": baseline,
        "summary": {
            "width": candidate["width"],
            "height": candidate["height"],
            "mlx_luma_median_us": candidate["operation_median_us"],
            "numpy_luma_median_us": baseline["operation_median_us"],
            "wall_speedup": (
                baseline["operation_median_us"]
                / candidate["operation_median_us"]
            ),
            "candidate_cpu_ms_per_frame": candidate["loop_cpu_ms_per_frame"],
            "baseline_cpu_ms_per_frame": baseline["loop_cpu_ms_per_frame"],
            "cpu_time_reduction": (
                baseline["loop_cpu_ms_per_frame"]
                / candidate["loop_cpu_ms_per_frame"]
            ),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--frames-1080", type=int, default=60)
    p.add_argument("--frames-4k", type=int, default=40)
    args = p.parse_args()

    cases = []
    for path in args.paths:
        frames = args.frames_4k if "4k" in path.lower() else args.frames_1080
        cases.append(benchmark(path, frames))

    print(json.dumps({"cases": cases}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
