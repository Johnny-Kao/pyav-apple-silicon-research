#!/usr/bin/env python3
import argparse
import json
import statistics
import time

import av
import numpy as np
import puma_metal_bridge
from av.codec.hwaccel import HWAccel


def hwaccel():
    return HWAccel(
        device_type="videotoolbox",
        allow_software_fallback=False,
        is_hw_owned=True,
    )


def exact_capability(path):
    with av.open(path, hwaccel=hwaccel()) as container:
        frame = next(container.decode(video=0))
        if frame.format.name != "videotoolbox_vld":
            raise RuntimeError(f"unexpected format: {frame.format.name}")
        pixel = frame._research_videotoolbox_pixel_buffer_address()
        gpu_sum = int(puma_metal_bridge.sum_luma_gpu(pixel))
        arr = frame.to_ndarray(format="nv12")
        cpu_sum = int(arr[: frame.height, : frame.width].sum(dtype=np.uint64))
        if gpu_sum != cpu_sum:
            raise RuntimeError(f"luma sum mismatch: gpu={gpu_sum} cpu={cpu_sum}")
        return {
            "width": frame.width,
            "height": frame.height,
            "format": frame.format.name,
            "sw_format": frame.sw_format.name if frame.sw_format else None,
            "gpu_sum": gpu_sum,
            "cpu_sum": cpu_sum,
            "equal": True,
        }


def run_gpu(path, max_frames):
    timings = []
    frame_count = 0
    checksum = 0
    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()

    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            pixel = frame._research_videotoolbox_pixel_buffer_address()
            t0 = time.perf_counter_ns()
            value = int(puma_metal_bridge.sum_luma_gpu(pixel))
            timings.append(time.perf_counter_ns() - t0)
            checksum ^= value
            frame_count += 1
            if frame_count >= max_frames:
                break

    wall = time.perf_counter_ns() - wall0
    cpu = time.process_time_ns() - cpu0
    trimmed = timings[5:] if len(timings) > 10 else timings
    return {
        "mode": "metal_full_luma_sum",
        "frames": frame_count,
        "checksum": checksum,
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "operation_p95_us": sorted(trimmed)[int((len(trimmed) - 1) * 0.95)] / 1e3,
        "loop_wall_ms_per_frame": wall / frame_count / 1e6,
        "loop_cpu_ms_per_frame": cpu / frame_count / 1e6,
    }


def run_cpu(path, max_frames):
    timings = []
    frame_count = 0
    checksum = 0
    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()

    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            t0 = time.perf_counter_ns()
            arr = frame.to_ndarray(format="nv12")
            value = int(arr[: frame.height, : frame.width].sum(dtype=np.uint64))
            timings.append(time.perf_counter_ns() - t0)
            checksum ^= value
            frame_count += 1
            if frame_count >= max_frames:
                break

    wall = time.perf_counter_ns() - wall0
    cpu = time.process_time_ns() - cpu0
    trimmed = timings[5:] if len(timings) > 10 else timings
    return {
        "mode": "numpy_download_full_luma_sum",
        "frames": frame_count,
        "checksum": checksum,
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "operation_p95_us": sorted(trimmed)[int((len(trimmed) - 1) * 0.95)] / 1e3,
        "loop_wall_ms_per_frame": wall / frame_count / 1e6,
        "loop_cpu_ms_per_frame": cpu / frame_count / 1e6,
    }


def benchmark(path, max_frames):
    cap = exact_capability(path)

    # Warm the Metal pipeline before timing.
    with av.open(path, hwaccel=hwaccel()) as container:
        frame = next(container.decode(video=0))
        puma_metal_bridge.sum_luma_gpu(
            frame._research_videotoolbox_pixel_buffer_address()
        )

    candidate = run_gpu(path, max_frames)
    baseline = run_cpu(path, max_frames)

    if candidate["checksum"] != baseline["checksum"]:
        raise RuntimeError(
            f"stream checksum mismatch: {candidate['checksum']} != {baseline['checksum']}"
        )

    return {
        "path": path,
        "capability": cap,
        "candidate": candidate,
        "baseline": baseline,
        "summary": {
            "width": cap["width"],
            "height": cap["height"],
            "metal_median_us": candidate["operation_median_us"],
            "numpy_median_us": baseline["operation_median_us"],
            "operation_speedup": baseline["operation_median_us"] / candidate["operation_median_us"],
            "loop_wall_speedup": baseline["loop_wall_ms_per_frame"] / candidate["loop_wall_ms_per_frame"],
            "cpu_time_reduction": baseline["loop_cpu_ms_per_frame"] / candidate["loop_cpu_ms_per_frame"],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--frames-1080", type=int, default=90)
    p.add_argument("--frames-4k", type=int, default=60)
    args = p.parse_args()

    cases = []
    for path in args.paths:
        n = args.frames_4k if "4k" in path.lower() else args.frames_1080
        cases.append(benchmark(path, n))

    print(json.dumps({"cases": cases}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
