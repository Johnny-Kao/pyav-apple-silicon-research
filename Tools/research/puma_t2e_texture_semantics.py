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


def cpu_samples(arr, height, width):
    y = arr[:height, :width]
    xs = [0, (width - 1) // 4, (width - 1) // 2, 3 * (width - 1) // 4, width - 1]
    ys = [0, (height - 1) // 3, 2 * (height - 1) // 3, height - 1]
    return [(x, yy, int(y[yy, x])) for yy in ys for x in xs]


def capability(path):
    with av.open(path, hwaccel=hwaccel()) as container:
        frame = next(container.decode(video=0))
        if frame.format.name != "videotoolbox_vld":
            raise RuntimeError(frame.format.name)

        pixel = frame._research_videotoolbox_pixel_buffer_address()
        gpu = [tuple(x) for x in puma_metal_bridge.sample_luma_gpu(pixel)]
        arr = frame.to_ndarray(format="nv12")
        cpu = cpu_samples(arr, frame.height, frame.width)

        if gpu != cpu:
            mismatches = [
                {"gpu": g, "cpu": c}
                for g, c in zip(gpu, cpu)
                if g != c
            ]
            raise RuntimeError(
                f"Metal texture semantic mismatch: {mismatches[:8]}"
            )

        mapped = puma_metal_bridge.map_once(pixel)
        return {
            "width": frame.width,
            "height": frame.height,
            "format": frame.format.name,
            "sw_format": frame.sw_format.name if frame.sw_format else None,
            "sample_count": len(gpu),
            "samples_equal": True,
            "samples": gpu,
            "metal_view": list(mapped),
        }


def run_gpu_samples(path, max_frames):
    timings = []
    checksum = 0
    frames = 0
    cpu0 = time.process_time_ns()
    wall0 = time.perf_counter_ns()

    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            pixel = frame._research_videotoolbox_pixel_buffer_address()
            t0 = time.perf_counter_ns()
            samples = puma_metal_bridge.sample_luma_gpu(pixel)
            timings.append(time.perf_counter_ns() - t0)
            checksum ^= sum(int(v) for _, _, v in samples)
            frames += 1
            if frames >= max_frames:
                break

    return {
        "mode": "metal_texture_20_samples",
        "frames": frames,
        "operation_median_us": statistics.median(timings[5:] if len(timings) > 10 else timings) / 1e3,
        "operation_p95_us": sorted(timings)[int((len(timings)-1)*0.95)] / 1e3,
        "loop_wall_ms_per_frame": (time.perf_counter_ns() - wall0) / frames / 1e6,
        "loop_cpu_ms_per_frame": (time.process_time_ns() - cpu0) / frames / 1e6,
        "checksum": checksum,
    }


def run_cpu_samples(path, max_frames):
    timings = []
    checksum = 0
    frames = 0
    cpu0 = time.process_time_ns()
    wall0 = time.perf_counter_ns()

    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            t0 = time.perf_counter_ns()
            arr = frame.to_ndarray(format="nv12")
            samples = cpu_samples(arr, frame.height, frame.width)
            timings.append(time.perf_counter_ns() - t0)
            checksum ^= sum(v for _, _, v in samples)
            frames += 1
            if frames >= max_frames:
                break

    return {
        "mode": "numpy_download_20_samples",
        "frames": frames,
        "operation_median_us": statistics.median(timings[5:] if len(timings) > 10 else timings) / 1e3,
        "operation_p95_us": sorted(timings)[int((len(timings)-1)*0.95)] / 1e3,
        "loop_wall_ms_per_frame": (time.perf_counter_ns() - wall0) / frames / 1e6,
        "loop_cpu_ms_per_frame": (time.process_time_ns() - cpu0) / frames / 1e6,
        "checksum": checksum,
    }


def run_map_only(path, max_frames):
    timings = []
    frames = 0
    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            pixel = frame._research_videotoolbox_pixel_buffer_address()
            t0 = time.perf_counter_ns()
            puma_metal_bridge.map_once(pixel)
            timings.append(time.perf_counter_ns() - t0)
            frames += 1
            if frames >= max_frames:
                break
    trimmed = timings[5:] if len(timings) > 10 else timings
    return {
        "mode": "metal_texture_map",
        "frames": frames,
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "operation_p95_us": sorted(trimmed)[int((len(trimmed)-1)*0.95)] / 1e3,
    }


def benchmark(path, max_frames):
    cap = capability(path)

    # Warm Metal pipeline before timing.
    with av.open(path, hwaccel=hwaccel()) as container:
        frame = next(container.decode(video=0))
        puma_metal_bridge.sample_luma_gpu(
            frame._research_videotoolbox_pixel_buffer_address()
        )

    candidate = run_gpu_samples(path, max_frames)
    baseline = run_cpu_samples(path, max_frames)
    mapped = run_map_only(path, max_frames)

    if candidate["checksum"] != baseline["checksum"]:
        raise RuntimeError(
            f"equivalent 20-sample checksum mismatch: "
            f"{candidate['checksum']} != {baseline['checksum']}"
        )

    return {
        "path": path,
        "capability": cap,
        "candidate": candidate,
        "baseline": baseline,
        "map_only": mapped,
        "summary": {
            "width": cap["width"],
            "height": cap["height"],
            "gpu_samples_median_us": candidate["operation_median_us"],
            "numpy_samples_median_us": baseline["operation_median_us"],
            "equivalent_speedup": baseline["operation_median_us"] / candidate["operation_median_us"],
            "cpu_time_reduction": baseline["loop_cpu_ms_per_frame"] / candidate["loop_cpu_ms_per_frame"],
            "map_only_median_us": mapped["operation_median_us"],
            "download_over_map": baseline["operation_median_us"] / mapped["operation_median_us"],
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
        n = args.frames_4k if "4k" in path.lower() else args.frames_1080
        cases.append(benchmark(path, n))

    print(json.dumps({"cases": cases}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
