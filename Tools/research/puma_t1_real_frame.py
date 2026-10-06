#!/usr/bin/env python3
import argparse
import gc
import json
import statistics
import time

import av
import puma_metal_bridge
from av.codec.hwaccel import HWAccel


def percentile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("empty values")
    idx = int((len(values) - 1) * q)
    return values[idx]


def run_hw_mode(path, mode, max_frames):
    hw = HWAccel(
        device_type="videotoolbox",
        allow_software_fallback=False,
        is_hw_owned=True,
    )
    container = av.open(path, hwaccel=hw)
    op_ns = []
    count = 0
    checksum = 0
    first = None

    loop0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()
    try:
        for frame in container.decode(video=0):
            if frame.format.name != "videotoolbox_vld":
                raise RuntimeError(
                    f"expected videotoolbox_vld, got {frame.format.name}"
                )
            if first is None:
                first = {
                    "format": frame.format.name,
                    "sw_format": frame.sw_format.name if frame.sw_format else None,
                    "width": frame.width,
                    "height": frame.height,
                }

            t0 = time.perf_counter_ns()
            if mode == "metal_map":
                ptr = frame._research_videotoolbox_pixel_buffer_address()
                dims = puma_metal_bridge.map_once(ptr)
                checksum ^= sum(dims)
            elif mode == "transfer_nv12":
                sw_name = frame.sw_format.name
                out = frame.reformat(format=sw_name)
                checksum ^= out.width ^ out.height ^ len(out.planes)
            elif mode == "rgb_numpy":
                arr = frame.to_ndarray(format="rgb24")
                checksum ^= int(arr[0, 0, 0])
                del arr
            else:
                raise ValueError(mode)
            t1 = time.perf_counter_ns()
            op_ns.append(t1 - t0)

            count += 1
            if count >= max_frames:
                break
    finally:
        container.close()

    cpu_ns = time.process_time_ns() - cpu0
    loop_ns = time.perf_counter_ns() - loop0
    gc.collect()

    trimmed = op_ns[10:] if len(op_ns) > 20 else op_ns
    return {
        "mode": mode,
        "frames": count,
        "first_frame": first,
        "operation_median_us": statistics.median(trimmed) / 1000.0,
        "operation_p95_us": percentile(trimmed, 0.95) / 1000.0,
        "loop_wall_ms_per_frame": loop_ns / count / 1e6,
        "loop_cpu_ms_per_frame": cpu_ns / count / 1e6,
        "checksum": checksum,
    }


def run_auto_download(path, max_frames):
    hw = HWAccel(
        device_type="videotoolbox",
        allow_software_fallback=False,
        is_hw_owned=False,
    )
    container = av.open(path, hwaccel=hw)
    count = 0
    checksum = 0
    first = None
    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()
    try:
        for frame in container.decode(video=0):
            if first is None:
                first = {
                    "format": frame.format.name,
                    "sw_format": frame.sw_format.name if frame.sw_format else None,
                    "width": frame.width,
                    "height": frame.height,
                }
            checksum ^= frame.width ^ frame.height ^ len(frame.planes)
            count += 1
            if count >= max_frames:
                break
    finally:
        container.close()
    cpu_ns = time.process_time_ns() - cpu0
    wall_ns = time.perf_counter_ns() - wall0
    gc.collect()
    return {
        "mode": "auto_download",
        "frames": count,
        "first_frame": first,
        "loop_wall_ms_per_frame": wall_ns / count / 1e6,
        "loop_cpu_ms_per_frame": cpu_ns / count / 1e6,
        "checksum": checksum,
    }


def benchmark(path, max_frames, rgb_frames):
    # Initialize Metal and VideoToolbox before timed rounds.
    metal_name = puma_metal_bridge.device_name()
    warm = run_hw_mode(path, "metal_map", min(12, max_frames))

    metal = run_hw_mode(path, "metal_map", max_frames)
    transfer = run_hw_mode(path, "transfer_nv12", max_frames)
    rgb = run_hw_mode(path, "rgb_numpy", min(rgb_frames, max_frames))
    auto = run_auto_download(path, max_frames)

    ratio_transfer = (
        transfer["operation_median_us"] / metal["operation_median_us"]
    )
    ratio_rgb = rgb["operation_median_us"] / metal["operation_median_us"]

    return {
        "path": path,
        "metal_device": metal_name,
        "warmup": warm,
        "metal_map": metal,
        "transfer_nv12": transfer,
        "rgb_numpy": rgb,
        "auto_download": auto,
        "summary": {
            "width": metal["first_frame"]["width"],
            "height": metal["first_frame"]["height"],
            "metal_map_median_us": metal["operation_median_us"],
            "transfer_nv12_median_us": transfer["operation_median_us"],
            "rgb_numpy_median_us": rgb["operation_median_us"],
            "transfer_over_metal": ratio_transfer,
            "rgb_over_metal": ratio_rgb,
            "retained_loop_ms_per_frame": metal["loop_wall_ms_per_frame"],
            "auto_download_loop_ms_per_frame": auto["loop_wall_ms_per_frame"],
            "auto_download_over_retained_loop": (
                auto["loop_wall_ms_per_frame"]
                / metal["loop_wall_ms_per_frame"]
            ),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--rgb-frames", type=int, default=60)
    args = p.parse_args()

    result = {
        "pyav_version": av.__version__,
        "cases": [
            benchmark(path, args.frames, args.rgb_frames)
            for path in args.paths
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
