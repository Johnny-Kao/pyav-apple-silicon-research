#!/usr/bin/env python3
import argparse
import gc
import json
import statistics
import time

import av
import puma_metal_bridge
from av.codec.hwaccel import HWAccel


def pctl(values, q):
    xs = sorted(values)
    if not xs:
        raise ValueError("empty timings")
    return xs[int((len(xs) - 1) * q)]


def hwaccel():
    return HWAccel(
        device_type="videotoolbox",
        allow_software_fallback=False,
        is_hw_owned=True,
    )


def run_mode(path, mode, max_frames):
    timings = []
    first = None
    checksum = 0
    frames = 0
    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()

    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            if frame.format.name != "videotoolbox_vld":
                raise RuntimeError(
                    f"expected VideoToolbox hardware frame, got {frame.format.name}"
                )
            if first is None:
                first = {
                    "width": frame.width,
                    "height": frame.height,
                    "format": frame.format.name,
                    "sw_format": frame.sw_format.name if frame.sw_format else None,
                }

            t0 = time.perf_counter_ns()
            if mode == "metal_buffer":
                ptr = frame._research_videotoolbox_pixel_buffer_address()
                length, storage_mode, cpu_cache_mode, contents = (
                    puma_metal_bridge.map_buffer_once(ptr)
                )
                checksum ^= int(length) ^ int(storage_mode) ^ int(cpu_cache_mode)
                metadata = {
                    "length": int(length),
                    "storage_mode": int(storage_mode),
                    "cpu_cache_mode": int(cpu_cache_mode),
                    "cpu_contents_nonnull": bool(contents),
                    "shared_storage": int(storage_mode) == 0,
                }
            elif mode == "metal_texture":
                ptr = frame._research_videotoolbox_pixel_buffer_address()
                dims = puma_metal_bridge.map_once(ptr)
                checksum ^= sum(dims)
                metadata = None
            elif mode == "transfer_nv12":
                sw_name = frame.sw_format.name
                out = frame.reformat(format=sw_name)
                checksum ^= out.width ^ out.height ^ len(out.planes)
                metadata = None
            elif mode == "rgb_numpy":
                arr = frame.to_ndarray(format="rgb24")
                checksum ^= int(arr[0, 0, 0])
                metadata = None
                del arr
            else:
                raise ValueError(mode)
            timings.append(time.perf_counter_ns() - t0)
            frames += 1
            if frames >= max_frames:
                break

    cpu = time.process_time_ns() - cpu0
    wall = time.perf_counter_ns() - wall0
    gc.collect()

    trimmed = timings[10:] if len(timings) > 20 else timings
    result = {
        "mode": mode,
        "frames": frames,
        "first_frame": first,
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "operation_p95_us": pctl(trimmed, 0.95) / 1e3,
        "loop_wall_ms_per_frame": wall / frames / 1e6,
        "loop_cpu_ms_per_frame": cpu / frames / 1e6,
        "checksum": checksum,
    }
    if mode == "metal_buffer":
        result["buffer_metadata"] = metadata
    return result


def benchmark(path, frames, rgb_frames):
    capability = {
        "buffer_cache_available": bool(
            puma_metal_bridge.buffer_cache_available()
        ),
        "metal_device": puma_metal_bridge.device_name(),
    }

    if not capability["buffer_cache_available"]:
        return {
            "path": path,
            "capability": capability,
            "supported": False,
            "reason": "CVMetalBufferCache initialization unavailable",
        }

    try:
        metal_buffer = run_mode(path, "metal_buffer", frames)
    except Exception as exc:
        return {
            "path": path,
            "capability": capability,
            "supported": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    metal_texture = run_mode(path, "metal_texture", frames)
    transfer = run_mode(path, "transfer_nv12", frames)
    rgb = run_mode(path, "rgb_numpy", min(frames, rgb_frames))

    bm = metal_buffer["operation_median_us"]
    tm = metal_texture["operation_median_us"]
    nv = transfer["operation_median_us"]
    rm = rgb["operation_median_us"]

    return {
        "path": path,
        "capability": capability,
        "supported": True,
        "metal_buffer": metal_buffer,
        "metal_texture": metal_texture,
        "transfer_nv12": transfer,
        "rgb_numpy": rgb,
        "summary": {
            "width": metal_buffer["first_frame"]["width"],
            "height": metal_buffer["first_frame"]["height"],
            "metal_buffer_median_us": bm,
            "metal_texture_median_us": tm,
            "transfer_nv12_median_us": nv,
            "rgb_numpy_median_us": rm,
            "transfer_over_buffer": nv / bm,
            "rgb_over_buffer": rm / bm,
            "texture_over_buffer": tm / bm,
            "buffer_metadata": metal_buffer["buffer_metadata"],
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--rgb-frames", type=int, default=60)
    args = p.parse_args()

    data = {
        "pyav_version": av.__version__,
        "cases": [
            benchmark(path, args.frames, args.rgb_frames)
            for path in args.paths
        ],
    }
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
