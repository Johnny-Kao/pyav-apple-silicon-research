#!/usr/bin/env python3
import gc
import json
import statistics
import time

import av
from av.codec.hwaccel import HWAccel
from av.datasets import curated

SAMPLE = "pexels/time-lapse-video-of-night-sky-857195.mp4"
MAX_FRAMES = 180
RGB_FRAMES = 60


def run_decode(path, mode, max_frames):
    owned = mode != "downloaded"
    hw = HWAccel(
        device_type="videotoolbox",
        allow_software_fallback=False,
        is_hw_owned=owned,
    )
    container = av.open(path, hwaccel=hw)
    count = 0
    first = None
    checksum = 0

    wall0 = time.perf_counter()
    cpu0 = time.process_time()

    try:
        for frame in container.decode(video=0):
            if first is None:
                first = {
                    "format": frame.format.name,
                    "sw_format": frame.sw_format.name if frame.sw_format else None,
                    "width": frame.width,
                    "height": frame.height,
                }

            if mode == "retained":
                checksum ^= frame.width ^ frame.height
            elif mode == "downloaded":
                checksum ^= frame.width ^ frame.height
            elif mode == "explicit_transfer":
                sw_format = frame.sw_format.name if frame.sw_format else None
                out = frame.reformat(format=sw_format)
                checksum ^= out.width ^ out.height ^ len(out.planes)
            elif mode == "rgb_numpy":
                arr = frame.to_ndarray(format="rgb24")
                checksum ^= int(arr[0, 0, 0])
                del arr
            else:
                raise ValueError(mode)

            count += 1
            if count >= max_frames:
                break
    finally:
        container.close()

    cpu = time.process_time() - cpu0
    wall = time.perf_counter() - wall0
    gc.collect()

    if not first:
        raise RuntimeError(f"{mode}: no frames decoded")

    return {
        "mode": mode,
        "frames": count,
        "wall_s": wall,
        "cpu_s": cpu,
        "fps": count / wall,
        "cpu_ms_per_frame": cpu * 1000.0 / count,
        "wall_ms_per_frame": wall * 1000.0 / count,
        "first_frame": first,
        "checksum": checksum,
    }


def median(rows, key):
    return statistics.median(r[key] for r in rows)


def main():
    path = curated(SAMPLE)

    # Warm-up: initialize decoder/device and touch the sample once.
    warm = run_decode(path, "retained", 12)

    rows = []
    # Alternate order to reduce runner drift bias.
    for order in [
        ("downloaded", "retained"),
        ("retained", "downloaded"),
        ("downloaded", "retained"),
    ]:
        for mode in order:
            rows.append(run_decode(path, mode, MAX_FRAMES))

    # Explicitly exercise the same software-transfer boundary from an
    # is_hw_owned VideoToolbox frame.
    transfer_rows = [
        run_decode(path, "explicit_transfer", MAX_FRAMES),
        run_decode(path, "explicit_transfer", MAX_FRAMES),
    ]

    # Common Python consumer path: materialize RGB NumPy arrays.
    rgb_rows = [
        run_decode(path, "rgb_numpy", RGB_FRAMES),
        run_decode(path, "rgb_numpy", RGB_FRAMES),
    ]

    downloaded = [r for r in rows if r["mode"] == "downloaded"]
    retained = [r for r in rows if r["mode"] == "retained"]

    d_wall = median(downloaded, "wall_ms_per_frame")
    r_wall = median(retained, "wall_ms_per_frame")
    d_cpu = median(downloaded, "cpu_ms_per_frame")
    r_cpu = median(retained, "cpu_ms_per_frame")
    t_wall = median(transfer_rows, "wall_ms_per_frame")
    rgb_wall = median(rgb_rows, "wall_ms_per_frame")

    first = retained[0]["first_frame"]
    # Approximate NV12 bytes/frame for scale context only.
    approx_nv12_bytes = first["width"] * first["height"] * 3 // 2

    result = {
        "sample": SAMPLE,
        "pyav_version": av.__version__,
        "warmup": warm,
        "runs": rows,
        "explicit_transfer_runs": transfer_rows,
        "rgb_numpy_runs": rgb_rows,
        "summary": {
            "retained_wall_ms_per_frame": r_wall,
            "downloaded_wall_ms_per_frame": d_wall,
            "download_vs_retained_wall_ratio": d_wall / r_wall,
            "retained_cpu_ms_per_frame": r_cpu,
            "downloaded_cpu_ms_per_frame": d_cpu,
            "download_vs_retained_cpu_ratio": d_cpu / r_cpu if r_cpu else None,
            "explicit_transfer_wall_ms_per_frame": t_wall,
            "explicit_transfer_added_ms_vs_retained": t_wall - r_wall,
            "rgb_numpy_wall_ms_per_frame": rgb_wall,
            "rgb_numpy_added_ms_vs_retained": rgb_wall - r_wall,
            "approx_nv12_bytes_per_frame": approx_nv12_bytes,
            "first_hw_frame": first,
        },
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
