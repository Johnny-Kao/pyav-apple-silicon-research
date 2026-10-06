#!/usr/bin/env python3
import argparse
import gc
import json
import statistics
import time

import av
import mlx.core as mx
import puma_metal_bridge
from av.codec.hwaccel import HWAccel


class MetalDLPack:
    """Single-use DLPack provider backed by a retained CVMetalBuffer."""

    def __init__(self, pixel_buffer_address):
        self._capsule = puma_metal_bridge.make_dlpack_capsule(
            pixel_buffer_address
        )
        (
            self.device_type,
            self.device_id,
            self.data_handle,
            self.length,
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
        capsule = self._capsule
        self._capsule = None
        return capsule


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


def inspect_mlx_export(arr):
    try:
        capsule = arr.__dlpack__()
        return puma_metal_bridge.inspect_dlpack_capsule(capsule)
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
        }


def force_read(arr):
    n = min(arr.size, 4096)
    # Force a real Metal kernel to read the imported backing buffer.
    value = mx.sum(arr[:n].astype(mx.uint32))
    mx.eval(value)
    return int(value.item())


def capability(path):
    with av.open(path, hwaccel=hwaccel()) as container:
        frame = next(container.decode(video=0))
        if frame.format.name != "videotoolbox_vld":
            raise RuntimeError(
                f"expected videotoolbox_vld, got {frame.format.name}"
            )
        pixel = frame._research_videotoolbox_pixel_buffer_address()
        provider = MetalDLPack(pixel)
        original = {
            "device_type": provider.device_type,
            "device_id": provider.device_id,
            "data_handle": provider.data_handle,
            "length": provider.length,
        }

        t0 = time.perf_counter_ns()
        arr = mx.from_dlpack(provider, copy=False)
        import_ns = time.perf_counter_ns() - t0

        checksum = force_read(arr)
        exported = inspect_mlx_export(arr)

        if isinstance(exported, tuple):
            export_info = {
                "device_type": int(exported[0]),
                "device_id": int(exported[1]),
                "data_handle": int(exported[2]),
                "length": int(exported[3]),
            }
            same_handle = export_info["data_handle"] == original["data_handle"]
        else:
            export_info = exported
            same_handle = None

        return {
            "frame": {
                "width": frame.width,
                "height": frame.height,
                "format": frame.format.name,
                "sw_format": frame.sw_format.name if frame.sw_format else None,
            },
            "original_dlpack": original,
            "mlx_shape": list(arr.shape),
            "mlx_dtype": str(arr.dtype),
            "import_us": import_ns / 1e3,
            "forced_read_checksum": checksum,
            "mlx_reexport": export_info,
            "same_metal_buffer_handle": same_handle,
            "copy_false_succeeded": True,
        }


def run_mode(path, mode, max_frames):
    timings = []
    frames = 0
    checksum = 0
    first = None
    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()

    with av.open(path, hwaccel=hwaccel()) as container:
        for frame in container.decode(video=0):
            if frame.format.name != "videotoolbox_vld":
                raise RuntimeError(
                    f"expected videotoolbox_vld, got {frame.format.name}"
                )
            if first is None:
                first = {
                    "width": frame.width,
                    "height": frame.height,
                    "format": frame.format.name,
                    "sw_format": frame.sw_format.name if frame.sw_format else None,
                }

            t0 = time.perf_counter_ns()
            if mode == "mlx_dlpack":
                pixel = frame._research_videotoolbox_pixel_buffer_address()
                provider = MetalDLPack(pixel)
                arr = mx.from_dlpack(provider, copy=False)
                checksum ^= int(arr.size)
                del arr
            elif mode == "metal_buffer":
                pixel = frame._research_videotoolbox_pixel_buffer_address()
                info = puma_metal_bridge.map_buffer_once(pixel)
                checksum ^= int(info[0])
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
            timings.append(time.perf_counter_ns() - t0)
            frames += 1
            if frames >= max_frames:
                break

    wall = time.perf_counter_ns() - wall0
    cpu = time.process_time_ns() - cpu0
    gc.collect()

    trimmed = timings[10:] if len(timings) > 20 else timings
    return {
        "mode": mode,
        "frames": frames,
        "first_frame": first,
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "operation_p95_us": pctl(trimmed, 0.95) / 1e3,
        "loop_wall_ms_per_frame": wall / frames / 1e6,
        "loop_cpu_ms_per_frame": cpu / frames / 1e6,
        "checksum": checksum,
    }


def benchmark(path, frames, rgb_frames):
    cap = capability(path)

    # Warm MLX/Metal runtime outside timed rounds.
    with av.open(path, hwaccel=hwaccel()) as container:
        frame = next(container.decode(video=0))
        provider = MetalDLPack(
            frame._research_videotoolbox_pixel_buffer_address()
        )
        warm = mx.from_dlpack(provider, copy=False)
        force_read(warm)
        del warm

    dl = run_mode(path, "mlx_dlpack", frames)
    mb = run_mode(path, "metal_buffer", frames)
    nv = run_mode(path, "transfer_nv12", frames)
    rgb = run_mode(path, "rgb_numpy", min(frames, rgb_frames))

    dm = dl["operation_median_us"]
    bm = mb["operation_median_us"]
    nm = nv["operation_median_us"]
    rm = rgb["operation_median_us"]

    return {
        "path": path,
        "capability": cap,
        "mlx_dlpack": dl,
        "metal_buffer": mb,
        "transfer_nv12": nv,
        "rgb_numpy": rgb,
        "summary": {
            "width": dl["first_frame"]["width"],
            "height": dl["first_frame"]["height"],
            "mlx_dlpack_median_us": dm,
            "metal_buffer_median_us": bm,
            "transfer_nv12_median_us": nm,
            "rgb_numpy_median_us": rm,
            "transfer_over_dlpack": nm / dm,
            "rgb_over_dlpack": rm / dm,
            "dlpack_over_raw_buffer": dm / bm,
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
        "mlx_version": getattr(mx, "__version__", None),
        "cases": [
            benchmark(path, args.frames, args.rgb_frames)
            for path in args.paths
        ],
    }
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
