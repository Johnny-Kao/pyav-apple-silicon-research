#!/usr/bin/env python3
import argparse
import gc
import json
import statistics
import time

import av
from av.codec.hwaccel import HWAccel, HWDevice


def pctl(values, q):
    xs = sorted(values)
    if not xs:
        raise ValueError("empty timings")
    return xs[int((len(xs) - 1) * q)]


def make_scale_graph(frame, out_w, out_h):
    device = HWDevice("videotoolbox")
    graph = av.filter.Graph(hw_device=device)
    src = graph.add_buffer(template=frame)
    scale = graph.add("scale_vt", f"w={out_w}:h={out_h}")
    sink = graph.add("buffersink")
    graph.link_nodes(src, scale, sink).configure()
    return graph, device


def pull_one(graph):
    while True:
        try:
            return graph.vpull()
        except BlockingIOError:
            continue


def bench(path, out_w, out_h, max_frames):
    hw = HWAccel(
        device_type="videotoolbox",
        allow_software_fallback=False,
        is_hw_owned=True,
    )

    with av.open(path, hwaccel=hw) as c:
        first = next(c.decode(video=0))
        if first.format.name != "videotoolbox_vld":
            raise RuntimeError(f"expected videotoolbox_vld, got {first.format.name}")
        graph, device = make_scale_graph(first, out_w, out_h)
        graph.vpush(first)
        warm_out = pull_one(graph)
        if warm_out.format.name != "videotoolbox_vld":
            raise RuntimeError(
                f"scale_vt fell off hardware: {warm_out.format.name}"
            )
        if (warm_out.width, warm_out.height) != (out_w, out_h):
            raise RuntimeError(
                f"wrong scale_vt size {(warm_out.width, warm_out.height)}"
            )

    def run_candidate():
        timings = []
        frames = 0
        cpu0 = time.process_time_ns()
        wall0 = time.perf_counter_ns()
        with av.open(path, hwaccel=hw) as c:
            graph = None
            device = None
            for frame in c.decode(video=0):
                if frame.format.name != "videotoolbox_vld":
                    raise RuntimeError(
                        f"expected hardware frame, got {frame.format.name}"
                    )
                if graph is None:
                    graph, device = make_scale_graph(frame, out_w, out_h)
                t0 = time.perf_counter_ns()
                graph.vpush(frame)
                out = pull_one(graph)
                t1 = time.perf_counter_ns()
                if out.format.name != "videotoolbox_vld":
                    raise RuntimeError(
                        f"candidate output fell off hardware: {out.format.name}"
                    )
                if (out.width, out.height) != (out_w, out_h):
                    raise RuntimeError("candidate wrong dimensions")
                timings.append(t1 - t0)
                frames += 1
                if frames >= max_frames:
                    break
        wall = time.perf_counter_ns() - wall0
        cpu = time.process_time_ns() - cpu0
        trimmed = timings[10:] if len(timings) > 20 else timings
        return {
            "mode": "scale_vt",
            "frames": frames,
            "operation_median_us": statistics.median(trimmed) / 1e3,
            "operation_p95_us": pctl(trimmed, 0.95) / 1e3,
            "loop_wall_ms_per_frame": wall / frames / 1e6,
            "loop_cpu_ms_per_frame": cpu / frames / 1e6,
            "output_format": "videotoolbox_vld",
        }

    def run_baseline():
        timings = []
        frames = 0
        cpu0 = time.process_time_ns()
        wall0 = time.perf_counter_ns()
        with av.open(path, hwaccel=hw) as c:
            for frame in c.decode(video=0):
                sw = frame.sw_format.name if frame.sw_format else "nv12"
                t0 = time.perf_counter_ns()
                out = frame.reformat(width=out_w, height=out_h, format=sw)
                t1 = time.perf_counter_ns()
                if out.format.name != sw:
                    raise RuntimeError(
                        f"baseline output format {out.format.name} != {sw}"
                    )
                if (out.width, out.height) != (out_w, out_h):
                    raise RuntimeError("baseline wrong dimensions")
                timings.append(t1 - t0)
                frames += 1
                if frames >= max_frames:
                    break
        wall = time.perf_counter_ns() - wall0
        cpu = time.process_time_ns() - cpu0
        trimmed = timings[10:] if len(timings) > 20 else timings
        return {
            "mode": "cpu_reformat",
            "frames": frames,
            "operation_median_us": statistics.median(trimmed) / 1e3,
            "operation_p95_us": pctl(trimmed, 0.95) / 1e3,
            "loop_wall_ms_per_frame": wall / frames / 1e6,
            "loop_cpu_ms_per_frame": cpu / frames / 1e6,
            "output_format": sw,
        }

    cand1 = run_candidate()
    base1 = run_baseline()
    base2 = run_baseline()
    cand2 = run_candidate()
    gc.collect()

    cand_med = statistics.median(
        [cand1["operation_median_us"], cand2["operation_median_us"]]
    )
    base_med = statistics.median(
        [base1["operation_median_us"], base2["operation_median_us"]]
    )
    return {
        "path": path,
        "input": {
            "width": first.width,
            "height": first.height,
            "format": first.format.name,
            "sw_format": first.sw_format.name if first.sw_format else None,
        },
        "output": {"width": out_w, "height": out_h},
        "candidate_runs": [cand1, cand2],
        "baseline_runs": [base1, base2],
        "summary": {
            "candidate_median_us": cand_med,
            "baseline_median_us": base_med,
            "baseline_over_candidate": base_med / cand_med,
            "candidate_keeps_hardware_frame": True,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--case",
        action="append",
        nargs=4,
        metavar=("PATH", "OUT_W", "OUT_H", "FRAMES"),
    )
    args = p.parse_args()
    if not args.case:
        raise SystemExit("at least one --case is required")

    cases = []
    for path, w, h, frames in args.case:
        cases.append(bench(path, int(w), int(h), int(frames)))

    print(
        json.dumps(
            {
                "pyav_version": av.__version__,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
