#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import time
from fractions import Fraction

import av
from av.codec.hwaccel import HWAccel, HWDevice


def make_hwaccel():
    return HWAccel(
        device_type="videotoolbox",
        allow_software_fallback=False,
        is_hw_owned=True,
    )


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


def make_encoder(path, out_w, out_h, rate):
    out = av.open(path, "w")
    stream = out.add_stream(
        "h264_videotoolbox",
        rate=rate,
        hwaccel=HWAccel(device_type="videotoolbox"),
    )
    stream.width = out_w
    stream.height = out_h
    stream.bit_rate = 8_000_000
    stream.gop_size = 60
    stream.codec_context.max_b_frames = 0
    return out, stream


def finish_encoder(out, stream):
    packets = 0
    bytes_out = 0
    for packet in stream.encode(None):
        packets += 1
        bytes_out += packet.size
        out.mux(packet)
    out.close()
    return packets, bytes_out


def run_candidate(src_path, out_path, out_w, out_h, max_frames, rate):
    hw = make_hwaccel()
    timings = []
    frames = 0
    packets = 0
    bytes_out = 0

    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()

    out, stream = make_encoder(out_path, out_w, out_h, rate)
    graph = None
    device = None

    with av.open(src_path, hwaccel=hw) as container:
        for frame in container.decode(video=0):
            if frame.format.name != "videotoolbox_vld":
                raise RuntimeError(f"candidate decode fell off hardware: {frame.format.name}")
            if graph is None:
                graph, device = make_scale_graph(frame, out_w, out_h)

            t0 = time.perf_counter_ns()
            graph.vpush(frame)
            scaled = pull_one(graph)
            if scaled.format.name != "videotoolbox_vld":
                raise RuntimeError(f"scale_vt fell off hardware: {scaled.format.name}")
            if (scaled.width, scaled.height) != (out_w, out_h):
                raise RuntimeError("candidate wrong dimensions")

            scaled.pts = frames
            scaled.time_base = Fraction(1, rate)
            for packet in stream.encode(scaled):
                packets += 1
                bytes_out += packet.size
                out.mux(packet)
            timings.append(time.perf_counter_ns() - t0)

            frames += 1
            if frames >= max_frames:
                break

    p, b = finish_encoder(out, stream)
    packets += p
    bytes_out += b

    wall = time.perf_counter_ns() - wall0
    cpu = time.process_time_ns() - cpu0
    trimmed = timings[5:] if len(timings) > 10 else timings

    return {
        "mode": "vt_decode_scale_vt_vt_encode",
        "frames": frames,
        "packets": packets,
        "bytes_out": bytes_out,
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "loop_wall_ms_per_frame": wall / frames / 1e6,
        "loop_cpu_ms_per_frame": cpu / frames / 1e6,
        "output_exists": os.path.getsize(out_path) > 0,
    }


def run_baseline(src_path, out_path, out_w, out_h, max_frames, rate):
    hw = make_hwaccel()
    timings = []
    frames = 0
    packets = 0
    bytes_out = 0

    wall0 = time.perf_counter_ns()
    cpu0 = time.process_time_ns()

    out, stream = make_encoder(out_path, out_w, out_h, rate)

    with av.open(src_path, hwaccel=hw) as container:
        for frame in container.decode(video=0):
            sw = frame.sw_format.name if frame.sw_format else "nv12"

            t0 = time.perf_counter_ns()
            scaled = frame.reformat(width=out_w, height=out_h, format=sw)
            if scaled.format.name != sw:
                raise RuntimeError(f"baseline wrong format: {scaled.format.name}")
            if (scaled.width, scaled.height) != (out_w, out_h):
                raise RuntimeError("baseline wrong dimensions")

            scaled.pts = frames
            scaled.time_base = Fraction(1, rate)
            for packet in stream.encode(scaled):
                packets += 1
                bytes_out += packet.size
                out.mux(packet)
            timings.append(time.perf_counter_ns() - t0)

            frames += 1
            if frames >= max_frames:
                break

    p, b = finish_encoder(out, stream)
    packets += p
    bytes_out += b

    wall = time.perf_counter_ns() - wall0
    cpu = time.process_time_ns() - cpu0
    trimmed = timings[5:] if len(timings) > 10 else timings

    return {
        "mode": "vt_decode_cpu_reformat_vt_upload_encode",
        "frames": frames,
        "packets": packets,
        "bytes_out": bytes_out,
        "operation_median_us": statistics.median(trimmed) / 1e3,
        "loop_wall_ms_per_frame": wall / frames / 1e6,
        "loop_cpu_ms_per_frame": cpu / frames / 1e6,
        "output_exists": os.path.getsize(out_path) > 0,
    }


def benchmark(path, out_w, out_h, max_frames, rate):
    # Candidate/base/candidate/base minimizes order bias while keeping every
    # encoder/decoder/container instance fresh.
    runs = []
    for idx, mode in enumerate(("candidate", "baseline", "baseline", "candidate")):
        out_path = f"/tmp/puma_pipeline_{mode}_{idx}.mp4"
        if mode == "candidate":
            result = run_candidate(path, out_path, out_w, out_h, max_frames, rate)
        else:
            result = run_baseline(path, out_path, out_w, out_h, max_frames, rate)
        if not result["output_exists"] or result["packets"] == 0:
            raise RuntimeError(f"{mode} produced no encoded output")
        runs.append(result)

    candidate = [x for x in runs if x["mode"].startswith("vt_decode_scale")]
    baseline = [x for x in runs if x["mode"].startswith("vt_decode_cpu")]

    cand_wall = statistics.median(x["loop_wall_ms_per_frame"] for x in candidate)
    base_wall = statistics.median(x["loop_wall_ms_per_frame"] for x in baseline)
    cand_cpu = statistics.median(x["loop_cpu_ms_per_frame"] for x in candidate)
    base_cpu = statistics.median(x["loop_cpu_ms_per_frame"] for x in baseline)
    cand_op = statistics.median(x["operation_median_us"] for x in candidate)
    base_op = statistics.median(x["operation_median_us"] for x in baseline)

    return {
        "input": path,
        "output": {"width": out_w, "height": out_h, "codec": "h264_videotoolbox"},
        "runs": runs,
        "summary": {
            "candidate_operation_us": cand_op,
            "baseline_operation_us": base_op,
            "operation_speedup": base_op / cand_op,
            "candidate_wall_ms_per_frame": cand_wall,
            "baseline_wall_ms_per_frame": base_wall,
            "end_to_end_wall_speedup": base_wall / cand_wall,
            "candidate_cpu_ms_per_frame": cand_cpu,
            "baseline_cpu_ms_per_frame": base_cpu,
            "cpu_time_reduction": base_cpu / cand_cpu,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--out-width", type=int, default=1920)
    p.add_argument("--out-height", type=int, default=1080)
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--rate", type=int, default=60)
    args = p.parse_args()

    result = benchmark(
        args.path,
        args.out_width,
        args.out_height,
        args.frames,
        args.rate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
