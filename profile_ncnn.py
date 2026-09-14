#!/usr/bin/env python3
"""
Profile an NCNN network's inference time, feeding it random input data.
Also prints a best-effort static estimate of parameter count and
MACs/output-pixel, from a plain-text parse of the .param file (no need to
run the network to get these numbers, but the two analyses use the same
--width/--height/--channels).

Usage:
    ./profile_ncnn.py <name> [options]

<name> is the network's .param/.bin pair, without the extension, e.g.

    ./profile_ncnn.py src/ipa/rpi/pisp/data/ncnn/imx662_32_2221e1221_1936x1104_fp16

The default --width/--height/--channels (968x552x4) match denoise_ncnn.cpp's
network; pass different values for a network with a different input shape
(e.g. --channels 9 for model_denoise's imx662_ncnn network -- though that one
uses custom layers this script's plain ncnn.Net can't load at all).

--processes N profiles N fully independent ncnn instances running
concurrently (separate OS processes, each with its own loaded net and its
own --threads), reporting one aggregate ms/call figure across all of them --
i.e. throughput under real concurrent load, not per-instance latency. Each
worker is a genuinely separate process (this script re-invoked via
subprocess), not a fork()ed child of an already-import-numpy/ncnn-initialised
parent -- forking after OpenMP has initialised in the parent is a known
deadlock hazard (that's what multiprocessing's default 'fork' start method
would do here), which subprocess's fork+exec avoids since exec() replaces
the process image outright.

Each worker signals readiness once its model is loaded and warmup is done,
and the parent releases all of them together, so every worker's *timed*
loop starts at roughly the same moment. The aggregate figure is computed
from those synchronized timed loops alone -- process startup, imports,
model load and --warmup are excluded -- so it reflects genuine concurrent
throughput regardless of --loops (unlike naively timing from process launch
to exit, where that fixed per-process overhead would dominate at low
--loops and understate throughput).

--thread-instances N is the single-process analogue of --processes: it
loads N fully independent ncnn::Net instances (each with its own --threads
worker threads) and runs them concurrently as Python threads of this one
process, rather than as separate OS processes. Unlike passing a plain
--threads N (which hands one shared instance's *internal* OpenMP threading
more workers -- often with diminishing returns), each --thread-instances
worker has its own net, its own extractor and its own timed loop, so this
measures genuine independent throughput -- as long as ncnn's C++ inference
releases the GIL while it runs, in which case the Python threads really do
overlap. All N instances warm up, then synchronize on a threading.Barrier
before their timed loops start, mirroring --processes' readiness handshake
so the aggregate figure reflects the synchronized concurrent phase alone.
Mutually exclusive with --processes.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import ncnn
import numpy as np

# ncnn param-file layer types this script knows have NO learnable weights
# and don't need special-casing below (shape-preserving or otherwise inert
# for our purposes). Anything else falls through to the "unhandled" case.
_SHAPE_PRESERVING = {
    "ReLU", "ReLU6", "Sigmoid", "TanH", "Clip", "Softmax", "Dropout",
    "Swish", "HardSwish", "HardSigmoid", "GELU", "Padding", "Cast", "Noop",
}


def _parse_params(tokens):
    p = {}
    for t in tokens:
        k, v = t.split("=", 1)
        k = int(k)
        if k < 0:
            vals = v.split(",")
            p[k] = [float(x) if ("." in x or "e" in x.lower()) else int(x) for x in vals[1:]]
        else:
            p[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
    return p


def analyse_param_file(path, in_w, in_h, in_c):
    """
    Best-effort static analysis of an ncnn .param file for a given input
    shape. Returns (num_params, num_macs, out_w, out_h, approximate,
    unhandled_types). Layer types this function doesn't understand are
    passed through with an unchanged shape and flagged in unhandled_types
    -- the result is then approximate, not wrong by omission.
    """
    lines = open(path).read().splitlines()
    n_layer, _ = map(int, lines[1].split())

    shape = {}
    num_params = 0
    num_macs = 0
    unhandled = set()
    last_out = None

    for line in lines[2:2 + n_layer]:
        parts = line.split()
        ltype, lname = parts[0], parts[1]
        nin, nout = int(parts[2]), int(parts[3])
        ins = parts[4:4 + nin]
        outs = parts[4 + nin:4 + nin + nout]
        params = _parse_params(parts[4 + nin + nout:])

        if ltype == "Input":
            shape[outs[0]] = (in_w, in_h, in_c)
        elif ltype == "MemoryData":
            w, h, c = params.get(0, 1), params.get(1, 1), params.get(2, 1)
            shape[outs[0]] = (w, h, c)
            num_params += w * h * c
        elif ltype == "Split":
            for o in outs:
                shape[o] = shape[ins[0]]
        elif ltype == "Slice":
            w, h, c = shape[ins[0]]
            axis = params.get(1, 0)
            sizes = params.get(-23300, [])
            dim = c if axis == 0 else h if axis == 1 else w
            n_auto = sum(1 for s in sizes if s == -233)
            explicit_sum = sum(s for s in sizes if s != -233)
            auto_size = (dim - explicit_sum) // n_auto if n_auto else 0
            for o, s in zip(outs, sizes):
                sz = auto_size if s == -233 else s
                shape[o] = (w, h, sz) if axis == 0 else \
                           (w, sz, c) if axis == 1 else (sz, h, c)
        elif ltype == "Crop":
            w, h, c = shape[ins[0]]
            st = params.get(-23309, [0])
            en = params.get(-23310, [c])
            ax = params.get(-23311, [0])
            nw, nh, nc = w, h, c
            for s, e, a in zip(st, en, ax):
                if a == 0:
                    nc = e - s
                elif a == 1:
                    nh = e - s
                elif a == 2:
                    nw = e - s
            shape[outs[0]] = (nw, nh, nc)
        elif ltype == "Concat":
            w, h, c = shape[ins[0]]
            axis = params.get(0, 0)
            if axis == 0:
                c = sum(shape[i][2] for i in ins)
            elif axis == 1:
                h = sum(shape[i][1] for i in ins)
            elif axis == 2:
                w = sum(shape[i][0] for i in ins)
            shape[outs[0]] = (w, h, c)
        elif ltype == "BinaryOp":
            shape[outs[0]] = shape[ins[0]]
        elif ltype == "LayerNorm":
            w, h, c = shape[ins[0]]
            shape[outs[0]] = (w, h, c)
            affine_size = params.get(0, c)
            if params.get(2, 0):
                num_params += 2 * affine_size
        elif ltype in ("BatchNorm", "Scale"):
            w, h, c = shape[ins[0]]
            shape[outs[0]] = (w, h, c)
            num_params += 2 * c
        elif ltype == "PReLU":
            w, h, c = shape[ins[0]]
            shape[outs[0]] = (w, h, c)
            num_params += params.get(0, c)
        elif ltype == "Pooling":
            w, h, c = shape[ins[0]]
            if params.get(4, 0):
                shape[outs[0]] = (1, 1, c)
            else:
                kw = params.get(1, 1)
                kh = params.get(11, kw)
                sw = params.get(2, 1)
                sh = params.get(12, sw)
                pl = params.get(3, 0)
                pt = params.get(13, pl)
                ow = max((w + 2 * pl - kw) // sw + 1, 1)
                oh = max((h + 2 * pt - kh) // sh + 1, 1)
                shape[outs[0]] = (ow, oh, c)
        elif ltype == "Interp":
            w, h, c = shape[ins[0]]
            oh = params.get(3, 0) or h
            ow = params.get(4, 0) or w
            shape[outs[0]] = (ow, oh, c)
        elif ltype == "InnerProduct":
            w, h, c = shape[ins[0]]
            num_out = params.get(0, 0)
            wsize = params.get(2, 0)
            bias = params.get(1, 0)
            num_params += wsize + (num_out if bias else 0)
            num_macs += num_out * (w * h * c)
            shape[outs[0]] = (1, 1, num_out)
        elif ltype in ("Convolution", "ConvolutionDepthWise"):
            w, h, c = shape[ins[0]]
            num_out = params.get(0, 0)
            kw = params.get(1, 1)
            kh = params.get(11, kw)
            dw = params.get(2, 1)
            dh = params.get(12, dw)
            sw = params.get(3, 1)
            sh = params.get(13, sw)
            pl = params.get(4, 0)
            pt = params.get(14, pl)
            group = params.get(7, 1) if ltype == "ConvolutionDepthWise" else 1
            ekw = dw * (kw - 1) + 1
            ekh = dh * (kh - 1) + 1
            ow = max((w + 2 * pl - ekw) // sw + 1, 1)
            oh = max((h + 2 * pt - ekh) // sh + 1, 1)
            shape[outs[0]] = (ow, oh, num_out)
            in_pg = c // group if ltype == "ConvolutionDepthWise" else c
            num_macs += ow * oh * num_out * kw * kh * in_pg
            wsize = params.get(6, 0)
            bias = params.get(5, 0)
            num_params += wsize + (num_out if bias else 0)
        elif ltype in _SHAPE_PRESERVING:
            shape[outs[0]] = shape[ins[0]]
        else:
            unhandled.add(ltype)
            if ins and ins[0] in shape:
                for o in outs:
                    shape[o] = shape[ins[0]]

        if outs and outs[0] in shape:
            last_out = shape[outs[0]]

    out_w, out_h = (last_out[0], last_out[1]) if last_out else (0, 0)
    return num_params, num_macs, out_w, out_h, bool(unhandled), unhandled


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="network name, excluding .param/.bin")
    parser.add_argument("--width", type=int, default=968, help="input width (default: 968)")
    parser.add_argument("--height", type=int, default=552, help="input height (default: 552)")
    parser.add_argument("--channels", type=int, default=4, help="input channels (default: 4)")
    parser.add_argument("--threads", type=int, default=ncnn.get_cpu_count(),
                         help="ncnn worker threads, per process (default: all cores)")
    parser.add_argument("--warmup", type=int, default=3, help="warm-up runs, discarded (default: 3)")
    parser.add_argument("--loops", type=int, default=10, help="timed runs to average (default: 10)")
    parser.add_argument("--fp32", action="store_true", help="disable fp16 storage/arithmetic")
    parser.add_argument("--input", help="input blob name (default: autodetect)")
    parser.add_argument("--output", help="output blob name (default: autodetect)")
    parser.add_argument("--processes", type=int, default=1,
                         help="run this many independent ncnn instances concurrently, in "
                              "separate processes, and report aggregate ms/call across all "
                              "of them (default: 1, i.e. the normal single-instance report)")
    parser.add_argument("--thread-instances", type=int, default=1, dest="thread_instances",
                         help="run this many independent ncnn instances concurrently, as "
                              "Python threads of this single process, and report aggregate "
                              "ms/call across all of them (default: 1; mutually exclusive "
                              "with --processes)")
    # Internal: re-invoked by --processes to run one worker. Not for direct use.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_inference(args, pre_loop_hook=None):
    """Load the network and run warmup + timed loops. Returns (times_ms,
    last output Mat, input_name, output_name).

    If given, pre_loop_hook() is called once after warmup and before the
    timed loop starts -- used by worker_main() to synchronize concurrent
    workers so their timed loops begin together."""
    net = ncnn.Net()
    net.opt.num_threads = args.threads
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_packed = not args.fp32
    net.opt.use_fp16_storage = not args.fp32
    net.opt.use_fp16_arithmetic = not args.fp32

    if net.load_param(args.name + ".param") != 0:
        raise SystemExit(f"Failed to load {args.name}.param")
    if net.load_model(args.name + ".bin") != 0:
        raise SystemExit(f"Failed to load {args.name}.bin")

    input_name = args.input or net.input_names()[0]
    output_name = args.output or net.output_names()[0]

    in_mat = ncnn.Mat(args.width, args.height, args.channels)
    arr = in_mat.numpy()
    arr[...] = np.random.default_rng().random(arr.shape, dtype=np.float32)

    def run_once():
        ex = net.create_extractor()
        ex.input(input_name, in_mat)
        ret, out = ex.extract(output_name)
        if ret != 0:
            raise RuntimeError(f"extract() failed ({ret})")
        return out

    for _ in range(args.warmup):
        run_once()

    if pre_loop_hook:
        pre_loop_hook()

    times_ms = []
    out = None
    for _ in range(args.loops):
        t0 = time.perf_counter()
        out = run_once()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return times_ms, out, input_name, output_name


def print_param_macs_report(args):
    try:
        num_params, num_macs, out_w, out_h, approx, unhandled = analyse_param_file(
            args.name + ".param", args.width, args.height, args.channels)
        out_pixels = out_w * out_h
        note = f" (approximate -- unhandled layer types: {', '.join(sorted(unhandled))})" if approx else ""
        print(f"\nParameters: {num_params:,} (~{num_params * 2 / 1e6:.2f} MB at fp16)")
        if out_pixels:
            print(f"MACs/output pixel: {num_macs / out_pixels:,.1f}{note}")
        else:
            print(f"MACs/output pixel: unknown (couldn't determine output size){note}")
    except Exception as e:
        print(f"\n(could not estimate parameters/MACs: {e})")


def _wait_for_release():
    """Signal readiness to the parent (once model load + warmup are done)
    and block until it releases every worker together, so timed loops
    across concurrent workers start at roughly the same moment."""
    print("__READY__", flush=True)
    sys.stdin.readline()


def worker_main(args):
    """Child process entry point (--_worker): run the timed loop and print
    nothing but a single JSON line of per-run ms times, for the parent to
    collect. Before the timed loop, synchronizes with the parent via
    _wait_for_release() -- see that function and multi_main()."""
    times_ms, _, _, _ = run_inference(args, pre_loop_hook=_wait_for_release)
    print(json.dumps(times_ms))


def single_main(args):
    times_ms, out, input_name, output_name = run_inference(args)

    print(f"Network:  {args.name}")
    print(f"Input:    '{input_name}' {args.width}x{args.height}x{args.channels}")
    print(f"Output:   '{output_name}'")
    print(f"Threads:  {args.threads}   fp16: {not args.fp32}")
    print(f"Output:   {out.w}x{out.h}x{out.c} (elempack={out.elempack})")

    times_ms = sorted(times_ms)
    print(f"\n{args.loops} runs, ms: "
          f"min={times_ms[0]:.2f} "
          f"median={times_ms[len(times_ms) // 2]:.2f} "
          f"max={times_ms[-1]:.2f} "
          f"mean={sum(times_ms) / len(times_ms):.2f}")

    print_param_macs_report(args)


def worker_command(args):
    cmd = [sys.executable, os.path.abspath(__file__), args.name,
           "--width", str(args.width), "--height", str(args.height),
           "--channels", str(args.channels), "--threads", str(args.threads),
           "--warmup", str(args.warmup), "--loops", str(args.loops)]
    if args.fp32:
        cmd.append("--fp32")
    if args.input:
        cmd += ["--input", args.input]
    if args.output:
        cmd += ["--output", args.output]
    cmd.append("--_worker")
    return cmd


def multi_main(args):
    print(f"Network:   {args.name}")
    print(f"Input:     {args.width}x{args.height}x{args.channels}")
    print(f"Processes: {args.processes} (each threads={args.threads}, fp16={not args.fp32})")
    print(f"Per process: {args.warmup} warmup + {args.loops} timed runs")

    cmd = worker_command(args)
    t0 = time.perf_counter()
    procs = [subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
             for _ in range(args.processes)]

    # Wait for every worker to finish loading + warmup (see _wait_for_release()),
    # then release them all together so their timed loops start in sync. A
    # worker that dies before signalling just hits EOF here rather than
    # hanging -- its failure is reported below via returncode.
    for p in procs:
        while True:
            line = p.stdout.readline()
            if line == "" or line.strip() == "__READY__":
                break
    for p in procs:
        if p.stdin:
            try:
                p.stdin.write("GO\n")
                p.stdin.flush()
            except OSError:
                pass  # worker already gone; its failure shows up via returncode

    results = [p.communicate() for p in procs]
    wall_s = time.perf_counter() - t0

    all_times = []
    for i, (proc, (stdout, stderr)) in enumerate(zip(procs, results)):
        if proc.returncode != 0:
            print(f"  process {i}: FAILED (exit {proc.returncode}): {stderr.strip()[-300:]}")
            continue
        try:
            all_times.append(json.loads(stdout.strip().splitlines()[-1]))
        except Exception as e:
            print(f"  process {i}: could not parse output ({e}): {stdout!r}")

    ok = len(all_times)
    if ok < args.processes:
        print(f"\n{args.processes - ok}/{args.processes} worker process(es) failed -- "
              f"aggregate below only reflects the {ok} that succeeded")

    total_calls = sum(len(t) for t in all_times)
    # Each worker's timed loop is sequential, so sum(times) is that worker's
    # own timed-loop duration. All timed loops started together (the
    # readiness handshake above), so the slowest one bounds how long the
    # synchronized concurrent phase actually took -- unlike wall_s, this
    # excludes per-process startup/model-load/warmup entirely.
    concurrent_s = max((sum(t) / 1000.0 for t in all_times), default=0.0)
    print(f"\n{ok} concurrent processes completed, {total_calls} total calls, "
          f"startup+load+warmup+timed wall={wall_s * 1000:.1f}ms")
    if total_calls and concurrent_s > 0:
        print(f"Aggregate throughput: {concurrent_s * 1000 / total_calls:.2f} ms/call "
              f"(= {total_calls / concurrent_s:.1f} calls/s across all {ok} processes)")
    for i, times in enumerate(all_times):
        times = sorted(times)
        print(f"  process {i}: mean={sum(times) / len(times):.2f}ms "
              f"min={times[0]:.2f} max={times[-1]:.2f}")

    print_param_macs_report(args)


def thread_main(args):
    n = args.thread_instances
    print(f"Network:  {args.name}")
    print(f"Input:    {args.width}x{args.height}x{args.channels}")
    print(f"Threads:  {n} concurrent ncnn instances in this process "
          f"(each threads={args.threads}, fp16={not args.fp32})")
    print(f"Per instance: {args.warmup} warmup + {args.loops} timed runs")

    # Mirrors multi_main()'s readiness handshake, but with a Barrier instead
    # of a pipe: every instance loads its model and runs warmup, then blocks
    # on the barrier, so all N timed loops start together. If any instance
    # raises before reaching the barrier (e.g. a model load failure), it
    # aborts the barrier so the others don't hang forever waiting for it.
    barrier = threading.Barrier(n)
    results = [None] * n
    errors = [None] * n

    def run_one(i):
        try:
            times_ms, _, _, _ = run_inference(args, pre_loop_hook=barrier.wait)
            results[i] = times_ms
        except Exception as e:
            errors[i] = e
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass

    threads = [threading.Thread(target=run_one, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.perf_counter() - t0

    all_times = []
    for i in range(n):
        if errors[i] is not None:
            print(f"  instance {i}: FAILED: {errors[i]}")
        elif results[i] is not None:
            all_times.append(results[i])

    ok = len(all_times)
    if ok < n:
        print(f"\n{n - ok}/{n} thread instance(s) failed -- "
              f"aggregate below only reflects the {ok} that succeeded")

    total_calls = sum(len(t) for t in all_times)
    # As in multi_main(): each instance's timed loop is sequential, so
    # sum(times) is that instance's own timed-loop duration, and since all
    # timed loops started together (the barrier above), the slowest one
    # bounds how long the synchronized concurrent phase actually took.
    concurrent_s = max((sum(t) / 1000.0 for t in all_times), default=0.0)
    print(f"\n{ok} concurrent thread instances completed, {total_calls} total calls, "
          f"load+warmup+timed wall={wall_s * 1000:.1f}ms")
    if total_calls and concurrent_s > 0:
        print(f"Aggregate throughput: {concurrent_s * 1000 / total_calls:.2f} ms/call "
              f"(= {total_calls / concurrent_s:.1f} calls/s across all {ok} instances)")
    for i, times in enumerate(all_times):
        times = sorted(times)
        print(f"  instance {i}: mean={sum(times) / len(times):.2f}ms "
              f"min={times[0]:.2f} max={times[-1]:.2f}")

    print_param_macs_report(args)


def main():
    args = build_parser().parse_args()

    if args.processes > 1 and args.thread_instances > 1:
        raise SystemExit("--processes and --thread-instances are mutually exclusive")

    if args._worker:
        worker_main(args)
        return

    print(f"processes={args.processes}  threads={args.threads}  thread-instances={args.thread_instances}")

    if args.processes > 1:
        multi_main(args)
    elif args.thread_instances > 1:
        thread_main(args)
    else:
        single_main(args)


if __name__ == "__main__":
    main()
