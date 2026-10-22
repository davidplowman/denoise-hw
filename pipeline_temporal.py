#!/usr/bin/env python3
"""Train -> test -> select -> QAT -> convert pipeline for the temporal DenoiseNet.

Chains train_temporal.py, test_temporal.py and convert_temporal.py (each run
as its own process, exactly as from the command line), writing everything
for one run to its own directory, experiments/<name>/:

    pipeline.log         every command and all of its output
    summary.txt          settings and the final PSNR report (also printed)
    loop<N>/model.pth    each float training loop's best checkpoint
    loop<N>/test/<clip>/ its test_temporal.py outputs (viewable PNGs)
    float_best.pth       the selected float loop's checkpoint
    qat_loop<N>/         QAT of float loop N: model.pth, and test/<clip>/ its tests
    qat_best.pth         the selected QAT checkpoint (when there were several;
                         from --float-checkpoint, it's just qat/model.pth)
    <name>_16x<h>x<w>_{fp32,int8}.tflite

Stages:
  1. Float training: --loops runs of train_temporal.py with the standard
     settings (blend output, diff 50, temporal 10, candidate 0.1 (--cand),
     gate clamp 8, gains 32-128, 15-frame sequences, ...). --wide-low gives
     the levels below full resolution 2x --ch channels (a "w" in the run
     name).
  2. Each loop's checkpoint is tested on --test-clips (test_temporal.py, gain
     64, seed 0, so every loop sees identical noise).
  3. The loop with the best mean temporal PSNR over the test clips is
     selected.
  4. Optionally (--qat), the --qat-top best loops (default 2) are each QAT
     fine-tuned (train_temporal.py --qat) and tested again, and the best
     (simulated-int8 test PSNR) is kept: a better float model doesn't always
     quantise better.
  5. Optionally (--convert), the final model is converted to .tflite and
     verified closed-loop against PyTorch (convert_temporal.py --verify).

Starting from --float-checkpoint skips stages 1-3; from --qat-checkpoint,
stages 1-4 (it's tested, then converted).

Examples:
    # float training only: 3 loops, tested on cabinet_D65_100_local_0, best one selected
    python pipeline_temporal.py

    # float training, then QAT of the 2 best loops, keeping the better
    python pipeline_temporal.py --qat

    # float training, QAT, then conversion to both fp32 and int8 .tflite
    python pipeline_temporal.py --qat --convert both

    # a design experiment: other network sizes, 5 loops, two test clips
    python pipeline_temporal.py --ch 32 --scaling "2 2" --encoders "1 1 1" --loops 5 \\
        --test-clips D65_100_local_0 D50_36_static_0 --name ch32_3level

    # start from an existing float model: QAT, then int8 conversion
    python pipeline_temporal.py --float-checkpoint checkpoints/imx662_64_2e11_g32-128_d50_t10_b_c01.pth \\
        --qat --convert int8

    # start from a QAT-tuned model: test and convert only
    python pipeline_temporal.py --qat-checkpoint checkpoints/run_64_2e11_g32-128_d50_t10_blend_c0.1_ft_gc8_qat_1.pth \\
        --convert int8
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# train_temporal.py settings shared by float training and QAT (the loss and
# data setup that run_temporal.sh uses)
STANDARD_TRAIN_ARGS = [
    "--gain-min", "32", "--gain-max", "192",
    "--diff-weight", "100", "--temporal-weight", "10",
    "--seq-len", "15", "--window-stride", "15", "--val-frac", "0.3",
    "--flicker-scale", "4", "--temporal-start", "3",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, test, select, QAT fine-tune and convert the temporal DenoiseNet.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Examples:")[1],
    )
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--float-checkpoint", type=Path, default=None, metavar="PTH",
                       help="start from this float model (skips float training)")
    start.add_argument("--qat-checkpoint", type=Path, default=None, metavar="PTH",
                       help="start from this QAT-tuned model (test and convert only)")

    parser.add_argument("--loops", type=int, default=3,
                        help="float training runs to choose the best from (default: %(default)s)")
    parser.add_argument("--dataset", default="/opt/datasets/IMX662_temporal", metavar="ROOT",
                        help="training and test clips (default: %(default)s)")
    parser.add_argument("--ch", type=int, default=64, help="channels (default: %(default)s)")
    parser.add_argument("--scaling", default="2",
                        help='space-separated scaling factors (default: "%(default)s")')
    parser.add_argument("--encoders", default="1 1",
                        help='space-separated encoder counts per level (default: "%(default)s")')
    parser.add_argument("--wide-low", action="store_true",
                        help="2x --ch channels at every level below full resolution")
    parser.add_argument("--gate-clamp", type=float, default=8.0,
                        help="clamp the gates' pre-sigmoid values to +-G, for int8 (0 = "
                             "off; with --float-checkpoint, 0 keeps the model's own) "
                             "(default: %(default)s)")
    parser.add_argument("--cand", type=float, default=0.1,
                        help="candidate loss weight (train_temporal.py --candidate-weight), "
                             "for float training and QAT (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=2e-3, help="float training (default: %(default)s)")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="stop when the learning rate drops below this (default: %(default)s)")
    parser.add_argument("--lr-patience", type=int, default=10,
                        help="epochs without improvement before halving the learning rate "
                             "(default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="float training epoch limit (default: %(default)s)")
    parser.add_argument("--train-args", default="", metavar="ARGS",
                        help="extra arguments for every train_temporal.py run, float and QAT "
                             '(e.g. "--batch-size 32")')

    parser.add_argument("--qat", action="store_true", help="QAT fine-tune the selected model")
    parser.add_argument("--qat-top", type=int, default=2, metavar="N",
                        help="QAT fine-tune each of the N best float loops and keep the best "
                             "result (simulated-int8 test PSNR) (default: %(default)s)")
    parser.add_argument("--qat-lr", type=float, default=2e-4, help="(default: %(default)s)")
    parser.add_argument("--qat-epochs", type=int, default=600, help="(default: %(default)s)")
    parser.add_argument("--qat-freeze-epoch", type=int, default=20,
                        help="freeze the int8 ranges after this epoch (default: %(default)s)")

    parser.add_argument("--convert", choices=["none", "fp32", "int8", "both"], default="none",
                        help="convert the final model to .tflite (default: %(default)s)")

    parser.add_argument("--test-clips", nargs="+", default=["cabinet_D65_100_local_0"], metavar="CLIP",
                        help="clips to test on, by folder name or a unique suffix of one "
                             "(default: %(default)s)")
    parser.add_argument("--test-gain", type=float, default=64.0, help="(default: %(default)s)")
    parser.add_argument("--still-checkpoint", default="checkpoints/imx662_64_2e11.pth",
                        metavar="PTH", help='single-image model the tests compare with ("" for '
                                            "none) (default: %(default)s)")

    parser.add_argument("--name", default=None,
                        help="run directory name (default: from the settings and the time)")
    parser.add_argument("--out-root", type=Path, default=Path("experiments"),
                        help="parent of the run directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="allow writing into an existing run directory")
    args = parser.parse_args(argv)
    if args.loops < 1:
        parser.error("--loops must be at least 1")
    if args.qat_top < 1:
        parser.error("--qat-top must be at least 1")
    if args.qat and args.qat_checkpoint:
        parser.error("--qat with --qat-checkpoint: that model is already QAT-tuned")
    for name in ("float_checkpoint", "qat_checkpoint"):
        ck = getattr(args, name)
        if ck is not None:
            if not ck.exists():
                parser.error(f"checkpoint not found: {ck}")
            setattr(args, name, ck.resolve())  # the stages run from this script's directory
    if args.still_checkpoint and Path(args.still_checkpoint).exists():
        args.still_checkpoint = str(Path(args.still_checkpoint).resolve())
    return args


class Pipeline:
    def __init__(self, args: argparse.Namespace, run_dir: Path, clips: list[str]) -> None:
        self.args = args
        self.dir = run_dir
        self.clips = clips
        self.log_file = open(run_dir / "pipeline.log", "a", buffering=1)
        self.report: list[tuple[str, dict]] = []  # (stage, result) rows for the summary
        self.selected: set[str] = set()  # stages chosen to carry forward, marked in it
        self.notes: list[str] = []

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.log_file.write(msg + "\n")

    def run(self, cmd: list[str], what: str) -> bool:
        """Run cmd, streaming its output to the terminal and the log."""
        self.log(f"\n{'=' * 78}\n{what}\n$ {shlex.join(cmd)}\n{'=' * 78}")
        t0 = time.time()
        # unbuffered, or the child's output (going to a pipe, not a terminal)
        # arrives in large blocks rather than line by line
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            self.log_file.write(line)
        proc.wait()
        status = "done" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
        self.log(f"--- {what}: {status} in {(time.time() - t0) / 60:.1f} min")
        return proc.returncode == 0

    # ---- stages ----------------------------------------------------------

    def train(self, out: Path, extra: list[str], what: str) -> bool:
        cmd = [sys.executable, "train_temporal.py", "--no-pbar", "--dataset", self.args.dataset,
               "--checkpoint-out", str(out), "--lr-min", str(self.args.lr_min),
               "--lr-patience", str(self.args.lr_patience),
               *STANDARD_TRAIN_ARGS, "--candidate-weight", str(self.args.cand), *extra, *shlex.split(self.args.train_args)]
        return self.run(cmd, what) and out.exists()

    def train_float(self, loop: int) -> Path | None:
        a = self.args
        out = self.dir / f"loop{loop}" / "model.pth"
        extra = ["--ch", str(a.ch), "--scaling", a.scaling, "--encoders", *a.encoders.split(),
                 "--blend-output", "--lr", str(a.lr), "--epochs", str(a.epochs)]
        if a.wide_low:
            extra += ["--wide-low"]
        if a.gate_clamp:
            extra += ["--gate-clamp", str(a.gate_clamp)]
        ok = self.train(out, extra, f"Float training, loop {loop}/{a.loops}")
        return out if ok else None

    def train_qat(self, start: Path, out_dir: Path) -> Path | None:
        a = self.args
        out = out_dir / "model.pth"
        extra = ["--resume", str(start), "--qat", "--lr", str(a.qat_lr),
                 "--epochs", str(a.qat_epochs), "--qat-freeze-epoch", str(a.qat_freeze_epoch)]
        if a.gate_clamp:
            extra += ["--gate-clamp", str(a.gate_clamp)]
        ok = self.train(out, extra, f"QAT fine-tuning of {start}")
        return out if ok else None

    def test(self, ckpt: Path, out_dir: Path, what: str, still_int8: bool = False) -> dict | None:
        """Test ckpt on every test clip; returns {clip: test_temporal.py's
        summary outputs}, or None if any test failed. still_int8 also runs
        the still model in simulated int8 (for QAT models)."""
        a = self.args
        results = {}
        for clip in self.clips:
            summary = out_dir / clip / "summary.json"
            cmd = [sys.executable, "test_temporal.py", clip, "--dataset", a.dataset,
                   "--checkpoint", str(ckpt), "--gain", str(a.test_gain), "--seed", "0",
                   "--change-threshold", "0.05", "--save-alpha",
                   "--output", str(out_dir / clip), "--summary-json", str(summary)]
            if a.still_checkpoint:
                cmd += ["--still-checkpoint", a.still_checkpoint]
                if still_int8:
                    cmd += ["--still-int8"]
            if not self.run(cmd, f"Test of {what} on {clip}") or not summary.exists():
                return None
            results[clip] = json.loads(summary.read_text())["outputs"]
        return results

    def convert(self, ckpt: Path, quantize: str) -> dict | None:
        summary = self.dir / f"convert_{quantize}.json"
        cmd = [sys.executable, "convert_temporal.py", str(ckpt), "--quantize", quantize,
               "--dataset", self.args.dataset, "--output-dir", str(self.dir),
               "--output-stem", self.dir.name, "--verify", "--verify-clip", self.clips[0],
               "--verify-gain", str(self.args.test_gain), "--summary-json", str(summary)]
        if not self.run(cmd, f"Conversion to {quantize} .tflite") or not summary.exists():
            return None
        return json.loads(summary.read_text())

    # ---- driver ----------------------------------------------------------

    def main(self) -> int:
        a = self.args
        self.log(f"Pipeline run {self.dir}  ({time.strftime('%Y-%m-%d %H:%M:%S')})")
        self.log(f"$ {shlex.join([sys.argv[0], *sys.argv[1:]])}")
        self.log(f"git: {_git_state()}")
        self.log(f"Test clips: {', '.join(self.clips)}")
        t_start = time.time()

        # stages 1-3: float model
        if a.qat_checkpoint:
            final = a.qat_checkpoint
            result = self.test(final, self.dir / "qat" / "test", "the given QAT model",
                               still_int8=True)
            self.report.append(("given QAT model (sim int8)", result or {}))
        else:
            if a.float_checkpoint:
                best = a.float_checkpoint
                result = self.test(best, self.dir / "float" / "test", "the given float model")
                self.report.append(("given float model", result or {}))
            else:
                loops = []
                for i in range(1, a.loops + 1):
                    ckpt = self.train_float(i)
                    if ckpt is None:
                        self.notes.append(f"float loop {i}: training failed")
                        continue
                    result = self.test(ckpt, ckpt.parent / "test", f"float loop {i}")
                    if result is None:
                        self.notes.append(f"float loop {i}: test failed")
                        continue
                    loops.append((i, ckpt, result))
                    self.report.append((f"float loop {i}", result))
                if not loops:
                    self.log("\nNo float loop trained and tested successfully; stopping.")
                    self.write_summary(t_start)
                    return 1
                loops.sort(key=lambda l: _mean(l[2], "temporal", "psnr"), reverse=True)
                i, ckpt, _ = loops[0]
                best = self.dir / "float_best.pth"
                shutil.copy(ckpt, best)
                self.notes.append(f"selected float loop {i} (best mean temporal PSNR) -> {best}")
                self.selected.add(f"float loop {i}")
                # QAT candidates: the --qat-top best loops
                candidates = [(f"loop {i}", f"qat_loop{i}", ckpt) for i, ckpt, _ in loops[:a.qat_top]]
            if a.float_checkpoint:
                candidates = [("given model", "qat", best)]
            final = best

            # stage 4: QAT of each candidate, keeping the best (simulated int8)
            if a.qat:
                qats = []
                for label, sub, start in candidates:
                    qat = self.train_qat(start, self.dir / sub)
                    if qat is None:
                        self.notes.append(f"QAT of {label}: training failed")
                        continue
                    result = self.test(qat, qat.parent / "test", f"the QAT model of {label}",
                                       still_int8=True)
                    if result is None:
                        self.notes.append(f"QAT of {label}: test failed")
                        continue
                    stage = f"QAT of {label} (sim int8)"
                    self.report.append((stage, result))
                    qats.append((stage, qat, result))
                if not qats:
                    self.log("\nQAT fine-tuning failed; stopping.")
                    self.write_summary(t_start)
                    return 1
                stage, final, _ = max(qats, key=lambda q: _mean(q[2], "temporal", "psnr"))
                if len(qats) > 1:
                    shutil.copy(final, self.dir / "qat_best.pth")
                    final = self.dir / "qat_best.pth"
                    self.notes.append(f"selected {stage.replace(' (sim int8)', '')} "
                                      f"(best mean temporal PSNR, simulated int8) -> {final}")
                    self.selected.add(stage)

        # stage 5: conversion
        failed = False
        for quantize in {"none": [], "fp32": ["fp32"], "int8": ["int8"],
                         "both": ["fp32", "int8"]}[a.convert]:
            result = self.convert(final, quantize)
            if result is None:
                self.notes.append(f"{quantize} conversion failed")
                failed = True
                break
            self.report.append((f"tflite {quantize}", {"_convert": result}))

        self.write_summary(t_start)
        return 1 if failed else 0

    def write_summary(self, t_start: float) -> None:
        lines = [f"Run: {self.dir}",
                 f"Elapsed: {(time.time() - t_start) / 3600:.2f} h",
                 f"Test clips: {', '.join(self.clips)} (gain {self.args.test_gain:g}, seed 0)", ""]
        lines += _report_table(self.report, self.clips, self.selected)
        if self.notes:
            lines += ["", *self.notes]
        text = "\n".join(lines)
        (self.dir / "summary.txt").write_text(text + "\n")
        self.log(f"\n{'#' * 78}\nSUMMARY\n{'#' * 78}\n{text}")


def _mean(result: dict, output: str, key: str) -> float:
    """Mean of result[clip][output][key] over the clips that have it (nan if none)."""
    vals = [r[output][key] for r in result.values() if key in r.get(output, {})]
    return sum(vals) / len(vals) if vals else float("nan")


def _fmt(v: float, width: int = 7, prec: int = 2) -> str:
    return f"{v:{width}.{prec}f}" if v == v else f"{'—':>{width}}"


def _report_table(report: list[tuple[str, dict]], clips: list[str],
                  selected: set[str]) -> list[str]:
    """The concise per-stage PSNR report. Test rows: mean temporal PSNR,
    SSIM, changed-region PSNR, frame 0's PSNR and flicker (processed RGB, as
    test_temporal.py reports them), averaged over the test clips, plus each clip's PSNR when
    there are several. Conversion rows: closed-loop linear PSNR of PyTorch
    and of the .tflite (convert_temporal.py --verify)."""
    per_clip = len(clips) > 1
    def metrics(result: dict, output: str) -> str:
        return (f"{_fmt(_mean(result, output, 'psnr'))} "
                f"{_fmt(_mean(result, output, 'ssim'), prec=4)} "
                f"{_fmt(_mean(result, output, 'changed_psnr'))} "
                f"{_fmt(_mean(result, output, 'frame0_psnr'))} "
                f"{_fmt(_mean(result, output, 'flicker'), 8, 5)}")

    head = (f"{'stage':32s} {'PSNR':>7s} {'SSIM':>7s} {'changed':>7s} {'frame0':>7s} "
            f"{'flicker':>8s}")
    if per_clip:
        head += "   per clip PSNR"
    lines = ["Test results (mean over frames, processed RGB):", head, "-" * len(head)]
    stills = {"still": None, "still_int8": None}
    for stage, result in report:
        if "_convert" in result:
            continue
        if stage in selected:
            stage += " *"
        if not result:
            lines.append(f"{stage:32s} (test failed)")
            continue
        row = f"{stage:32s} {metrics(result, 'temporal')}"
        if per_clip:
            row += "   " + " ".join(_fmt(result[c]["temporal"]["psnr"]).strip() if c in result
                                    else "—" for c in clips)
        lines.append(row)
        for name in stills:
            if stills[name] is None and any(name in r for r in result.values()):
                stills[name] = result
    for name, label in (("still", "(still model, for reference)"),
                        ("still_int8", "(still model, sim int8)")):
        still = stills[name]
        if still is not None:
            lines.append(f"{label:32s} {metrics(still, name)}")
    if selected:
        lines.append("* = selected (best of its stage)")

    converts = [(s, r["_convert"]) for s, r in report if "_convert" in r]
    if converts:
        lines += ["", "Conversion (closed-loop verify, linear PSNR):"]
        for stage, c in converts:
            lines.append(f"  {stage:12s} torch {c['psnr_torch']:.3f} dB  lite {c['psnr_lite']:.3f} dB "
                         f"({c['psnr_lite'] - c['psnr_torch']:+.3f})  on {c['clip']}, "
                         f"{c['frames']} frames -> {c['tflite']}")
    return lines


def resolve_clips(dataset: str, names: list[str]) -> list[str]:
    """Test clip names (folder names, or a unique suffix of one) -> folder names."""
    root = Path(dataset)
    folders = sorted(p.name for p in root.iterdir() if p.is_dir())
    clips = []
    for c in names:
        matches = [c] if c in folders else [n for n in folders if n.endswith(c)]
        if len(matches) != 1:
            sys.exit(f"error: test clip '{c}' matches {len(matches)} clips in {root}"
                     + (f": {', '.join(matches)}" if matches else ""))
        clips.append(matches[0])
    return clips


def _git_state() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True,
                             capture_output=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE,
                               text=True, capture_output=True).stdout.strip()
        return f"{rev}{' (uncommitted changes)' if dirty else ''}"
    except OSError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clips = resolve_clips(args.dataset, args.test_clips)
    if args.name:
        name = args.name
    elif args.qat_checkpoint or args.float_checkpoint:
        name = f"{(args.qat_checkpoint or args.float_checkpoint).stem}_{time.strftime('%Y%m%d-%H%M')}"
    else:
        name = (f"{args.ch}_{args.scaling.replace(' ', '')}e{args.encoders.replace(' ', '')}"
                f"{'w' if args.wide_low else ''}_c{args.cand:g}{f'_gc{args.gate_clamp:g}' if args.gate_clamp else ''}"
                f"_{time.strftime('%Y%m%d-%H%M')}")
    run_dir = (HERE / args.out_root / name) if not args.out_root.is_absolute() else args.out_root / name
    if run_dir.exists() and not args.force:
        sys.exit(f"error: {run_dir} already exists (use --force to write into it, or --name)")
    run_dir.mkdir(parents=True, exist_ok=True)
    return Pipeline(args, run_dir, clips).main()


if __name__ == "__main__":
    sys.exit(main())
