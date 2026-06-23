# denoise-hw

NAFNet-based Bayer denoiser training pipeline.

## Overview

The workflow has three stages:

1. **Build a patch cache** — `dataset.py` scans the raw dataset once, splits
   every image into fixed-size patches, and scores each patch by spatial detail.
   This cache is reused across all training runs.
2. **Train a model** — `train.py` loads the cache and trains a NAFNet-based
   U-Net, saving the best checkpoint to `checkpoints/best.pth`.
3. **Evaluate** — `test.py` runs the trained model on a single image and
   compares it against the hardware denoiser baseline, reporting PSNR and SSIM.

For unattended overnight runs of multiple configurations, use `run.sh`.

---

## Prerequisites

The dataset is expected at `/opt/datasets/PI_RAW` (override with `--dataset`).
Each image folder must contain a paired `noisy.dng` and `gt.dng`.

Install Python dependencies into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 1. Building the dataset cache — `dataset.py`

### Build

```bash
python dataset.py build [--name DIR] [--patch-size N] [--dataset ROOT] [--filter KEYWORD ...]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--name` | `cache` | Output directory for the cache |
| `--patch-size` | `256` | Patch side length in Bayer pixels |
| `--dataset` | `/opt/datasets/PI_RAW` | Root of the PI_RAW dataset tree |
| `--filter` | *(none)* | One or more substrings; only folders whose path contains **all** of them are included |

Example — cache only IMX219 frames:

```bash
python dataset.py build --name cache_imx219 --filter imx219
```

### List

Show a summary of what is stored in a cache:

```bash
python dataset.py list [--name DIR]
```

### Show

Visualise which patches were extracted from a particular image, annotated with
their detail scores:

```bash
python dataset.py show --key PATH/TO/FOLDER [--cache DIR] [--name OUTPUT.jpg]
```

---

## 2. Training — `train.py`

Trains a `DenoiseNet` on the cached patches. The best checkpoint (lowest
validation loss) is saved to `checkpoints/best.pth` automatically.

```bash
python train.py [OPTIONS]
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--cache` | `cache` | Cache directory produced by `dataset.py build` |
| `--filter KEYWORD ...` | *(none)* | Restrict training data to matching folder paths |
| `--detail-top` | `1.0` | Fraction of patches to use, highest-detail first (e.g. `0.5` keeps the sharpest half) |
| `--val-frac` | `0.1` | Fraction of patches held out for validation |
| `--resume WEIGHTS` | *(none)* | Resume from a `.pth` checkpoint; model architecture is inferred from the file |
| `--epochs` | `300` | Maximum number of epochs |
| `--batch-size` | `16` | Training batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--lr-min` | `1e-5` | Training stops when the LR falls below this |
| `--lr-patience` | `5` | Epochs without improvement before LR is halved |
| `--ch` | *(model default)* | Base channel width of the network |
| `--scaling N ...` | *(model default)* | Downscale factor at each encoder level, e.g. `--scaling 2 2 1` |
| `--encoders N ...` | *(model default)* | Number of NAFBlocks per encoder level, e.g. `--encoders 1 2 2` |
| `--no-pbar` | | Suppress per-batch progress bars (useful when logging to a file) |

### Example

```bash
# Train a fresh model
python train.py --ch 16 --scaling 2 2 1 --encoders 1 2 2

# Fine-tune the best checkpoint on a specific subset
python train.py --resume checkpoints/best.pth --filter imx219 ag12 --lr 1e-4 --lr-min 1e-6
```

At the start of each run the script prints the parameter count, MACs/pixel, and
receptive field of the model so you can compare architectures at a glance.

---

## 3. Evaluating a model — `test.py`

Runs a trained model on a single image folder and compares it against the
hardware denoiser (`hw`) and hardware denoiser with CDN chroma filtering
(`hw_cdn`). PSNR and SSIM are printed for each method and PNG outputs are
written to `output/`.

```bash
python test.py FOLDER --weights WEIGHTS.pth [OPTIONS]
```

`FOLDER` is a path relative to `PI_RAW/Data`, e.g.
`cabinet_D50_100/imx219_ag12_test`.

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | *(required)* | Path to a `.pth` checkpoint |
| `--output` | `output` | Directory for PNG outputs |
| `--dataset` | `/opt/datasets/PI_RAW` | Root of the dataset |
| `--quick` | | Skip the noisy/hw/hw_cdn outputs; only produce `net.png` and its metrics |

### Example

```bash
python test.py cabinet_D50_100/imx219_ag12_test --weights checkpoints/best.pth
```

Output files: `noisy.png`, `gt.png`, `hw.png`, `hw_cdn.png`, `net.png`.

---

## 4. Batch training overnight — `run.sh`

`run.sh` is a convenience wrapper for running several model configurations in
sequence, typically left to run overnight. For each configuration it repeats a
two-stage training cycle *N* times:

1. Full train from scratch on the entire cache.
2. Quick test to record baseline metrics.
3. Fine-tune the best checkpoint on `imx219 ag12` data only.
4. Quick test to record fine-tuned metrics.
5. Save the final checkpoint as `checkpoints/run_<tag>_<i>.pth`.

All output is tee'd to `run_<tag>.txt` for later review.

```bash
bash run.sh [N]   # N repetitions per configuration, default 10
```

To add or change configurations, edit the calls to `run_training` at the bottom
of the script:

```bash
run_training <ch> "<scaling factors>" "<encoder counts>"
# e.g.
run_training 16 "2 2 1" "1 2 2"
```

The checkpoint files and log files from previous runs in this repo follow the
naming convention `run_<ch>_<scaling><e><encoders>.txt` /
`checkpoints/run_<tag>_<i>.pth`.
