from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("matplotlib is required to run frequency_power_spectrum_compare.py") from exc

from ultralytics import YOLO

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


"""Plot Figure 4: radial power spectra and normalized power differences.

The script compares baseline vs. STS feature maps from the same degraded input:
channel mean -> Hanning window -> 2D-FFT -> radial binning -> log power.
"""

# ---------------------------- User config ---------------------------------
BASELINE_MODEL_PATH = "runs/detect/baseline-l-bs16/weights/best.pt"
STSG_MODEL_PATH = "runs/detect/spike-module-v1-12-12IFv4t2LIF-l-bs16/weights/best.pt"
BASELINE_WEIGHTS_PATH = ""
STSG_WEIGHTS_PATH = ""
IMAGE_PATH = "datasets/snow_storm-331.jpg"
OUTDIR = "frequency_power_output"
RUN_NAME = "snow331_layer3_compare"

LAYER_INDEX = 2  # 0-based index; compare baseline and STSG at the same layer
DEVICE = "0"  # "cpu" or "0"
IMGSZ = 640
CONF = 0.25
IOU = 0.45

FIG_DPI = 600
FIGSIZE = (11.6, 4.8)
EPS = 1e-8
LOW_FREQ_MAX = 0.05
MID_FREQ_MIN = 0.08
MID_FREQ_MAX = 0.25
HIGH_FREQ_MIN = 0.30
FIT_LOW = 0.03
FIT_HIGH = 0.25
RADIAL_BINS = 220
DISPLAY_FREQ_MIN = 0.01
SMOOTH_WINDOW = 7

LINEWIDTH = 2.4
EMPH_LINEWIDTH = 2.8
REF_LINEWIDTH = 2.0
BASELINE_COLOR = "#1f77b4"
STSG_COLOR = "#ff7f0e"
REF_COLOR = "#222222"
DIFF_COLOR = "#d62728"
ZERO_COLOR = "#444444"
# --------------------------------------------------------------------------


@dataclass
class SpectrumResult:
    freqs: np.ndarray
    log_power: np.ndarray
    raw_power: np.ndarray


def load_yolo(model_path: str, weights_path: str):
    yolo = YOLO(model_path)
    if str(model_path).endswith((".yaml", ".yml")) and weights_path:
        yolo.load(weights_path)
    return yolo


def pick_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = pick_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = pick_tensor(item)
            if tensor is not None:
                return tensor
    return None


def pick_feature_map(output):
    tensor = pick_tensor(output)
    if tensor is None:
        return None
    if tensor.ndim == 4:
        return tensor
    if isinstance(output, (list, tuple, dict)):
        candidates = []
        if isinstance(output, dict):
            iterable = output.values()
        else:
            iterable = output
        for item in iterable:
            candidate = pick_feature_map(item)
            if candidate is not None:
                candidates.append(candidate)
        if candidates:
            return candidates[0]
    return None


def load_image(image_path: str, image_size: int) -> np.ndarray:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def extract_feature_map(model_path: str, weights_path: str, image_path: str, layer_index: int) -> np.ndarray:
    yolo = load_yolo(model_path, weights_path)
    modules = yolo.model.model
    if layer_index < 0 or layer_index >= len(modules):
        raise IndexError(f"Layer index {layer_index} out of range, valid range is [0, {len(modules) - 1}]")

    captured = {}

    def hook(_, __, output):
        fmap = pick_feature_map(output)
        if fmap is None:
            raise ValueError(f"Layer {layer_index} output does not contain a 4D tensor")
        captured["tensor"] = fmap.detach()

    handle = modules[layer_index].register_forward_hook(hook)
    try:
        yolo.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False,
            save=False,
        )
    finally:
        handle.remove()

    if "tensor" not in captured:
        raise RuntimeError(f"Failed to capture output from layer {layer_index}")

    tensor = captured["tensor"][0].detach().float().cpu().numpy()
    return tensor


def hanning_window(height: int, width: int) -> np.ndarray:
    wy = np.hanning(height)
    wx = np.hanning(width)
    return np.outer(wy, wx).astype(np.float32)


def radial_power_spectrum(feature_map: np.ndarray, bins: int = RADIAL_BINS) -> SpectrumResult:
    if feature_map.ndim != 3:
        raise ValueError(f"Expected feature map with shape [C, H, W], got {feature_map.shape}")

    fmap = feature_map.mean(axis=0).astype(np.float32)
    window = hanning_window(fmap.shape[0], fmap.shape[1])
    fmap = fmap * window

    spectrum = np.fft.fftshift(np.fft.fft2(fmap))
    power = np.abs(spectrum) ** 2

    h, w = power.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h, d=1.0))
    fx = np.fft.fftshift(np.fft.fftfreq(w, d=1.0))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)

    mask = radius <= 0.5
    radius = radius[mask]
    power = power[mask]

    bin_edges = np.linspace(0.0, 0.5, bins + 1, dtype=np.float32)
    bin_ids = np.digitize(radius, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    radial_sum = np.bincount(bin_ids, weights=power, minlength=bins)
    radial_count = np.bincount(bin_ids, minlength=bins)
    radial_mean = radial_sum / np.maximum(radial_count, 1)
    freqs = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return SpectrumResult(freqs=freqs, log_power=np.log10(radial_mean + EPS), raw_power=radial_mean)


def fit_reference(freqs: np.ndarray, base_power: np.ndarray) -> np.ndarray:
    mask = (freqs >= FIT_LOW) & (freqs <= FIT_HIGH)
    if mask.sum() < 2:
        mask = freqs > 0

    x = np.log(np.maximum(freqs[mask], EPS))
    y = np.log(np.maximum(base_power[mask], EPS))
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -float(slope)
    scale = float(np.exp(intercept))
    return scale / np.maximum(freqs, EPS) ** alpha


def smooth_curve(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def db_difference(base_power: np.ndarray, stsg_power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(stsg_power, EPS) / np.maximum(base_power, EPS))


def annotate_frequency_bands(ax):
    ax.axvspan(0.0, LOW_FREQ_MAX, color="#d9ecff", alpha=0.35, lw=0)
    ax.axvspan(MID_FREQ_MIN, MID_FREQ_MAX, color="#ffe9d6", alpha=0.35, lw=0)
    ax.axvspan(HIGH_FREQ_MIN, 0.5, color="#eaeaea", alpha=0.30, lw=0)
    labels = [
        (0.010, 0.95, "Low band"),
        (0.090, 0.88, "Noise-dominant band"),
        (0.355, 0.81, "Ultra-high band"),
    ]
    for x, y, text in labels:
        ax.text(
            x,
            y,
            text,
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            color="#333333",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
        )


def style_axis(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.42)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.22)
    ax.tick_params(axis="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def plot_figure(freqs: np.ndarray, base: np.ndarray, stsg: np.ndarray, ref: np.ndarray, diff_db: np.ndarray, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=FIG_DPI)

    ax = axes[0]
    ax.plot(freqs, base, color=BASELINE_COLOR, linewidth=LINEWIDTH, label="Baseline", zorder=3)
    ax.plot(freqs, stsg, color=STSG_COLOR, linewidth=EMPH_LINEWIDTH, label="STSG", zorder=4)
    ax.plot(
        freqs,
        np.log10(ref + EPS),
        color=REF_COLOR,
        linewidth=REF_LINEWIDTH,
        linestyle=(0, (5, 3)),
        label=r"$1/f^{2.0}$ reference",
        zorder=2,
    )
    ax.set_xlim(0.0, 0.5)
    ax.set_xlabel(r"$\omega$ (cycle/pixel)")
    ax.set_ylabel(r"$\log P(\omega)$")
    style_axis(ax)
    annotate_frequency_bands(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.02, 0.05, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.plot(freqs, diff_db, color=DIFF_COLOR, linewidth=EMPH_LINEWIDTH, label=r"$10\log_{10}(P_{\mathrm{stsg}}/P_{\mathrm{base}})$", zorder=3)
    ax.axhline(0.0, color=ZERO_COLOR, linewidth=1.5, linestyle="--", zorder=2)
    ax.set_xlim(0.0, 0.5)
    ax.set_xlabel(r"$\omega$ (cycle/pixel)")
    ax.set_ylabel("Power difference (dB)")
    style_axis(ax)
    ax.set_ylim(min(-8.0, float(np.min(diff_db)) - 1.0), max(4.0, float(np.max(diff_db)) + 1.0))
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(0.02, 0.05, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.subplots_adjust(wspace=0.24)
    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    outdir = Path(OUTDIR) / RUN_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    image = load_image(IMAGE_PATH, IMGSZ)
    baseline_fmap = extract_feature_map(BASELINE_MODEL_PATH, BASELINE_WEIGHTS_PATH, IMAGE_PATH, LAYER_INDEX)
    stsg_fmap = extract_feature_map(STSG_MODEL_PATH, STSG_WEIGHTS_PATH, IMAGE_PATH, LAYER_INDEX)

    if baseline_fmap.shape[-2:] != stsg_fmap.shape[-2:]:
        raise ValueError(
            f"Feature map spatial size mismatch: baseline={baseline_fmap.shape[-2:]}, stsg={stsg_fmap.shape[-2:]}"
        )

    base_spec = radial_power_spectrum(baseline_fmap)
    stsg_spec = radial_power_spectrum(stsg_fmap)

    min_len = min(len(base_spec.freqs), len(stsg_spec.freqs))
    freqs = base_spec.freqs[:min_len]
    base_power = base_spec.raw_power[:min_len]
    stsg_power = stsg_spec.raw_power[:min_len]

    display_mask = freqs >= DISPLAY_FREQ_MIN
    freqs = freqs[display_mask]
    base_power = base_power[display_mask]
    stsg_power = stsg_power[display_mask]

    reference = fit_reference(freqs, base_power)
    base_log = np.log10(np.maximum(smooth_curve(base_power), EPS))
    stsg_log = np.log10(np.maximum(smooth_curve(stsg_power), EPS))
    diff_db = smooth_curve(db_difference(base_power, stsg_power))

    fig_path = outdir / "figure4_frequency_compare.png"
    plot_figure(freqs, base_log, stsg_log, reference, diff_db, fig_path)

    summary = [
        f"baseline_model: {BASELINE_MODEL_PATH}",
        f"stsg_model: {STSG_MODEL_PATH}",
        f"image_path: {IMAGE_PATH}",
        f"layer_index: {LAYER_INDEX}",
        f"baseline_feature_map_hw: {baseline_fmap.shape[-2]}x{baseline_fmap.shape[-1]}",
        f"stsg_feature_map_hw: {stsg_fmap.shape[-2]}x{stsg_fmap.shape[-1]}",
        f"display_freq_min: {DISPLAY_FREQ_MIN}",
        f"smooth_window: {SMOOTH_WINDOW}",
        f"radial_bins_after_mask: {len(freqs)}",
        f"output: {fig_path}",
        f"image_shape: {image.shape[0]}x{image.shape[1]}",
    ]
    (outdir / "run_info.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"figure: {fig_path}")
    print(f"run info: {outdir / 'run_info.txt'}")


if __name__ == "__main__":
    main()
