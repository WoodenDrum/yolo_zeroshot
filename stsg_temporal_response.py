from __future__ import annotations

from pathlib import Path
from types import MethodType

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


"""STS pseudo-temporal response visualization for a single image.

This script captures the internal 5-step membrane potential H_t and spike output
S_t of the PLIF node inside an STS-like module, then plots temporal curves for
three manually selected spatial regions:
1. target edge
2. flat background
3. isolated noisy high-response area
"""

# ---------------------------- User config ---------------------------------
MODEL_PATH = "runs/detect/spike-module-v1-12-12IFv4t2LIF-l-bs16/weights/best.pt"
WEIGHTS_PATH = ""
IMAGE_PATH = "datasets/snow_storm-331.jpg"
OUTDIR = "temporal_response_output/sts/snow331_layer3"

DEVICE = "0"  # "cpu" or "0"
IMGSZ = 640
CONF = 0.25
IOU = 0.45

LAYER_INDEX = 3
SHIFTS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))
NEIGHBORHOOD_RADIUS = 1  # 1 -> 3x3 mean

# Coordinates are on the hooked feature map, not the input image.
# If AUTO_REGION_SELECTION is False, these manual coordinates will be used.
REGIONS = {
    "edge": (40, 52),
    "background": (100, 20),
    "noise": (10, 35),
}

# Optional manual override. If None, edge/background/noise are inferred automatically.
AUTO_REGION_SELECTION = True

PLOT_THRESHOLD = True
SAVE_STEP_MAPS = True
# --------------------------------------------------------------------------


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


def shift_feature(x: torch.Tensor, dx: int = 0, dy: int = 0, mode: str = "reflect") -> torch.Tensor:
    if dx == 0 and dy == 0:
        return x

    _, _, h, w = x.shape
    pad_l = max(dx, 0)
    pad_r = max(-dx, 0)
    pad_t = max(dy, 0)
    pad_b = max(-dy, 0)
    x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    return x_pad[:, :, pad_b : pad_b + h, pad_r : pad_r + w]


def build_temporal_sequence(module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(module, "build_temporal_sequence"):
        return module.build_temporal_sequence(x)
    if hasattr(module, "build_shift_sequence"):
        return module.build_shift_sequence(x)
    shifts = getattr(module, "shifts", SHIFTS)
    padding_mode = getattr(module, "padding_mode", "reflect")
    seq = [shift_feature(x, dx=dx, dy=dy, mode=padding_mode) for dx, dy in shifts]
    return torch.stack(seq, dim=0)


def get_tau_value(plif) -> float:
    if hasattr(plif, "tau") and not callable(plif.tau):
        tau = plif.tau
        if isinstance(tau, torch.Tensor):
            return float(tau.detach().float().mean().cpu())
        return float(tau)

    if hasattr(plif, "w"):
        w = plif.w
        if isinstance(w, torch.Tensor):
            w = w.detach().float().mean().cpu()
        return float(1.0 / torch.sigmoid(torch.as_tensor(w)).item())

    if hasattr(plif, "init_tau"):
        return float(plif.init_tau)

    return 2.0


def get_v_threshold(plif) -> float:
    v_threshold = getattr(plif, "v_threshold", 1.0)
    if isinstance(v_threshold, torch.Tensor):
        return float(v_threshold.detach().float().mean().cpu())
    return float(v_threshold)


def get_v_reset(plif) -> float:
    v_reset = getattr(plif, "v_reset", 0.0)
    if v_reset is None:
        return 0.0
    if isinstance(v_reset, torch.Tensor):
        return float(v_reset.detach().float().mean().cpu())
    return float(v_reset)


def simulate_plif_membrane(plif, x_seq: torch.Tensor, s_seq: torch.Tensor) -> torch.Tensor:
    tau = max(get_tau_value(plif), 1.0 + 1e-6)
    v_threshold = get_v_threshold(plif)
    v_reset = get_v_reset(plif)
    decay_input = bool(getattr(plif, "decay_input", True))

    mem = torch.zeros_like(x_seq[0])
    mem_seq = []
    alpha = 1.0 - 1.0 / tau

    for t in range(x_seq.shape[0]):
        x_t = x_seq[t]
        if decay_input:
            mem = alpha * mem + x_t / tau
        else:
            mem = alpha * mem + x_t

        mem_seq.append(mem.clone())

        spike_t = (s_seq[t] > 0).to(mem.dtype)
        if v_reset == 0.0:
            mem = mem - spike_t * v_threshold
        else:
            mem = mem * (1.0 - spike_t) + v_reset * spike_t

    return torch.stack(mem_seq, dim=0)


def reduce_step_map(seq: torch.Tensor, mode: str = "mean_abs") -> np.ndarray:
    seq = seq.detach().float().cpu()
    if mode == "mean_abs":
        step_map = seq.abs().mean(dim=2)
    elif mode == "max_abs":
        step_map = seq.abs().max(dim=2).values
    else:
        step_map = torch.sqrt((seq ** 2).sum(dim=2))
    return step_map.numpy()


def normalize_map(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    data -= data.min()
    vmax = data.max()
    if vmax > 0:
        data /= vmax
    return data


def save_step_maps(step_maps: np.ndarray, base_image: np.ndarray, outdir: Path, stem: str, prefix: str):
    h, w = base_image.shape[:2]
    for step_idx in range(step_maps.shape[0]):
        step_map = normalize_map(step_maps[step_idx, 0])
        step_map = cv2.resize(step_map, (w, h), interpolation=cv2.INTER_LINEAR)
        heat_u8 = np.clip(step_map * 255.0, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(base_image, 0.62, color, 0.38, 0)
        cv2.imwrite(str(outdir / f"{stem}_{prefix}_t{step_idx + 1}.png"), color)
        cv2.imwrite(str(outdir / f"{stem}_{prefix}_t{step_idx + 1}_overlay.png"), overlay)


def clip_patch_bounds(y: int, x: int, h: int, w: int, radius: int):
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    return y0, y1, x0, x1


def sample_region_series(seq: torch.Tensor, region_map: dict[str, tuple[int, int]], radius: int) -> dict[str, np.ndarray]:
    seq = seq.detach().float().cpu()
    _, _, _, h, w = seq.shape
    out = {}
    for name, (x, y) in region_map.items():
        y0, y1, x0, x1 = clip_patch_bounds(y, x, h, w, radius)
        # After selecting batch index 0, patch shape is [T, C, patch_h, patch_w].
        patch = seq[:, 0, :, y0:y1, x0:x1]
        out[name] = patch.mean(dim=(1, 2, 3)).numpy()
    return out


def save_region_overlay(image: np.ndarray, fmap_hw: tuple[int, int], region_map: dict[str, tuple[int, int]], out_path: Path):
    canvas = image.copy()
    img_h, img_w = image.shape[:2]
    feat_h, feat_w = fmap_hw
    color = (0, 0, 0)

    for name, (x, y) in region_map.items():
        px = int(round((x + 0.5) * img_w / feat_w))
        py = int(round((y + 0.5) * img_h / feat_h))
        cv2.drawMarker(canvas, (px, py), color, markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.putText(canvas, name, (px + 8, py - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


def image_to_feature_coords(xyxy: np.ndarray, img_shape: tuple[int, int], fmap_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    img_h, img_w = img_shape[:2]
    fmap_h, fmap_w = fmap_hw
    x1, y1, x2, y2 = xyxy.tolist()
    fx1 = int(np.clip(round(x1 / img_w * fmap_w), 0, fmap_w - 1))
    fy1 = int(np.clip(round(y1 / img_h * fmap_h), 0, fmap_h - 1))
    fx2 = int(np.clip(round(x2 / img_w * fmap_w), 0, fmap_w - 1))
    fy2 = int(np.clip(round(y2 / img_h * fmap_h), 0, fmap_h - 1))
    return fx1, fy1, fx2, fy2


def select_regions_from_boxes(result, image: np.ndarray, fmap_hw: tuple[int, int]) -> dict[str, tuple[int, int]]:
    fmap_h, fmap_w = fmap_hw
    if result.boxes is None or len(result.boxes) == 0:
        return REGIONS.copy()

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = image_to_feature_coords(boxes[best_idx], image.shape, fmap_hw)

    edge_x = int(np.clip(round((x1 + x2) / 2), 0, fmap_w - 1))
    edge_y = int(np.clip(round(y1), 0, fmap_h - 1))

    return {
        "edge": (edge_x, edge_y),
        "background": REGIONS["background"],
        "noise": REGIONS["noise"],
    }


def plot_series(mem_series: dict[str, np.ndarray], spike_series: dict[str, np.ndarray], threshold: float, out_path: Path):
    steps = np.arange(1, len(next(iter(mem_series.values()))) + 1)
    colors = {
        "edge": "#d94841",
        "background": "#2b8a3e",
        "noise": "#1c7ed6",
    }

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True)

    for name, values in mem_series.items():
        axes[0].plot(steps, values, marker="o", linewidth=2.0, color=colors.get(name), label=name)
    if PLOT_THRESHOLD:
        axes[0].axhline(threshold, color="#444444", linestyle="--", linewidth=1.5, label="V_th")
    axes[0].set_ylabel(r"Membrane potential $H_t$")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False, ncol=4)

    for name, values in spike_series.items():
        axes[1].plot(steps, values, marker="o", linewidth=2.0, color=colors.get(name), label=name)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks([0.0, 1.0])
    axes[1].set_ylabel(r"Spike output $S_t$")
    axes[1].set_xlabel(r"Pseudo-time step $t$")
    axes[1].set_xticks(steps)
    axes[1].grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_series_csv(mem_series: dict[str, np.ndarray], spike_series: dict[str, np.ndarray], out_path: Path):
    steps = np.arange(1, len(next(iter(mem_series.values()))) + 1)
    header = ["step"]
    for name in mem_series:
        header.extend([f"{name}_H", f"{name}_S"])

    lines = [",".join(header)]
    for i, step in enumerate(steps):
        row = [str(int(step))]
        for name in mem_series:
            row.append(f"{mem_series[name][i]:.6f}")
            row.append(f"{spike_series[name][i]:.6f}")
        lines.append(",".join(row))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    image_path = Path(IMAGE_PATH)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    yolo = load_yolo(MODEL_PATH, WEIGHTS_PATH)
    modules = yolo.model.model
    if LAYER_INDEX < 0 or LAYER_INDEX >= len(modules):
        raise IndexError(f"Layer index {LAYER_INDEX} out of range, valid range is [0, {len(modules) - 1}]")

    target_module = modules[LAYER_INDEX]
    if not hasattr(target_module, "plif"):
        raise AttributeError(f"Layer {LAYER_INDEX} has no internal 'plif' module")

    original_shifts = getattr(target_module, "shifts", None)
    original_T = getattr(target_module, "T", None)
    if hasattr(target_module, "shifts"):
        target_module.shifts = SHIFTS
    if hasattr(target_module, "T"):
        target_module.T = len(SHIFTS)

    captured = {}

    def plif_forward_wrapper(plif_self, x_seq: torch.Tensor):
        captured["plif_input"] = x_seq.detach()
        out = captured["original_plif_forward"](x_seq)
        captured["spike_seq"] = out.detach()
        return out

    original_plif_forward = target_module.plif.forward
    captured["original_plif_forward"] = original_plif_forward
    target_module.plif.forward = MethodType(plif_forward_wrapper, target_module.plif)

    def module_hook(_, __, output):
        tensor = pick_tensor(output)
        if tensor is not None:
            captured["layer_output"] = tensor.detach()

    handle = target_module.register_forward_hook(module_hook)
    try:
        results = yolo.predict(
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
        target_module.plif.forward = original_plif_forward
        if hasattr(target_module, "shifts"):
            target_module.shifts = original_shifts
        if hasattr(target_module, "T"):
            target_module.T = original_T

    stem = image_path.stem
    if "plif_input" not in captured or "spike_seq" not in captured:
        raise RuntimeError("Failed to capture PLIF input/spike sequence")

    x_seq = captured["plif_input"]
    s_seq = captured["spike_seq"]
    h_seq = simulate_plif_membrane(target_module.plif, x_seq, s_seq)

    result = results[0]
    region_map = select_regions_from_boxes(result, image, (h_seq.shape[-2], h_seq.shape[-1])) if AUTO_REGION_SELECTION else REGIONS.copy()

    mem_series = sample_region_series(h_seq, region_map, NEIGHBORHOOD_RADIUS)
    spike_series = sample_region_series(s_seq, region_map, NEIGHBORHOOD_RADIUS)
    threshold = get_v_threshold(target_module.plif)

    save_series_csv(mem_series, spike_series, outdir / f"{stem}_layer{LAYER_INDEX}_temporal_series.csv")
    plot_series(mem_series, spike_series, threshold, outdir / f"{stem}_layer{LAYER_INDEX}_temporal_response.png")
    save_region_overlay(image, (h_seq.shape[-2], h_seq.shape[-1]), region_map, outdir / f"{stem}_region_marks.png")

    if SAVE_STEP_MAPS:
        save_step_maps(reduce_step_map(h_seq, mode="mean_abs"), image, outdir, stem, "membrane")
        save_step_maps(reduce_step_map(s_seq, mode="mean_abs"), image, outdir, stem, "spike")

    plotted = result.plot()
    cv2.imwrite(str(outdir / f"{stem}_detections.png"), plotted)

    info_lines = [
        f"layer_index: {LAYER_INDEX}",
        f"module_type: {getattr(target_module, 'type', type(target_module).__name__)}",
        f"feature_map_hw: {h_seq.shape[-2]}x{h_seq.shape[-1]}",
        f"steps: {h_seq.shape[0]}",
        f"v_threshold: {threshold:.6f}",
        f"tau: {get_tau_value(target_module.plif):.6f}",
        f"regions: {region_map}",
        f"shifts: {SHIFTS}",
    ]
    (outdir / "run_info.txt").write_text("\n".join(info_lines), encoding="utf-8")

    print(f"temporal response figure: {outdir / f'{stem}_layer{LAYER_INDEX}_temporal_response.png'}")
    print(f"series csv: {outdir / f'{stem}_layer{LAYER_INDEX}_temporal_series.csv'}")
    print(f"region overlay: {outdir / f'{stem}_region_marks.png'}")
    print(f"run info: {outdir / 'run_info.txt'}")


if __name__ == "__main__":
    main()
