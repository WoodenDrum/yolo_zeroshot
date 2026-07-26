from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from ultralytics import YOLO


"""Feature heatmap and detection visualization for a single image.

Edit the configuration below and run this file directly.
"""

# ---------------------------- User config ---------------------------------
MODEL_PATH = "runs/detect/baseline-l-bs16/weights/best.pt"  # .yaml or .pt
MODEL_PATH  = "runs/detect/spike-module-v1-12-12IFv4t2-l-bs16/weights/best.pt"  # .yaml or .pt
WEIGHTS_PATH = ""  # Only used when MODEL_PATH is a .yaml file
# IMAGE_PATH = "/disk/duanww/DAWN/Snow/images/snow_storm-010.jpg"
# IMAGE_PATH = "/disk/duanww/DAWN/Fog/images/foggy-085.jpg"
IMAGE_PATH = "datasets/snow_storm-331.jpg"
LAYER_INDEX = 3
DEVICE = "0"  # "cpu" or "0"
IMGSZ = 640
CONF = 0.25
IOU = 0.45
OUTDIR = "heatmap_output/sts/snow331/3"

REDUCE = "mean_abs"  # "mean_abs", "max_abs", "l2"
GATE_REDUCE = "mean_abs"  # "mean_abs", "max_abs", "l2"
GATE_CHANNEL_INDEX = None  # None saves all channels; set to an int to save one channel only
GATE_CHANNEL_LIMIT = None  # Optional int cap when saving all channels
GATE_OVERLAY_ALPHA = 0.32
GATE_CHANNEL_GAMMA = 1.25
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


def reduce_feature_map(feature_map: torch.Tensor, mode: str):
    if feature_map.ndim != 4:
        raise ValueError(f"Expected a 4D feature map, but got shape {tuple(feature_map.shape)}")

    fmap = feature_map[0].detach().float().cpu()
    if mode == "mean_abs":
        heatmap = fmap.abs().mean(dim=0)
    elif mode == "max_abs":
        heatmap = fmap.abs().max(dim=0).values
    else:
        heatmap = torch.sqrt((fmap ** 2).sum(dim=0))
    return heatmap.numpy()


def normalize_heatmap(heatmap: np.ndarray):
    heatmap = heatmap.astype(np.float32)
    heatmap -= heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap /= max_value
    return heatmap


def normalize_with_percentile(data: np.ndarray, low: float = 1.0, high: float = 99.5):
    data = data.astype(np.float32)
    lower = np.percentile(data, low)
    upper = np.percentile(data, high)
    if upper <= lower:
        return normalize_heatmap(data)
    data = np.clip(data, lower, upper)
    data -= data.min()
    data /= max(data.max(), 1e-6)
    return data


def feature_frequency_map(feature_map: torch.Tensor, mode: str):
    if feature_map.ndim != 4:
        raise ValueError(f"Expected a 4D feature map, but got shape {tuple(feature_map.shape)}")

    fmap = feature_map[0].detach().float().cpu().numpy()
    spectra = np.fft.fftshift(np.fft.fft2(fmap, axes=(-2, -1)), axes=(-2, -1))
    magnitude = np.log1p(np.abs(spectra))
    if mode == "max_abs":
        magnitude = magnitude.max(axis=0)
    else:
        magnitude = magnitude.mean(axis=0)
    return normalize_heatmap(magnitude)


def save_heatmap(heatmap: np.ndarray, image_shape, save_path: Path):
    height, width = image_shape[:2]
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(str(save_path), heatmap_color)


def blend_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    height, width = image.shape[:2]
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1.0 - alpha, heatmap_color, alpha, 0)


def save_frequency_map(freq_map: np.ndarray, image_shape, save_path: Path):
    height, width = image_shape[:2]
    freq_map = cv2.resize(freq_map, (width, height), interpolation=cv2.INTER_CUBIC)
    freq_u8 = np.clip(freq_map * 255.0, 0, 255).astype(np.uint8)
    canvas = cv2.applyColorMap(freq_u8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(save_path), canvas)


def save_gate_map(gate_map: np.ndarray, image_shape, save_path: Path):
    height, width = image_shape[:2]
    gate_map = cv2.resize(gate_map, (width, height), interpolation=cv2.INTER_LINEAR)
    gate_map = normalize_with_percentile(gate_map, low=2.0, high=99.0)
    gate_map = np.power(gate_map, 0.7)
    gate_u8 = np.clip(gate_map * 255.0, 0, 255).astype(np.uint8)
    canvas = cv2.applyColorMap(gate_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(save_path), canvas)


def gate_channel_to_color(channel_map: np.ndarray):
    channel_map = normalize_with_percentile(channel_map, low=2.0, high=98.5)
    channel_map = np.power(channel_map, GATE_CHANNEL_GAMMA)
    channel_u8 = np.clip(channel_map * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(channel_u8, cv2.COLORMAP_VIRIDIS)


def save_gate_channel_map(channel_map: np.ndarray, save_path: Path):
    cv2.imwrite(str(save_path), gate_channel_to_color(channel_map))


def save_gate_channel_overlay(image: np.ndarray, channel_map: np.ndarray, save_path: Path):
    height, width = image.shape[:2]
    color = gate_channel_to_color(cv2.resize(channel_map, (width, height), interpolation=cv2.INTER_LINEAR))
    overlay = cv2.addWeighted(image, 1.0 - GATE_OVERLAY_ALPHA, color, GATE_OVERLAY_ALPHA, 0)
    cv2.imwrite(str(save_path), overlay)


def save_gate_channels(gate_tensor: torch.Tensor, image: np.ndarray, outdir: Path, stem: str, layer_index: int):
    if gate_tensor.ndim != 4:
        raise ValueError(f"Expected a 4D gate tensor, but got shape {tuple(gate_tensor.shape)}")

    gate = gate_tensor[0].detach().float().cpu().numpy()
    channel_dir = outdir / f"{stem}_layer{layer_index}_gate_channels"
    channel_dir.mkdir(parents=True, exist_ok=True)

    channel_scores = []
    num_channels = gate.shape[0]
    indices = [GATE_CHANNEL_INDEX] if GATE_CHANNEL_INDEX is not None else list(range(num_channels))
    if GATE_CHANNEL_LIMIT is not None:
        indices = indices[:GATE_CHANNEL_LIMIT]

    for idx in indices:
        if idx < 0 or idx >= num_channels:
            continue
        channel_map = gate[idx]
        save_gate_channel_map(channel_map, channel_dir / f"ch{idx:03d}.png")
        save_gate_channel_overlay(image, channel_map, channel_dir / f"ch{idx:03d}_overlay.png")
        channel_scores.append((idx, float(channel_map.mean()), float(channel_map.max()), float(channel_map.min())))

    summary_path = channel_dir / "summary.txt"
    lines = ["index mean max min"]
    for idx, mean_v, max_v, min_v in channel_scores:
        lines.append(f"{idx} {mean_v:.6f} {max_v:.6f} {min_v:.6f}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return channel_dir


def save_detections(result, image_shape, save_path: Path, txt_path: Path):
    plotted = result.plot()
    cv2.imwrite(str(save_path), plotted)

    height, width = image_shape[:2]
    lines = []
    if result.boxes is not None:
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy() if len(boxes) else np.empty((0, 4), dtype=np.float32)
        confs = boxes.conf.detach().cpu().numpy() if len(boxes) else np.empty((0,), dtype=np.float32)
        clss = boxes.cls.detach().cpu().numpy() if len(boxes) else np.empty((0,), dtype=np.float32)
        for box, conf, cls in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = box.tolist()
            lines.append(
                f"{int(cls)} {conf:.6f} "
                f"{x1 / width:.6f} {y1 / height:.6f} {x2 / width:.6f} {y2 / height:.6f}"
            )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


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

    captured = {}
    captured_gate = {}

    def hook(_, __, output):
        tensor = pick_tensor(output)
        if tensor is None:
            raise ValueError(f"Layer {LAYER_INDEX} output does not contain a tensor")
        captured["tensor"] = tensor.detach()

    handle = modules[LAYER_INDEX].register_forward_hook(hook)
    gate_handle = None
    if hasattr(modules[LAYER_INDEX], "readout") and isinstance(modules[LAYER_INDEX].readout, torch.nn.Module):
        def gate_hook(_, __, output):
            tensor = pick_tensor(output)
            if tensor is None:
                raise ValueError(f"Layer {LAYER_INDEX} readout output does not contain a tensor")
            captured_gate["tensor"] = tensor.detach()

        gate_handle = modules[LAYER_INDEX].readout.register_forward_hook(gate_hook)
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
        if gate_handle is not None:
            gate_handle.remove()

    if "tensor" not in captured:
        raise RuntimeError(f"Failed to capture output from layer {LAYER_INDEX}")

    result = results[0]
    heatmap = normalize_heatmap(reduce_feature_map(captured["tensor"], REDUCE))
    frequency = feature_frequency_map(captured["tensor"], REDUCE)
    gate_map = None
    if "tensor" in captured_gate:
        gate_map = captured_gate["tensor"]

    stem = image_path.stem
    save_heatmap(heatmap, image.shape, outdir / f"{stem}_layer{LAYER_INDEX}_heatmap.png")
    save_frequency_map(frequency, image.shape, outdir / f"{stem}_layer{LAYER_INDEX}_frequency.png")
    save_detections(result, image.shape, outdir / f"{stem}_detections.png", outdir / f"{stem}_detections.txt")
    combined = result.plot(img=blend_heatmap(image.copy(), heatmap))
    cv2.imwrite(str(outdir / f"{stem}_layer{LAYER_INDEX}_heatmap_boxes.png"), combined)
    if gate_map is not None:
        gate_channel_dir = save_gate_channels(gate_map, image, outdir, stem, LAYER_INDEX)

    print(f"layer: {LAYER_INDEX}")
    print(f"module: {modules[LAYER_INDEX].type}")
    print(f"heatmap: {outdir / f'{stem}_layer{LAYER_INDEX}_heatmap.png'}")
    print(f"frequency: {outdir / f'{stem}_layer{LAYER_INDEX}_frequency.png'}")
    if gate_map is not None:
        print(f"gate channels: {gate_channel_dir}")
    print(f"detections image: {outdir / f'{stem}_detections.png'}")
    print(f"combined: {outdir / f'{stem}_layer{LAYER_INDEX}_heatmap_boxes.png'}")
    print(f"detections txt: {outdir / f'{stem}_detections.txt'}")


if __name__ == "__main__":
    main()
