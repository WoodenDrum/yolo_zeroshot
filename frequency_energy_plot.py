from pathlib import Path

import cv2
import numpy as np

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError("matplotlib is required to run frequency_energy_plot.py") from exc

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

"""Plot radial frequency-energy curves for multiple images in one figure."""

# ---------------------------- User config ---------------------------------
CLEAN_IMAGE = "datasets/2008_000028.jpg"
FOG_IMAGE = "datasets/AM_Google_309.png"
SNOW_IMAGE = "datasets/snow_storm-219.jpg"
DARK_IMAGE = "datasets/2015_00028.jpg"
OUTDIR = "frequency_energy_output"
RUN_NAME = "sample_case_01"
OUTPUT_NAME = "radial_frequency_energy.png"
IMAGE_SIZE = 400
FIG_DPI = 600
EPS = 1e-8
FIT_LOW = 0.03
FIT_HIGH = 0.35
DISPLAY_FREQ_MIN = 0.0000001
# --------------------------------------------------------------------------


def load_grayscale_image(image_path: str, image_size: int) -> np.ndarray:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def compute_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    return np.abs(spectrum) ** 2


def radial_profile(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = spectrum.shape
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    yy, xx = np.indices((height, width))
    radii = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    radii = radii.astype(np.int32)

    radial_sum = np.bincount(radii.ravel(), weights=spectrum.ravel())
    radial_count = np.bincount(radii.ravel())
    radial_mean = radial_sum / np.maximum(radial_count, 1)

    max_valid_radius = int(np.ceil(radii.max())) + 1
    radial_mean = radial_mean[:max_valid_radius]

    freqs = np.arange(len(radial_mean), dtype=np.float32)
    freqs /= max(len(radial_mean) - 1, 1)

    return freqs, radial_mean


def fit_reference_curve(freqs: np.ndarray, clean_energy: np.ndarray, fit_low: float = FIT_LOW, fit_high: float = FIT_HIGH) -> tuple[np.ndarray, float]:
    fit_mask = (freqs >= fit_low) & (freqs <= fit_high)
    if fit_mask.sum() < 2:
        fit_mask = freqs > 0

    x = np.log(np.maximum(freqs[fit_mask], EPS))
    y = np.log(np.maximum(clean_energy[fit_mask], EPS))
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -float(slope)
    scale = float(np.exp(intercept))
    reference = scale / np.maximum(freqs, EPS) ** alpha
    return reference, alpha


def plot_curves(
    clean_freqs: np.ndarray,
    clean_energy: np.ndarray,
    fog_energy: np.ndarray,
    snow_energy: np.ndarray,
    dark_energy: np.ndarray,
    ref_energy: np.ndarray,
    save_path: Path,
) -> None:
    display_mask = clean_freqs >= DISPLAY_FREQ_MIN
    plot_freqs = clean_freqs[display_mask]
    clean_curve = clean_energy[display_mask]
    fog_curve = fog_energy[display_mask]
    snow_curve = snow_energy[display_mask]
    dark_curve = dark_energy[display_mask]
    ref_curve = ref_energy[display_mask]

    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(plot_freqs, clean_curve, color="#1f77b4", linewidth=2.5, label="Original image", zorder=3)
    ax.plot(plot_freqs, fog_curve, color="#ff7f0e", linewidth=2.5, label="Foggy image", zorder=4)
    ax.plot(plot_freqs, snow_curve, color="#d62728", linewidth=2.5, label="Snowy image", zorder=4)
    ax.plot(plot_freqs, dark_curve, color="#9467bd", linewidth=2.5, linestyle="--", label="Dark image", zorder=4)
    ax.plot(
        plot_freqs,
        ref_curve,
        color="#222222",
        linewidth=2.0,
        linestyle=(0, (5, 3)),
        label=rf"$1/f^{{\alpha}}$ reference",
        zorder=2,
    )

    ax.set_xlim(plot_freqs[0], plot_freqs[-1])
    y_min = max(min(clean_curve.min(), fog_curve.min(), snow_curve.min(), dark_curve.min(), ref_curve.min()) * 0.75, 1e-12)
    y_max = max(clean_curve.max(), fog_curve.max(), snow_curve.max(), dark_curve.max(), ref_curve.max()) * 1.18
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Normalized radial frequency")
    ax.set_ylabel("Radially averaged power")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.42)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.22)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_input_copy(image_path: str, image_size: int, save_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(save_path), image):
        raise ValueError(f"Failed to save image copy: {save_path}")


def build_copy_name(prefix: str, image_path: str) -> str:
    return f"{prefix}_{Path(image_path).name}"


def main():
    run_dir = Path(OUTDIR) / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_image = load_grayscale_image(CLEAN_IMAGE, IMAGE_SIZE)
    fog_image = load_grayscale_image(FOG_IMAGE, IMAGE_SIZE)
    snow_image = load_grayscale_image(SNOW_IMAGE, IMAGE_SIZE)
    dark_image = load_grayscale_image(DARK_IMAGE, IMAGE_SIZE)

    clean_spectrum = compute_magnitude_spectrum(clean_image)
    fog_spectrum = compute_magnitude_spectrum(fog_image)
    snow_spectrum = compute_magnitude_spectrum(snow_image)
    dark_spectrum = compute_magnitude_spectrum(dark_image)

    clean_freqs, clean_energy = radial_profile(clean_spectrum)
    fog_freqs, fog_energy = radial_profile(fog_spectrum)
    snow_freqs, snow_energy = radial_profile(snow_spectrum)
    dark_freqs, dark_energy = radial_profile(dark_spectrum)

    min_length = min(len(clean_freqs), len(fog_freqs), len(snow_freqs), len(dark_freqs))
    clean_freqs = clean_freqs[:min_length]
    clean_energy = clean_energy[:min_length]
    fog_energy = fog_energy[:min_length]
    snow_energy = snow_energy[:min_length]
    dark_energy = dark_energy[:min_length]

    ref_energy, fitted_alpha = fit_reference_curve(clean_freqs, clean_energy)

    save_path = run_dir / OUTPUT_NAME
    plot_curves(clean_freqs, clean_energy, fog_energy, snow_energy, dark_energy, ref_energy, save_path)

    for prefix, image_path in [
        ("clean", CLEAN_IMAGE),
        ("fog", FOG_IMAGE),
        ("snow", SNOW_IMAGE),
        ("dark", DARK_IMAGE),
    ]:
        save_input_copy(image_path, IMAGE_SIZE, run_dir / build_copy_name(prefix, image_path))

    print(f"Saved plot to: {save_path}")
    print(f"Fitted alpha: {fitted_alpha:.3f}")


if __name__ == "__main__":
    main()
