"""
Выделение объектов на изображении методом оконного (матричного) преобразования.

Конвейер:
  1. Сглаживание (blur)
  2. Повышение резкости (sharpening)
  3. Выделение контуров (edge detection)
  4. Пороговая бинаризация
  5. Морфологическая постобработка -> бинарная маска объектов
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes


def apply_kernel(img_gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Оконное преобразование: свёртка изображения с ядром."""
    h, w = img_gray.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img_gray.astype(np.float64), ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros((h, w), dtype=np.float64)
    for ky in range(kh):
        for kx in range(kw):
            out += kernel[ky, kx] * padded[ky : ky + h, kx : kx + w]
    return out


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def gaussian_kernel(size: int = 7, sigma: float = 1.5) -> np.ndarray:
    k = size // 2
    y, x = np.mgrid[-k : k + 1, -k : k + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


def sharpen_kernel(strength: float = 1.0) -> np.ndarray:
    """
    Ядро повышения резкости (Laplacian sharpening).

    Итог = оригинал + strength * (оригинал - размытое)
        = оригинал * (1 + strength) - strength * размытое

    Рекомендуемый диапазон strength: 0.5 - 3.0
    """
    center = 1.0 + 4.0 * strength
    return np.array(
        [
            [0, -strength, 0],
            [-strength, center, -strength],
            [0, -strength, 0],
        ],
        dtype=np.float64,
    )


KERNELS = {
    "blur_simple": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16,
    "edge_4dir": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64),
    "edge_8dir": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64),
}


def auto_select(gray: np.ndarray) -> dict:
    """Подбор комбинации blur / edge / threshold по контрасту внутри и снаружи маски."""
    print("\n" + "=" * 65)
    print("  АВТОПОДБОР ПАРАМЕТРОВ")
    print("=" * 65)
    print(f"  {'Blur':>12}  {'Edge':>10}  {'Порог':>6}  {'Оценка':>8}")
    print("  " + "-" * 47)

    blur_options = [
        ("gauss_7_1.5", gaussian_kernel(7, 1.5)),
        ("blur_simple", KERNELS["blur_simple"]),
    ]
    edge_options = [
        ("edge_4dir", KERNELS["edge_4dir"]),
        ("sobel_comb", None),
    ]
    thresholds = [20, 35, 50]

    best_score = -1.0
    best_params: dict = {}

    for blur_name, blur_k in blur_options:
        blurred = apply_kernel(gray, blur_k)
        for edge_name, edge_k in edge_options:
            if edge_k is None:
                ex = apply_kernel(blurred, KERNELS["sobel_x"])
                ey = apply_kernel(blurred, KERNELS["sobel_y"])
                edges = np.sqrt(ex**2 + ey**2)
            else:
                edges = np.abs(apply_kernel(blurred, edge_k))
            edges_norm = normalize_to_uint8(edges)

            for thr in thresholds:
                raw_edges_mask = (edges_norm > thr).astype(np.uint8)
                mask = binary_closing(raw_edges_mask, iterations=6).astype(np.uint8)
                mask = binary_fill_holes(mask).astype(np.uint8)
                mask = binary_dilation(mask, iterations=2).astype(np.uint8)

                inside = gray[mask == 1]
                outside = gray[mask == 0]
                score = abs(inside.mean() - outside.mean()) if len(inside) and len(outside) else 0.0

                marker = "  <-- ЛУЧШИЙ" if score > best_score and score > 0 else ""
                print(f"  {blur_name:>12}  {edge_name:>10}  {thr:>6}  {score:>8.2f}{marker}")

                if score > best_score:
                    best_score = score
                    best_params = dict(
                        blur_name=blur_name,
                        blur_k=blur_k,
                        edge_name=edge_name,
                        edge_k=edge_k,
                        threshold=thr,
                        score=score,
                    )

    print("=" * 65)
    if best_params:
        print("\n  ВЫБРАНО:")
        print(f"    Blur   : {best_params['blur_name']}")
        print(f"    Edge   : {best_params['edge_name']}")
        print(f"    Порог  : {best_params['threshold']}")
        print(f"    Оценка : {best_params['score']:.2f}")
        print("\n  Матрица размытия (Blur kernel):")
        print(best_params["blur_k"])
        if best_params["edge_k"] is None:
            print("\n  Матрица границ (Edge kernel): комбинированный Собель")
            print("  sobel_x:")
            print(KERNELS["sobel_x"])
            print("  sobel_y:")
            print(KERNELS["sobel_y"])
        else:
            print("\n  Матрица границ (Edge kernel):")
            print(best_params["edge_k"])
    print("=" * 65 + "\n")
    return best_params


def run_pipeline(img_path: str, sharpen_strength: float = 1.0):
    img = Image.open(img_path)
    if img.width > 500:
        scale = 500 / img.width
        img = img.resize((500, int(img.height * scale)), Image.LANCZOS)

    orig_rgb = img.convert("RGB")
    gray = np.array(img.convert("L"), dtype=np.float64)

    print(f"\n  Файл         : {os.path.basename(img_path)}")
    print(f"  Размер       : {img.width} x {img.height} px")
    print(f"  Резкость     : strength = {sharpen_strength}")

    params = auto_select(gray)
    if not params:
        print("  [!] Автоподбор не дал результата - применяем умолчания.")
        params = dict(
            blur_name="gauss_7_1.5",
            blur_k=gaussian_kernel(7, 1.5),
            edge_name="edge_4dir",
            edge_k=KERNELS["edge_4dir"],
            threshold=35,
            score=0.0,
        )

    blurred = apply_kernel(gray, params["blur_k"])

    if sharpen_strength > 0:
        sharpened = np.clip(apply_kernel(blurred, sharpen_kernel(sharpen_strength)), 0, 255)
    else:
        sharpened = blurred

    if params["edge_k"] is None:
        ex = apply_kernel(sharpened, KERNELS["sobel_x"])
        ey = apply_kernel(sharpened, KERNELS["sobel_y"])
        edges = np.sqrt(ex**2 + ey**2)
    else:
        edges = np.abs(apply_kernel(sharpened, params["edge_k"]))

    edges_norm = normalize_to_uint8(edges)
    edge_mask = (edges_norm > params["threshold"]).astype(np.uint8)

    mask = binary_closing(edge_mask, iterations=6).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask = binary_dilation(mask, iterations=2).astype(np.uint8)

    steps = [
        ("Оригинал (RGB)", np.array(orig_rgb)),
        ("Исходное (gray)", gray),
        ("Размытие (blur)", normalize_to_uint8(blurred)),
        ("Резкость (sharpening)", normalize_to_uint8(sharpened)),
        ("Контуры (производная)", edges_norm),
        ("Бинарные контуры (edge)", edge_mask * 255),
        ("Маска (замкнутые объекты)", mask * 255),
    ]
    return steps, params


def visualize(steps, sharpen_strength: float, save_path: str | None = None, show: bool = True):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    axes_flat = axes.ravel()

    for i, (title, img_arr) in enumerate(steps):
        ax = axes_flat[i]
        ax.set_title(title, fontsize=9)
        if img_arr.ndim == 3:
            ax.imshow(img_arr)
        else:
            ax.imshow(img_arr, cmap="gray")
        ax.axis("off")

    axes_flat[-1].axis("off")
    plt.suptitle(f"Резкость: strength={sharpen_strength}", fontsize=10, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Сохранено: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Выделение объектов на изображении методом оконного преобразования"
    )
    parser.add_argument("image", nargs="?", help="Путь к изображению (jpg/png/...)")
    parser.add_argument(
        "--sharpen",
        type=float,
        default=1.0,
        help="Сила резкости (0 = отключено, по умолчанию 1.0)",
    )
    parser.add_argument("--save", metavar="PATH", help="Сохранить визуализацию в файл")
    parser.add_argument("--no-show", action="store_true", help="Не открывать окно matplotlib")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image:
        img_path = args.image
    else:
        print("\n" + "=" * 50)
        print("  МАШИННОЕ ЗРЕНИЕ - выделение объектов")
        print("=" * 50)
        img_path = input("  Введите имя файла (например, input.jpg): ").strip()

    if not os.path.isfile(img_path):
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_path)
        if os.path.isfile(candidate):
            img_path = candidate
        else:
            print(f"\n  [ОШИБКА] Файл не найден: {img_path}")
            sys.exit(1)

    steps, _params = run_pipeline(img_path, sharpen_strength=args.sharpen)
    visualize(steps, args.sharpen, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
