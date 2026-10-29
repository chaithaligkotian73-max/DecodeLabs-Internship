"""
Generate the sample images used by the demo (optional - they are already included).

    python make_sample_images.py

Creates in sample_images/:
    sample_clean.png  - clean black text on a white background
    sample_noisy.png  - same text with uneven lighting, noise and a slight tilt
                        (this is what the pre-processing stage is designed to fix)
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

LINES = [
    "DECODELABS AI INTERNSHIP",
    "Project 4: Text Recognition",
    "Invoice No: 20260429",
    "Date: 29 October 2026",
    "Item: Industrial Training Kit",
    "Total Amount: $499.00",
]

FONT_CANDIDATES = [
    "arial.ttf", "Arial.ttf", "calibri.ttf",                       # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",              # Linux
    "/System/Library/Fonts/Supplemental/Arial.ttf",                 # macOS
]


def get_font(size):
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_clean(width=1000, height=520):
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = get_font(44)
    y = 40
    for line in LINES:
        draw.text((50, y), line, fill="black", font=font)
        y += 75
    return np.array(img)[:, :, ::-1]          # RGB -> BGR for OpenCV


def make_noisy(clean_bgr, angle=3.0, seed=7):
    rng = np.random.default_rng(seed)
    h, w = clean_bgr.shape[:2]
    img = clean_bgr.astype(np.float32)

    # uneven lighting: dark gradient shadow across the page
    shadow = np.tile(np.linspace(0.55, 1.0, w, dtype=np.float32), (h, 1))
    img *= shadow[:, :, None]

    # sensor noise
    img += rng.normal(0, 14, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # slight rotation (tilted scan)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=(200, 200, 200))


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    clean = make_clean()
    cv2.imwrite(os.path.join(OUT_DIR, "sample_clean.png"), clean)
    cv2.imwrite(os.path.join(OUT_DIR, "sample_noisy.png"), make_noisy(clean))
    print("Saved sample_clean.png and sample_noisy.png in", OUT_DIR)
