"""
Image / Text Recognition (Basic) - OCR Pipeline
DecodeLabs AI Internship - Project 4  (Path 1: Optical Character Recognition)
============================================================================
Uses a pre-trained OCR engine (Google's Tesseract via pytesseract) plus OpenCV
pre-processing. No model is trained - this is "transfer learning" in practice:
we plug a ready-made recognition engine into our own workflow.

IPO model
    INPUT   -> load an image (an image is a 3D array: height x width x 3 channels)
    PROCESS -> grayscale -> Gaussian blur -> adaptive threshold -> deskew -> OCR
    OUTPUT  -> text + per-word confidence, 80% confidence gate, annotated image

Gatekeeper validations (printed at the end of every run)
    1. Library integration   2. Pre-processing integrity
    3. Accuracy >= 80%       4. Visual confirmation

Run
    python ocr_pipeline.py
    python ocr_pipeline.py --image sample_images/sample_clean.png
    python ocr_pipeline.py --image my_photo.jpg --psm 6
    python ocr_pipeline.py --compare        # raw image vs pre-processed image
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:  # friendly message instead of a traceback
    sys.exit("pytesseract is not installed. Run:  pip install -r requirements.txt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(BASE_DIR, "sample_images", "sample_noisy.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CONFIDENCE_THRESHOLD = 0.80        # the "80% gate" from the slides
DEFAULT_PSM = 3                    # fully automatic page segmentation

PSM_HELP = {
    3: "fully automatic (varied layouts)",
    6: "single uniform block of text",
    7: "single text line",
    11: "sparse, scattered text",
}

WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


# ---------------------------------------------------------------------------
# LIBRARY INTEGRATION
# ---------------------------------------------------------------------------
def configure_tesseract():
    """Locate the Tesseract engine and return its version string."""
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd and os.name == "nt":
        cmd = next((p for p in WINDOWS_TESSERACT_PATHS if os.path.exists(p)), None)
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        return str(pytesseract.get_tesseract_version())
    except pytesseract.TesseractNotFoundError:
        sys.exit(
            "\nTesseract OCR engine not found.\n"
            "pytesseract is only a wrapper - the engine must be installed separately.\n"
            "  Windows: download the installer from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "           (install to the default folder), then reopen VS Code.\n"
            "  Ubuntu : sudo apt install tesseract-ocr\n"
            "  macOS  : brew install tesseract\n"
            "Or set the environment variable TESSERACT_CMD to the full path of tesseract.exe."
        )


# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------
def load_image(path):
    image = cv2.imread(path)           # BGR array: (height, width, 3), values 0-255
    if image is None:
        raise FileNotFoundError(f"Could not read an image at: {path}")
    return image


# ---------------------------------------------------------------------------
# PROCESS: pre-processing (grayscale -> blur -> adaptive threshold -> deskew)
# ---------------------------------------------------------------------------
def to_grayscale(image):
    """Collapse the 3D RGB matrix into a 1D intensity matrix."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray, ksize=5):
    """Smooth micro-imperfections and sensor noise."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def adaptive_threshold(blurred, block_size=31, c=10):
    """
    Force every pixel to choose a side (pure black or pure white).
    'Adaptive' means the cut-off is computed per neighbourhood, so uneven
    lighting and shadows do not wreck the result.
    """
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def rotate(image, angle, border=255):
    """Rotate around the centre (positive angle = counter-clockwise)."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    value = border if image.ndim == 2 else (border,) * 3
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=value)


def estimate_skew(binary, max_angle=10.0, step=0.5):
    """
    Find the rotation (in degrees) that straightens the text.
    Idea: when text lines are perfectly horizontal, the row-by-row count of
    black pixels has sharp peaks and valleys (highest variance).
    """
    small = binary
    if binary.shape[1] > 600:                        # speed: work on a smaller copy
        scale = 600 / binary.shape[1]
        small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ink = (small < 128).astype(np.float32)          # text pixels = 1

    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        profile = rotate(ink, float(angle), border=0).sum(axis=1)
        score = float(np.var(profile))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def preprocess(image, deskew=True):
    """
    Run the full pre-processing chain.
    Returns a dict with every stage so each one can be saved and shown.
    """
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray)
    binary = adaptive_threshold(blurred)

    angle = estimate_skew(binary) if deskew else 0.0
    if abs(angle) < 0.5:                              # already straight enough
        angle = 0.0
    final = binary
    if angle:
        rotated = rotate(binary, angle)
        # rotation blends edge pixels into greys -> snap back to pure black / white
        final = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY)[1]
    straight_color = rotate(image, angle, border=255) if angle else image

    return {
        "gray": gray, "blurred": blurred, "binary": binary,
        "final": final, "angle": angle, "color": straight_color,
    }


# ---------------------------------------------------------------------------
# PROCESS: recognition + confidence filtering
# ---------------------------------------------------------------------------
def run_ocr(image, psm=DEFAULT_PSM):
    """Return every recognised word with its box and confidence (0.0 - 1.0)."""
    data = pytesseract.image_to_data(
        image, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:                      # skip empty boxes / layout rows
            continue
        words.append({
            "text": text,
            "conf": conf / 100.0,
            "box": (data["left"][i], data["top"][i], data["width"][i], data["height"][i]),
            "line": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
        })
    return words


def apply_confidence_gate(words, threshold=CONFIDENCE_THRESHOLD):
    """if confidence >= 0.80: keep  else: drop   (the slide's IF statement)"""
    kept = [w for w in words if w["conf"] >= threshold]
    dropped = [w for w in words if w["conf"] < threshold]
    return kept, dropped


def mean_confidence(words):
    return sum(w["conf"] for w in words) / len(words) if words else 0.0


def words_to_text(words):
    """Rebuild readable text, one output line per detected text line."""
    lines = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w)
    out = []
    for key in sorted(lines):
        row = sorted(lines[key], key=lambda w: w["box"][0])
        out.append(" ".join(w["text"] for w in row))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# OUTPUT: annotated image
# ---------------------------------------------------------------------------
def annotate(color_image, kept, dropped):
    """Green boxes + label for words that pass the gate, thin red for rejected ones."""
    canvas = color_image.copy()
    for w in dropped:
        x, y, bw, bh = w["box"]
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 0, 220), 1)
    for w in kept:
        x, y, bw, bh = w["box"]
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 160, 0), 2)
        label = f"{w['conf'] * 100:.0f}%"
        cv2.putText(canvas, label, (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1, cv2.LINE_AA)
    return canvas


def save_outputs(name, stages, annotated, text):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = {
        "gray": os.path.join(OUTPUT_DIR, f"{name}_1_grayscale.png"),
        "blurred": os.path.join(OUTPUT_DIR, f"{name}_2_blur.png"),
        "binary": os.path.join(OUTPUT_DIR, f"{name}_3_threshold.png"),
        "annotated": os.path.join(OUTPUT_DIR, f"{name}_4_result.png"),
        "text": os.path.join(OUTPUT_DIR, f"{name}_recognised_text.txt"),
    }
    cv2.imwrite(paths["gray"], stages["gray"])
    cv2.imwrite(paths["blurred"], stages["blurred"])
    cv2.imwrite(paths["binary"], stages["final"])
    cv2.imwrite(paths["annotated"], annotated)
    with open(paths["text"], "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return paths


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def process_image(path, psm=DEFAULT_PSM, threshold=CONFIDENCE_THRESHOLD, deskew=True):
    """Run the whole pipeline on one image and return a results dict."""
    image = load_image(path)
    stages = preprocess(image, deskew=deskew)
    all_words = run_ocr(stages["final"], psm=psm)
    kept, dropped = apply_confidence_gate(all_words, threshold)
    text = words_to_text(kept)
    annotated = annotate(stages["color"], kept, dropped)
    return {
        "image": image, "stages": stages, "all_words": all_words,
        "kept": kept, "dropped": dropped, "text": text, "annotated": annotated,
    }


def is_pure_binary(image):
    """True when every pixel is exactly 0 (black) or 255 (white)."""
    return set(np.unique(image).tolist()) <= {0, 255}


def validate(results, engine_version, threshold):
    """The four Gatekeeper checks from the slides -> list of (name, passed, detail)."""
    kept, all_words = results["kept"], results["all_words"]
    checks = [
        ("1. Library integration", True,
         f"pytesseract + Tesseract {engine_version} ran without errors"),
        ("2. Pre-processing integrity", is_pure_binary(results["stages"]["final"]),
         f"grayscale + Gaussian blur + adaptive threshold"
         f" + deskew ({results['stages']['angle']:+.1f} deg), output is pure black/white"),
        ("3. Accuracy benchmark", bool(kept) and mean_confidence(kept) >= threshold,
         f"final output mean confidence {mean_confidence(kept) * 100:.1f}%"
         f" (minimum {threshold * 100:.0f}%)"),
        ("4. Visual confirmation", bool(results["text"].strip()),
         f"{len(kept)} words recognised, annotated image saved"),
    ]
    return checks


def print_report(path, results, engine_version, psm, threshold, saved):
    stages = results["stages"]
    h, w = results["image"].shape[:2]
    bar = "=" * 64
    print(bar)
    print("STEP 1: INPUT")
    print(bar)
    print(f"Image          : {os.path.basename(path)}")
    print(f"Matrix shape   : {h} x {w} x 3  ({h * w * 3:,} data points)")
    print(f"PSM mode       : {psm} ({PSM_HELP.get(psm, 'custom')})")

    print("\n" + bar)
    print("STEP 2: PRE-PROCESSING")
    print(bar)
    print("Grayscale conversion ... done  (3 colour channels -> 1)")
    print("Gaussian blur .......... done  (noise removed)")
    print("Adaptive threshold ..... done  (pure black / white)")
    print(f"Deskew ................. {stages['angle']:+.1f} degrees corrected")

    print("\n" + bar)
    print(f"STEP 3: RECOGNITION + {threshold * 100:.0f}% CONFIDENCE GATE")
    print(bar)
    total = len(results["all_words"])
    print(f"Words detected : {total}")
    print(f"Kept (>= {threshold * 100:.0f}%) : {len(results['kept'])}")
    print(f"Dropped        : {len(results['dropped'])}")
    print(f"Mean confidence: {mean_confidence(results['all_words']) * 100:.1f}% (all words)"
          f" | {mean_confidence(results['kept']) * 100:.1f}% (kept words)")
    if results["dropped"]:
        print("Rejected words : " + ", ".join(
            f"{w['text']} ({w['conf'] * 100:.0f}%)" for w in results["dropped"]))

    print("\n" + bar)
    print("STEP 4: RECOGNISED TEXT")
    print(bar)
    print(results["text"] or "(no text passed the confidence gate)")

    print("\n" + bar)
    print("GATEKEEPER VALIDATION")
    print(bar)
    for name, passed, detail in validate(results, engine_version, threshold):
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print("\nSaved files:")
    for key in ("gray", "blurred", "binary", "annotated", "text"):
        print("  " + os.path.relpath(saved[key], BASE_DIR))


def compare(path, psm, threshold):
    """Show the value of pre-processing: OCR on the raw image vs the cleaned image."""
    image = load_image(path)
    raw_words = run_ocr(image, psm=psm)
    cleaned = process_image(path, psm=psm, threshold=threshold)
    print("=" * 64)
    print("COMPARISON: raw image vs pre-processed image")
    print("=" * 64)
    print(f"{'':<22}{'words':>8}{'mean conf':>12}{'kept >= gate':>15}")
    raw_kept = apply_confidence_gate(raw_words, threshold)[0]
    print(f"{'Raw image':<22}{len(raw_words):>8}{mean_confidence(raw_words) * 100:>11.1f}%{len(raw_kept):>15}")
    c_words = cleaned["all_words"]
    print(f"{'Pre-processed':<22}{len(c_words):>8}{mean_confidence(c_words) * 100:>11.1f}%{len(cleaned['kept']):>15}")
    print()


def main():
    parser = argparse.ArgumentParser(description="OCR pipeline with pytesseract + OpenCV")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="path to an image file")
    parser.add_argument("--psm", type=int, default=DEFAULT_PSM, choices=sorted(PSM_HELP),
                        help="page segmentation mode (3, 6, 7 or 11)")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help="confidence gate between 0 and 1 (default 0.80)")
    parser.add_argument("--no-deskew", action="store_true", help="skip rotation correction")
    parser.add_argument("--compare", action="store_true", help="compare raw vs pre-processed OCR")
    args = parser.parse_args()

    engine_version = configure_tesseract()
    try:
        if args.compare:
            compare(args.image, args.psm, args.threshold)
        results = process_image(args.image, args.psm, args.threshold, deskew=not args.no_deskew)
    except FileNotFoundError as err:
        sys.exit(str(err))

    name = os.path.splitext(os.path.basename(args.image))[0]
    saved = save_outputs(name, results["stages"], results["annotated"], results["text"])
    print_report(args.image, results, engine_version, args.psm, args.threshold, saved)


if __name__ == "__main__":
    main()
