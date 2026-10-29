import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocr_pipeline as p

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_images")
CLEAN = os.path.join(SAMPLES, "sample_clean.png")
NOISY = os.path.join(SAMPLES, "sample_noisy.png")


def tesseract_available():
    try:
        p.configure_tesseract()
        return True
    except SystemExit:
        return False


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = p.load_image(NOISY)
        cls.stages = p.preprocess(cls.image)

    def test_image_is_3d_array(self):
        self.assertEqual(self.image.ndim, 3)
        self.assertEqual(self.image.shape[2], 3)

    def test_grayscale_is_2d(self):
        self.assertEqual(self.stages["gray"].ndim, 2)
        self.assertEqual(self.stages["gray"].shape, self.image.shape[:2])

    def test_threshold_output_is_pure_black_and_white(self):
        self.assertTrue(p.is_pure_binary(self.stages["binary"]))
        self.assertTrue(p.is_pure_binary(self.stages["final"]))

    def test_blur_reduces_noise(self):
        self.assertLess(float(np.std(self.stages["blurred"])), float(np.std(self.stages["gray"])) + 1e-6)

    def test_deskew_detects_the_tilt(self):
        self.assertAlmostEqual(self.stages["angle"], -3.0, delta=0.6)

    def test_deskew_straightens_a_rotated_image(self):
        clean = p.load_image(CLEAN)
        tilted = p.rotate(clean, 4.0)
        angle = p.preprocess(tilted)["angle"]
        self.assertAlmostEqual(angle, -4.0, delta=0.6)

    def test_straight_image_is_not_rotated(self):
        self.assertEqual(p.preprocess(p.load_image(CLEAN))["angle"], 0.0)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            p.load_image("does_not_exist.png")


class TestConfidenceGate(unittest.TestCase):
    WORDS = [
        {"text": "A", "conf": 0.95, "box": (0, 0, 10, 10), "line": (1, 1, 1)},
        {"text": "B", "conf": 0.80, "box": (20, 0, 10, 10), "line": (1, 1, 1)},
        {"text": "C", "conf": 0.45, "box": (40, 0, 10, 10), "line": (1, 1, 1)},
    ]

    def test_gate_keeps_80_percent_and_above(self):
        kept, dropped = p.apply_confidence_gate(self.WORDS)
        self.assertEqual([w["text"] for w in kept], ["A", "B"])
        self.assertEqual([w["text"] for w in dropped], ["C"])

    def test_custom_threshold(self):
        kept, _ = p.apply_confidence_gate(self.WORDS, threshold=0.9)
        self.assertEqual([w["text"] for w in kept], ["A"])

    def test_mean_confidence(self):
        self.assertAlmostEqual(p.mean_confidence(self.WORDS[:2]), 0.875)
        self.assertEqual(p.mean_confidence([]), 0.0)

    def test_text_rebuilt_in_reading_order(self):
        words = [
            {"text": "world", "conf": 0.9, "box": (60, 0, 5, 5), "line": (1, 1, 1)},
            {"text": "hello", "conf": 0.9, "box": (0, 0, 5, 5), "line": (1, 1, 1)},
            {"text": "next", "conf": 0.9, "box": (0, 20, 5, 5), "line": (1, 1, 2)},
        ]
        self.assertEqual(p.words_to_text(words), "hello world\nnext")

    def test_annotation_keeps_image_size(self):
        img = np.full((50, 80, 3), 255, np.uint8)
        out = p.annotate(img, self.WORDS[:1], self.WORDS[2:])
        self.assertEqual(out.shape, img.shape)


@unittest.skipUnless(tesseract_available(), "Tesseract engine not installed")
class TestEndToEnd(unittest.TestCase):
    def test_clean_image_is_read_correctly(self):
        r = p.process_image(CLEAN)
        self.assertIn("Invoice", r["text"])
        self.assertIn("20260429", r["text"])
        self.assertGreaterEqual(p.mean_confidence(r["kept"]), 0.80)

    def test_noisy_tilted_image_passes_all_four_checks(self):
        r = p.process_image(NOISY)
        checks = p.validate(r, "test", p.CONFIDENCE_THRESHOLD)
        self.assertTrue(all(ok for _, ok, _ in checks), checks)
        self.assertIn("Recognition", r["text"])

    def test_preprocessing_improves_confidence(self):
        raw = p.run_ocr(p.load_image(NOISY))
        cleaned = p.process_image(NOISY)["all_words"]
        self.assertGreater(p.mean_confidence(cleaned), p.mean_confidence(raw))


if __name__ == "__main__":
    unittest.main()
