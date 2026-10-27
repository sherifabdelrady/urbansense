"""
tests/test_detect.py — UrbanSense detection unit tests
Runs without any ONNX model (uses a mock session).
"""
import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from detect import (
    UrbanSenseDetector, letterbox, COCO_NAMES, COLORS,
)


# ---------------------------------------------------------------------------
# letterbox
# ---------------------------------------------------------------------------

class TestLetterbox:
    def test_square_input_no_padding(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, scale, (dw, dh) = letterbox(img, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)

    def test_wide_image_horizontal_bars(self):
        img = np.zeros((320, 640, 3), dtype=np.uint8)
        padded, scale, (dw, dh) = letterbox(img, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert dh > 0, "Expected vertical padding for wide image"
        assert dw == pytest.approx(0.0, abs=1)

    def test_tall_image_vertical_bars(self):
        img = np.zeros((640, 320, 3), dtype=np.uint8)
        padded, scale, (dw, dh) = letterbox(img, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert dw > 0, "Expected horizontal padding for tall image"

    def test_scale_is_correct(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        _, scale, _ = letterbox(img, (640, 640))
        assert scale == pytest.approx(640 / 200)


# ---------------------------------------------------------------------------
# Postprocessing — critical YOLOv8 format tests
# ---------------------------------------------------------------------------

def _make_mock_detector():
    """Build a UrbanSenseDetector without loading any real ONNX model."""
    det = object.__new__(UrbanSenseDetector)
    det.conf        = 0.25
    det.iou         = 0.45
    det.input_shape = (640, 640)
    det.num_classes = 80
    return det


class TestPostprocessYOLOv8:
    """Test the fixed YOLOv8 postprocessing (no objectness at index 4)."""

    def test_yolov8_format_detection(self):
        """Single high-confidence detection: shape [4+80, N], transposed internally."""
        det = _make_mock_detector()
        # Build a row: cx=320, cy=320, w=100, h=100, then 80 class scores
        class_scores = np.zeros(80, dtype=np.float32)
        class_scores[0] = 0.9   # "person" at 90% confidence
        row = np.array([320, 320, 100, 100], dtype=np.float32)
        row = np.concatenate([row, class_scores])  # (84,)
        # Shape [84, 1] — YOLOv8 format before T transpose
        output = row.reshape(84, 1)   # [4+C, N]
        results = det.postprocess(output, scale=1.0, pad=(0.0, 0.0))
        assert len(results) == 1
        assert results[0]["class_name"] == "person"
        assert results[0]["confidence"] == pytest.approx(0.9, abs=0.01)

    def test_no_objectness_at_index_4(self):
        """Verify we do NOT treat index 4 as objectness (old bug)."""
        det = _make_mock_detector()
        # index 4 = class 0 score = 0.05 (below threshold)
        # index 10 = class 6 score = 0.8 (above threshold)
        class_scores = np.zeros(80, dtype=np.float32)
        class_scores[0] = 0.05   # class 0: low
        class_scores[6] = 0.8    # class 6: "train" — high
        row = np.concatenate([[320, 320, 100, 100], class_scores])  # (84,)
        output = row.reshape(84, 1)
        results = det.postprocess(output, scale=1.0, pad=(0.0, 0.0))
        assert len(results) == 1
        assert results[0]["class_id"] == 6
        assert results[0]["class_name"] == "train"

    def test_below_threshold_filtered(self):
        det = _make_mock_detector()
        class_scores = np.zeros(80, dtype=np.float32)
        class_scores[0] = 0.1   # below 0.25 threshold
        row = np.concatenate([[320, 320, 100, 100], class_scores])
        output = row.reshape(84, 1)
        results = det.postprocess(output, scale=1.0, pad=(0.0, 0.0))
        assert results == []

    def test_bbox_coords_with_padding(self):
        """Bounding box should correctly un-pad letterbox offset."""
        det = _make_mock_detector()
        class_scores = np.zeros(80, dtype=np.float32)
        class_scores[2] = 0.8   # "car"
        # Box centred at (320, 320) with size 200x100 — no scaling
        row = np.concatenate([[320, 320, 200, 100], class_scores])
        output = row.reshape(84, 1)
        # Simulate: scale=1.0, dw=0, dh=80 (image padded top+bottom)
        results = det.postprocess(output, scale=1.0, pad=(0.0, 80.0))
        assert len(results) == 1
        x1, y1, x2, y2 = results[0]["bbox"]
        # y should be adjusted by -80 (pad removed)
        assert y1 == pytest.approx(320 - 50 - 80, abs=2)
        assert y2 == pytest.approx(320 + 50 - 80, abs=2)

    def test_out_of_range_class_id_skipped(self):
        """class_id >= len(COCO_NAMES) must be silently skipped."""
        det = _make_mock_detector()
        det.num_classes = 120   # more than COCO_NAMES has
        class_scores = np.zeros(120, dtype=np.float32)
        class_scores[100] = 0.99   # index 100 > 80 COCO classes
        row = np.concatenate([[320, 320, 100, 100], class_scores])
        output = row.reshape(124, 1)
        results = det.postprocess(output, scale=1.0, pad=(0.0, 0.0))
        assert results == [], "Out-of-range class_id should be dropped"


class TestColors:
    def test_deterministic(self):
        """COLORS must be identical across two imports (fixed seed)."""
        from detect import COLORS as C1
        import importlib, detect
        importlib.reload(detect)
        from detect import COLORS as C2
        np.testing.assert_array_equal(C1, C2)


class TestCocoNames:
    def test_80_classes(self):
        assert len(COCO_NAMES) == 80

    def test_first_and_last(self):
        assert COCO_NAMES[0]  == "person"
        assert COCO_NAMES[79] == "toothbrush"
