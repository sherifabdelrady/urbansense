"""
UrbanSense - Real-Time Urban Object Detection
YOLOv8 + INT8 ONNX pipeline for edge deployment.

Audit fixes (audit/production-remediation):
- P1 FIX: YOLOv8 ONNX output has NO objectness at index 4.
  Output shape is [1, 84, 8400] (transposed → [1, 8400, 84]).
  Each row = [cx, cy, w, h, cls0_score ... cls79_score].
  Old code: conf=row[4] was reading first class score as objectness.
  Fix: class confidence = max of class columns (row[4:]).
- P1 FIX: Added letterboxing to preserve aspect ratio.
  Plain cv2.resize() distorted non-square images, causing wrong bbox coords.
- P2 FIX: Fixed seed for COLORS so colors are deterministic across runs.
- P2 FIX: cap.release() + destroyAllWindows() in try/finally to prevent leaks.
- P2 FIX: FPS now measures full pipeline (preprocess+infer+postprocess).
- P2 FIX: Added bounds-check on class_id vs COCO_NAMES length.
- P2 FIX: model metadata reading made safe (was silently ignored on failure).
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
from pathlib import Path

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

# Fixed seed: deterministic colors across runs
_rng = np.random.default_rng(42)
COLORS = _rng.integers(80, 255, size=(len(COCO_NAMES), 3), dtype=np.uint8)


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize image to new_shape preserving aspect ratio with grey padding.

    Returns:
        padded_img : resized + padded image
        scale      : scale factor applied to original image
        (dw, dh)   : total horizontal and vertical padding in pixels
    """
    h, w = img.shape[:2]
    nh, nw = new_shape
    scale = min(nw / w, nh / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    dw = (nw - new_w) / 2
    dh = (nh - new_h) / 2
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, scale, (dw, dh)


class UrbanSenseDetector:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        self.conf = conf
        self.iou  = iou
        self.input_shape = (640, 640)   # (H, W)

        # Determine actual number of classes from model output shape
        # YOLOv8 output: [1, 4+num_classes, 8400]
        out_meta   = self.session.get_outputs()[0]
        # out_meta.shape is [1, num_attributes, num_boxes]
        try:
            num_attrs = out_meta.shape[1]   # e.g. 84 for 80-class COCO
            self.num_classes = num_attrs - 4
        except Exception:
            self.num_classes = len(COCO_NAMES)

        active_provider = self.session.get_providers()[0]
        print(f"Loaded: {model_path} | Provider: {active_provider} | Classes: {self.num_classes}")

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Letterbox → BGR→RGB → normalise → NCHW tensor.
        Returns (blob, scale, (dw, dh)) for accurate coordinate remapping.
        """
        padded, scale, (dw, dh) = letterbox(frame, self.input_shape)
        rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis]  # HWC → NCHW
        return blob, scale, (dw, dh)

    def postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
    ) -> list[dict]:
        """
        Parse YOLOv8 ONNX output and map boxes back to original image coordinates.

        YOLOv8 ONNX output shape: [1, 4+num_classes, num_boxes]
        After transpose:           [1, num_boxes, 4+num_classes]
        Each row = [cx, cy, w, h, cls0_score, cls1_score, ..., clsN_score]

        IMPORTANT: YOLOv8 has NO objectness score (unlike YOLOv5).
        Confidence = max class score directly.
        """
        dw, dh = pad
        # output is already the raw ort output, shape [1, 4+C, N] or [1, N, 4+C]
        raw = output[0]  # strip batch dim
        # Normalise to [N, 4+C]
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            # shape is [4+C, N] — transpose to [N, 4+C]
            raw = raw.T

        boxes, scores, class_ids = [], [], []
        for row in raw:
            class_scores = row[4:]                          # [C]
            class_id     = int(np.argmax(class_scores))    # highest-confidence class
            confidence   = float(class_scores[class_id])   # no objectness in YOLOv8

            if confidence < self.conf:
                continue
            if class_id >= len(COCO_NAMES):
                continue

            cx, cy, w, h = row[:4]
            # Remove letterbox padding, then undo scale
            x1 = int((cx - w / 2 - dw) / scale)
            y1 = int((cy - h / 2 - dh) / scale)
            x2 = int((cx + w / 2 - dw) / scale)
            y2 = int((cy + h / 2 - dh) / scale)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
        results = []
        for i in indices:
            x, y, bw, bh = boxes[i]
            results.append({
                "bbox":       [x, y, x + bw, y + bh],
                "class_id":   class_ids[i],
                "class_name": COCO_NAMES[class_ids[i]],
                "confidence": round(scores[i], 4),
            })
        return results

    def detect(self, frame: np.ndarray) -> tuple[list[dict], float]:
        """Full pipeline: preprocess → infer → postprocess. Returns (detections, pipeline_ms)."""
        t0   = time.perf_counter()
        blob, scale, pad = self.preprocess(frame)
        output = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        detections = self.postprocess(output[0], scale, pad)
        latency_ms = (time.perf_counter() - t0) * 1000
        return detections, latency_ms

    def draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cid   = d["class_id"]
            color = COLORS[cid % len(COLORS)].tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['class_name']} {d['confidence']:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame


def run_video(detector: UrbanSenseDetector, source, output_path: str | None = None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

    frame_count  = 0
    total_latency = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections, latency_ms = detector.detect(frame)
            total_latency += latency_ms
            frame_count   += 1
            frame = detector.draw(frame, detections)
            avg_fps = 1000.0 / (total_latency / frame_count)
            cv2.putText(frame, f"FPS: {avg_fps:.1f} | {latency_ms:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if writer:
                writer.write(frame)
            cv2.imshow("UrbanSense", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if frame_count:
        print(f"Processed {frame_count} frames | Avg pipeline FPS: {1000/(total_latency/frame_count):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanSense real-time object detection")
    parser.add_argument("--model",  required=True, help="Path to YOLOv8 ONNX model")
    parser.add_argument("--source", default="0",   help="Video source (file, URL, or 0 for webcam)")
    parser.add_argument("--output", default=None,  help="Output video path (optional)")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--iou",    type=float, default=0.45)
    args = parser.parse_args()

    source   = int(args.source) if args.source.isdigit() else args.source
    detector = UrbanSenseDetector(args.model, args.conf, args.iou)
    run_video(detector, source, args.output)
