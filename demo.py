"""
UrbanSense — Demo with webcam or sample video.
Usage:
  Webcam:      python demo.py --model yolov8n.onnx
  Video file:  python demo.py --model yolov8n.onnx --source video.mp4
  Download sample YOLOv8n ONNX:
    pip install ultralytics && yolo export model=yolov8n.pt format=onnx
"""

from detect import UrbanSenseDetector, run_video
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="UrbanSense Demo")
    p.add_argument("--model",  required=True, help="ONNX model path (e.g. yolov8n.onnx)")
    p.add_argument("--source", default="0",   help="Video source: 0=webcam, or file path")
    p.add_argument("--output", default=None,  help="Save output to this path (optional)")
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.45)
    a = p.parse_args()

    print("Controls: press Q to quit")
    detector = UrbanSenseDetector(a.model, a.conf, a.iou)
    source = int(a.source) if a.source.isdigit() else a.source
    run_video(detector, source, a.output)
