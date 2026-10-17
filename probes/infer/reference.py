"""Independent .pt/Ultralytics comparison against the real C++ probe XML.

Requires locally supplied matching yolov8n.pt, torch, ultralytics, numpy and
opencv-python. This script never downloads weights or installs dependencies.
Usage: python reference.py --weights yolov8n.pt --probe infer.xml --report python-reference.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def number(parent, name, cast=float):
    node = parent.find(name)
    if node is None or node.text is None:
        raise ValueError(f"Missing probe field: {name}")
    return cast(node.text.strip())


def text(parent, name):
    node = parent.find(name)
    if node is None or node.text is None:
        raise ValueError(f"Missing probe field: {name}")
    value = node.text.strip()
    # OpenCV XML quotes strings containing whitespace, unlike numeric fields.
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value


def detection(node):
    coordinates = [float(value) for value in text(node, "box").split()]
    if len(coordinates) != 4:
        raise ValueError("Probe detection box must have four continuous edges")
    return {"box": coordinates, "score": number(node, "score"), "class_id": number(node, "class_id", int)}


def overlap(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def compare(native, independent):
    edges = []
    metrics = {}
    for i, left in enumerate(native):
        eligible = []
        for j, right in enumerate(independent):
            if left["class_id"] != right["class_id"]:
                continue
            box = left["box"]
            small = min(box[2] - box[0], box[3] - box[1]) < 16.0
            iou = overlap(box, right["box"])
            score = abs(left["score"] - right["score"])
            edge = max(abs(x - y) for x, y in zip(box, right["box"]))
            metrics[i, j] = {"native_index": i, "python_index": j, "class_id": left["class_id"],
                             "iou": iou, "score_difference": score, "maximum_edge_error": edge, "small_box": small}
            if score <= 0.01 and (edge <= 0.5 if small else iou >= 0.98):
                eligible.append(j)
        edges.append(sorted(eligible, key=lambda j: -metrics[i, j]["iou"]))
    owners = [-1] * len(independent)

    def augment(index, seen):
        for candidate in edges[index]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if owners[candidate] < 0 or augment(owners[candidate], seen):
                owners[candidate] = index
                return True
        return False

    for index in range(len(native)):
        augment(index, set())
    matches = [metrics[index, candidate] for candidate, index in enumerate(owners) if index >= 0]
    used = {index for index in owners if index >= 0}
    unmatched_native = [index for index in range(len(native)) if index not in used]
    unmatched_python = [index for index, owner in enumerate(owners) if owner < 0]
    return {"matches": matches, "unmatched_native": unmatched_native, "unmatched_python": unmatched_python,
            "all_same_class_pair_metrics": list(metrics.values()),
            "passed": not unmatched_native and not unmatched_python}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or args.weights.suffix.lower() != ".pt":
        parser.error("--weights must be an existing matching local yolov8n.pt; automatic downloads are prohibited")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        import cv2
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        parser.error(f"Independent reference dependencies unavailable (not installed by this script): {error}")
    root = ET.parse(args.probe).getroot()
    if text(root, "schema") != "VisionRT.VR3-002.VR3-027.probe.2":
        parser.error("Expected current production-path probe schema")
    model = YOLO(str(args.weights.resolve()), task="detect")
    count = number(root, "num_classes", int)
    if len(model.names) != count:
        parser.error(f".pt has {len(model.names)} classes but native model has {count}")
    records = []
    for node in root.findall("images/_"):
        image_path = text(node, "path")
        if image_path == "generated:black:321x197":
            image = np.zeros((197, 321, 3), dtype=np.uint8)
        else:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # Independent Ultralytics preprocessing, .pt execution, decode and NMS;
        # no C++ engine, cached detections or business decoder are used here.
        predictions = model.predict(source=image, device="cpu", half=False,
            imgsz=(number(root, "height", int), number(root, "width", int)), rect=False,
            conf=number(root, "confidence"), iou=number(root, "nms"),
            max_det=number(root, "maximum_detections", int), agnostic_nms=False, verbose=False)
        boxes = predictions[0].boxes
        independent = [{"box": coordinates.tolist(), "score": float(score), "class_id": int(category)}
                       for coordinates, score, category in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]
        native = [detection(value) for value in node.findall("build/detections/_")]
        records.append({"image": image_path, "native": native, "python": independent,
                        "comparison": compare(native, independent)})
    if not records:
        raise ValueError("Probe report has no images")
    result = {"schema": "VisionRT.independent-python.1", "weights": str(args.weights.resolve()),
              "probe": str(args.probe.resolve()), "device": "cpu", "precision": "fp32",
              "native_precision": text(root, "precision"), "torch_version": torch.__version__,
              "ultralytics_version": ultralytics.__version__, "opencv_version": cv2.__version__,
              "geometry_note": "Ultralytics uses its independent original-image remapping; native uses actual per-axis integer resize geometry",
              "images": records, "passed": all(value["comparison"]["passed"] for value in records)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Independent reference: {args.report}; passed={result['passed']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
