"""Independent .pt/Ultralytics comparison against the real C++ probe YAML (YR-04 / YVT2-04).

Requires locally supplied matching yolov8n.pt, torch, ultralytics, numpy and
opencv-python. This script never downloads weights or installs dependencies.
Usage: python reference.py --weights yolov8n.pt --probe probe_report.yaml --report reference_report.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import yaml

class UniqueKeyLoader(yaml.SafeLoader):
    pass

def _construct_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key detected in YAML: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping

UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=UniqueKeyLoader)

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
            metrics[i, j] = {
                "native_index": i, "python_index": j, "class_id": left["class_id"],
                "iou": iou, "score_difference": score, "maximum_edge_error": edge, "small_box": small
            }
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
    return {
        "matches": matches,
        "unmatched_native": unmatched_native,
        "unmatched_python": unmatched_python,
        "all_same_class_pair_metrics": list(metrics.values()),
        "passed": not unmatched_native and not unmatched_python
    }

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
        parser.error(f"Independent reference dependencies unavailable: {error}")

    probe_doc = load_yaml(args.probe)
    if not isinstance(probe_doc, dict):
        parser.error(f"Probe file {args.probe} must be a valid YAML dictionary")

    scope = probe_doc.get("scope")
    if scope != "gpu_release":
        parser.error(f"Expected scope 'gpu_release' in probe YAML, got '{scope}'")

    model = YOLO(str(args.weights.resolve()), task="detect")
    params = probe_doc.get("parameters", {})
    count = params.get("classes", 80)
    if len(model.names) != count:
        parser.error(f".pt has {len(model.names)} classes but probe expects {count}")

    records = []
    images = probe_doc.get("images", [])
    for node in images:
        image_path = node.get("path")
        if not image_path:
            continue
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

        h = params.get("height", 640)
        w = params.get("width", 640)
        conf = params.get("confidence", 0.25)
        nms = params.get("nms", 0.45)
        max_det = params.get("max_detections", 300)

        predictions = model.predict(
            source=image,
            device="cpu",
            half=False,
            imgsz=(h, w),
            rect=False,
            conf=conf,
            iou=nms,
            max_det=max_det,
            agnostic_nms=False,
            verbose=False
        )
        boxes = predictions[0].boxes
        independent = [
            {
                "box": coordinates.tolist(),
                "score": float(score),
                "class_id": int(category)
            }
            for coordinates, score, category in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy()
            )
        ]
        native = node.get("detections", [])
        records.append({
            "image": image_path,
            "native": native,
            "python": independent,
            "comparison": compare(native, independent)
        })

    if not records:
        raise ValueError("Probe report has no valid image entries")

    result = {
        "schema": "VisionRT.independent-python.yaml.1",
        "weights": str(args.weights.resolve()),
        "probe": str(args.probe.resolve()),
        "device": "cpu",
        "precision": "fp32",
        "native_precision": params.get("precision", "fp32"),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "geometry_note": "Ultralytics uses its independent original-image remapping; native uses actual per-axis integer resize geometry",
        "images": records,
        "passed": all(val["comparison"]["passed"] for val in records)
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        yaml.dump(result, f, sort_keys=False, allow_unicode=True)

    print(f"Independent reference report written: {args.report}; passed={result['passed']}")
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
