"""Independent hand detection and regression; actor identity is not inferred here."""

from importlib.metadata import version
from pathlib import Path

import numpy as np

from .detector_pilot import box_iou
from .pose_models import checked_model, crop_interior_mask, preprocess_pose_crop, require_cuda, require_cuda_provider, simcc_batch


HAND_DETECTOR_SHA256 = "568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2"
HAND_POSE_SHA256 = "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29"


def tile_rectangles(width: int, height: int, size: int = 864, overlap: float = .25) -> list[tuple]:
    """Full frame plus overlapping tiles, including the right and bottom edges."""
    if min(width, height, size) <= 0 or not 0 <= overlap < 1:
        raise ValueError("Invalid tile geometry")
    step = max(1, round(size * (1 - overlap)))
    def starts(length):
        return sorted(set([*range(0, max(1, length - size + 1), step), max(0, length - size)]))
    full = (0, 0, width, height)
    tiles = [(x, y, min(x + size, width), min(y + size, height))
             for y in starts(height) for x in starts(width)]
    return list(dict.fromkeys([full, *tiles]))


def suppress_boxes(detections: list[dict], threshold: float = .5) -> list[dict]:
    if not 0 < threshold <= 1:
        raise ValueError("Invalid NMS threshold")
    kept = []
    for item in sorted(detections, key=lambda d: -d["confidence"]):
        if all(box_iou(item["bbox_xyxy_px"], previous["bbox_xyxy_px"]) < threshold for previous in kept):
            kept.append(item)
    return kept


class HandDetector:
    def __init__(self, path: Path, threshold: float = .05, tiled: bool = False):
        if not 0 < threshold < 1:
            raise ValueError("Invalid hand detection threshold")
        self.provenance = {"model": "rtmdet_nano_hand_320", "input_color": "BGR",
                           "checkpoint_sha256": checked_model(path, HAND_DETECTOR_SHA256),
                           "input_size_wh": [320, 320], "confidence_threshold": threshold,
                           "embedded_nms_iou": .6, "tile_nms_iou": .5,
                           "tile_size_px": 864 if tiled else None, "tile_overlap": .25 if tiled else None,
                           "runtime": "onnxruntime_cuda", "rtmlib": version("rtmlib"),
                           "cuda": require_cuda(), "actor_identity": "unknown"}
        self.threshold, self.tiled = threshold, tiled
        from rtmlib import RTMDet
        self.model = RTMDet(str(path), model_input_size=(320, 320), backend="onnxruntime", device="cuda")
        require_cuda_provider(self.model)
        if tuple(self.model.session.get_inputs()[0].shape[-2:]) != (320, 320):
            raise ValueError("Unexpected hand detector input shape")

    def __call__(self, bgr: np.ndarray) -> list[dict]:
        height, width = bgr.shape[:2]
        rectangles = tile_rectangles(width, height) if self.tiled else [(0, 0, width, height)]
        detections = []
        for rectangle in rectangles:
            x0, y0, x1, y1 = rectangle
            crop, ratio = self.model.preprocess(bgr[y0:y1, x0:x1])
            outputs = self.model.inference(crop)[0]
            if outputs.ndim != 3 or outputs.shape[0] != 1 or outputs.shape[-1] != 5:
                raise ValueError(f"Unexpected hand detector output shape: {outputs.shape}")
            for row in outputs[0]:
                if row[4] < self.threshold:
                    continue
                if not np.isfinite(row).all():
                    raise ValueError("Nonfinite hand detection")
                box = row[:4] / ratio + [x0, y0, x0, y0]
                box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
                if np.any(box[2:] <= box[:2]):
                    continue
                detections.append({"bbox_xyxy_px": box.tolist(), "confidence": float(row[4]),
                                   "class_name": "hand", "actor": "unknown",
                                   "source_crop_xyxy_px": list(rectangle)})
        return suppress_boxes(detections)


class HandPose:
    topology = "hand5_21"
    point_count = 21

    def __init__(self, path: Path):
        self.provenance = {"model": "rtmpose_m_hand5_256", "topology": self.topology,
                           "checkpoint_sha256": checked_model(path, HAND_POSE_SHA256),
                           "input_color": "RGB", "input_size_wh": [256, 256], "bbox_padding": 1.25,
                           "crop_boundary_margin_input_px": 2.0,
                           "confidence_kind": "simcc_response_not_calibrated",
                           "runtime": "onnxruntime_cuda", "rtmlib": version("rtmlib"),
                           "cuda": require_cuda(), "flip_test": False,
                           "preprocessing_note": "256 square follows model shape, codec, and RTMLib; archive affine metadata says 192x256"}
        from rtmlib import RTMPose
        self.model = RTMPose(str(path), model_input_size=(256, 256), to_openpose=False,
                             backend="onnxruntime", device="cuda")
        require_cuda_provider(self.model)
        if tuple(self.model.session.get_inputs()[0].shape[-2:]) != (256, 256):
            raise ValueError("Unexpected hand pose input shape")

    def __call__(self, bgr: np.ndarray, bbox: list) -> tuple[np.ndarray, np.ndarray]:
        crop, center, scale = preprocess_pose_crop(self.model,bgr,bbox)
        outputs = self.model.inference(crop)
        points, scores = self.model.postprocess(outputs, center, scale)
        if points.shape != (1, 21, 2) or scores.shape != (1, 21):
            raise ValueError(f"Unexpected hand pose shape: {points.shape}, {scores.shape}")
        self.last_crop_valid = crop_interior_mask(points[0], center, scale, (256, 256))
        return points[0], scores[0]

    def batch(self, images:list[np.ndarray], boxes:list):
        return simcc_batch(self.model,images,boxes,(256,256),21)
