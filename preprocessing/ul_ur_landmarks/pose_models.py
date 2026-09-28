"""GPU participant-pose adapters with explicit topology and checkpoint provenance.

RTMW uses the OpenMMLab ONNX export through RTMLib. Model adapters receive
OpenCV BGR images and explicit participant boxes; they never invent a person
when the detector returns no box.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np

from .schema import sha256


RTMW_SHA256 = "bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054"
YOLOX_SHA256 = "3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23"
SAPIENS_133_SHA256 = "911c4a26dbede2d21fa1a3bffb8d1b5ecac6bcb4983951408ba80092bc07b054"
SAPIENS_DECODER_SHA256 = "78aa6c9d4631c00fe05362ea1e9066821e8a1e4c7b1f58d1a34cf2715e3342f7"


def checked_model(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return actual


def visible_points(xy: np.ndarray, scores: np.ndarray, width: int, height: int,
                   threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validity is explicit; (0, 0) can be valid and missing coordinates are NaN."""
    xy, scores = np.asarray(xy, dtype=np.float32), np.asarray(scores, dtype=np.float32)
    if xy.shape[:-1] != scores.shape or xy.shape[-1] != 2:
        raise ValueError("Coordinate and confidence shapes differ")
    valid = (np.isfinite(xy).all(axis=-1) & np.isfinite(scores) & (scores >= threshold)
             & (xy[..., 0] >= 0) & (xy[..., 0] < width)
             & (xy[..., 1] >= 0) & (xy[..., 1] < height))
    xy = xy.copy()
    xy[~valid] = np.nan
    return xy, scores.copy(), valid


def crop_interior_mask(points: np.ndarray, center: np.ndarray, scale: np.ndarray,
                       size_wh: tuple[int, int], margin: float = 2.0) -> np.ndarray:
    """Flag saturated crop-edge predictions; this does not estimate occlusion."""
    normalized = (np.asarray(points) - center + scale / 2) / scale
    position = normalized * np.asarray(size_wh)
    return (np.isfinite(position).all(axis=-1) & (position >= margin).all(axis=-1)
            & (position < np.asarray(size_wh) - margin).all(axis=-1))


def require_cuda() -> dict:
    # Import torch first so its CUDA/cuDNN libraries are available to ONNX Runtime.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA inference is unavailable; no implicit CPU fallback")
    value = torch.ones((32, 32), device="cuda")
    result = value @ value
    torch.cuda.synchronize()
    if not torch.all(result == 32).item():
        raise RuntimeError("CUDA arithmetic check failed")
    return {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "cuda_operation_verified": True}


def require_cuda_provider(tool) -> None:
    if "CUDAExecutionProvider" not in tool.session.get_providers():
        raise RuntimeError("ONNX model fell back to CPU; comparison requires CUDA")


def select_participant_box(boxes: np.ndarray, image_height: int) -> int | None:
    """QUB UL/UR seated participant is above the other actor entering from below.

    Rank eligible detections by vertical center, with area as a deterministic
    tie-breaker. Never manufacture a participant from a lower-half-only box.
    This camera-layout prior is recorded and must be audited on new views.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    eligible = [i for i, box in enumerate(boxes) if box[1] < image_height / 2]
    if not eligible:
        return None
    return min(eligible, key=lambda i: ((boxes[i,1] + boxes[i,3]) / 2,
                                       -np.prod(boxes[i,2:] - boxes[i,:2])))


def simcc_batch(model, images: list[np.ndarray], boxes: list, size_wh: tuple[int,int], point_count:int):
    """Use the official export's dynamic batch axis; checkpoint bytes are unchanged."""
    if not images or len(images)!=len(boxes):
        raise ValueError('Pose batches must be nonempty with one box per image')
    crops,centers,scales=[],[],[]
    for image,box in zip(images,boxes):
        crop,center,scale=preprocess_pose_crop(model,image,box)
        crops.append(crop.transpose(2,0,1))
        centers.append(center)
        scales.append(scale)
    tensor=np.ascontiguousarray(np.stack(crops),dtype=np.float32)
    outputs=model.session.run(None,{model.session.get_inputs()[0].name:tensor})
    centers=np.asarray(centers)[:,None,:]
    scales=np.asarray(scales)[:,None,:]
    points,scores=model.postprocess(outputs,centers,scales)
    if points.shape!=(len(images),point_count,2) or scores.shape!=(len(images),point_count):
        raise ValueError(f'Unexpected batched pose shape: {points.shape}, {scores.shape}')
    interior=crop_interior_mask(points,centers,scales,size_wh)
    return points,scores,interior


def preprocess_pose_crop(model, bgr:np.ndarray, box):
    """Warp before swapping channels, avoiding a full-frame RGB copy per hand.

    The affine transform acts independently on channels. Crop normalization
    retains RTMLib's arithmetic and the export's required RGB ordering.
    """
    from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs,top_down_affine
    center,scale=bbox_xyxy2cs(np.asarray(box),padding=1.25)
    crop,scale=top_down_affine(model.model_input_size,scale,center,bgr)
    crop=np.ascontiguousarray(crop[...,::-1])
    if model.mean is not None:
        crop=(crop-np.asarray(model.mean))/np.asarray(model.std)
    return crop,center,scale


class PersonDetector:
    """YOLOX HumanArt boxes; participant selection is recorded separately."""

    def __init__(self, model: Path, threshold: float = .7):
        if not 0 < threshold < 1:
            raise ValueError("Invalid person detector threshold")
        self.provenance = {"model": "yolox_m_humanart",
                           "checkpoint_sha256": checked_model(model, YOLOX_SHA256),
                           "input_size_wh": [640, 640], "score_threshold": threshold,
                           "nms_threshold": .65, "nms_location": "embedded_in_pinned_export",
                           "input_color": "BGR", "runtime": "onnxruntime_cuda",
                           "rtmlib": version("rtmlib")}
        require_cuda()
        from rtmlib import YOLOX

        self.model = YOLOX(str(model), model_input_size=(640, 640),
                           backend="onnxruntime", device="cuda")
        require_cuda_provider(self.model)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        image, ratio = self.model.preprocess(bgr)
        outputs = self.model.inference(image)[0]
        if outputs.ndim != 3 or outputs.shape[0] != 1 or outputs.shape[-1] != 5:
            raise ValueError(f"Unexpected YOLOX export output: {outputs.shape}")
        # This export includes NMS. RTMLib's __call__ path hard-codes 0.3 for
        # such exports, so enforce the recorded confidence threshold directly.
        detections = outputs[0]
        keep = detections[:, 4] >= self.provenance["score_threshold"]
        boxes = np.asarray(detections[keep, :4] / ratio, dtype=np.float32).reshape(-1, 4)
        if not np.isfinite(boxes).all():
            raise ValueError("Person detector returned nonfinite boxes")
        height, width = bgr.shape[:2]
        boxes = np.clip(boxes, [0,0,0,0], [width,height,width,height]).astype(np.float32)
        usable = np.all(boxes[:,2:] > boxes[:,:2], axis=1)
        self.last_scores = detections[keep,4][usable].copy()
        boxes = boxes[usable]
        return boxes


class RTMW:
    topology = "coco_wholebody_133"
    point_count = 133

    def __init__(self, model: Path):
        self.provenance = {"model": "rtmw_l_384x288", "topology": self.topology,
                           "checkpoint_sha256": checked_model(model, RTMW_SHA256),
                           "runtime": "onnxruntime_cuda", "rtmlib": version("rtmlib"),
                           "onnxruntime_gpu": version("onnxruntime-gpu"),
                           "input_size_wh": [288, 384], "input_color": "RGB",
                           "bbox_padding": 1.25, "flip_test": False,
                           "crop_boundary_margin_input_px": 2.0,
                           "confidence_kind": "simcc_response_not_calibrated",
                           "cuda": require_cuda()}
        from rtmlib import RTMPose

        self.model = RTMPose(str(model), model_input_size=(288, 384),
                             to_openpose=False, backend="onnxruntime", device="cuda")
        require_cuda_provider(self.model)
        shape = self.model.session.get_inputs()[0].shape
        if tuple(shape[-2:]) != (384, 288):
            raise ValueError(f"Unexpected RTMW input shape: {shape}")

    def __call__(self, bgr: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The export's Normalize transform requires to_rgb=true. RTMLib 0.0.16
        # normalizes its input directly, so perform this conversion explicitly.
        crop, center, scale = preprocess_pose_crop(self.model,bgr,bbox)
        outputs = self.model.inference(crop)
        points, scores = self.model.postprocess(outputs, center, scale)
        if points.shape != (1, self.point_count, 2) or scores.shape != (1, self.point_count):
            raise ValueError(f"Unexpected RTMW output: {points.shape}, {scores.shape}")
        self.last_crop_valid = crop_interior_mask(points[0], center, scale, (288, 384))
        return points[0], scores[0]

    def batch(self, images:list[np.ndarray], boxes:list):
        return simcc_batch(self.model,images,boxes,(288,384),self.point_count)


def udp_crop_geometry(bbox: np.ndarray, size_wh: tuple[int, int], padding: float = 1.25
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-rotation UDP crop with the model's aspect ratio and explicit inverse."""
    box = np.asarray(bbox, dtype=np.float32)
    size = np.asarray(size_wh, dtype=np.float32)
    if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
            or np.any(size <= 1) or not np.isfinite(padding) or padding <= 0):
        raise ValueError("Invalid pose crop geometry")
    center = (box[:2] + box[2:]) / 2
    scale = (box[2:] - box[:2]) * padding
    ratio = size[0] / size[1]
    scale = np.array([max(scale[0], scale[1] * ratio),
                      max(scale[1], scale[0] / ratio)], dtype=np.float32)
    factor = (size - 1) / scale
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0], matrix[1, 1] = factor
    matrix[:, 2] = (scale / 2 - center) * factor
    return matrix, center, scale


class Sapiens:
    topology = "coco_wholebody_133"
    point_count = 133
    size_wh = (768, 1024)

    def __init__(self, model: Path, decoder_source: Path):
        import importlib.util
        import torch

        self.provenance = {"model": "sapiens_1b_coco_wholebody_133",
                           "topology": self.topology,
                           "checkpoint_sha256": checked_model(model, SAPIENS_133_SHA256),
                           "decoder_sha256": checked_model(decoder_source, SAPIENS_DECODER_SHA256),
                           "source_commit": "2cb07227a740cf09896309ea3a3b8fa44429865c",
                           "license": "CC-BY-NC-4.0", "runtime": "torchscript_cuda",
                           "precision": "float32", "input_size_wh": list(self.size_wh),
                           "input_color": "RGB", "bbox_padding": 1.25,
                           "preprocessing": "native_config_model_aspect_udp",
                           "flip_test": False,
                           "crop_boundary_margin_input_px": 2.0,
                           "confidence_kind": "heatmap_response_not_calibrated",
                           "cuda": require_cuda()}
        # The exact decoder is retained privately with Meta's license; no model
        # code or weights are silently fetched or executed from an arbitrary URL.
        spec = importlib.util.spec_from_file_location("qub_sapiens_decoder", decoder_source)
        decoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(decoder)
        self.decode = decoder.udp_decode
        self.model = torch.jit.load(str(model), map_location="cpu").eval().to("cuda")

    def __call__(self, bgr: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        import torch

        matrix, center, scale = udp_crop_geometry(bbox, self.size_wh)
        crop = cv2.warpAffine(bgr, matrix, self.size_wh, flags=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(crop[..., ::-1], dtype=np.float32)
        normalized = (rgb - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array(
            [58.395, 57.12, 57.375], dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)
        with torch.inference_mode():
            heatmaps = self.model(tensor.to("cuda"))
        if tuple(heatmaps.shape) != (1, self.point_count, 256, 192):
            raise ValueError(f"Unexpected Sapiens heatmap shape: {heatmaps.shape}")
        maps = heatmaps[0].float().cpu().numpy()
        points, scores = self.decode(maps, np.array(self.size_wh), (192, 256))
        points = points[0] / self.size_wh * scale + center - scale / 2
        self.last_crop_valid = crop_interior_mask(points, center, scale, self.size_wh)
        return points.astype(np.float32), scores[0]
