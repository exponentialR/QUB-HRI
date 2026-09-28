"""MediaPipe Tasks baseline, face mesh, and independent hand pass."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import numpy as np

from .schema import Frame, Hand, sha256
from .tracking import HandTracker, assign_actor


def normalized_points(landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray([(point.x * width, point.y * height) for point in landmarks], dtype=np.float32)
    score = np.asarray([1.0 if getattr(point, "visibility", None) is None else point.visibility
                        for point in landmarks], dtype=np.float32)
    in_frame = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    score[~in_frame] = 0
    xy[~in_frame] = np.nan
    return xy, score


class MediaPipeBackend:
    pose_topology = "mediapipe_pose_33"
    pose_count = 33

    def __init__(self, models_dir: Path, width: int, height: int, *, separate_face: bool = True,
                 face_crop: str = "full") -> None:
        import mediapipe as mp

        self.mp = mp
        self.width = width
        self.height = height
        if face_crop not in {"full", "focused", "pose_guided"}:
            raise ValueError(f"Unknown face crop: {face_crop}")
        self.face_crop = face_crop
        self.face_rect = ((0, 0, width, height) if face_crop == "full" else
                          (int(.22 * width), 0, int(.78 * width), int(.60 * height)))
        self.stack = ExitStack()
        self.tracker = HandTracker(width, height)
        vision = mp.tasks.vision
        mode = vision.RunningMode.VIDEO
        holistic_path = models_dir / "holistic_landmarker.task"
        hand_path = models_dir / "hand_landmarker.task"
        face_path = models_dir / "face_landmarker.task"
        needed = [holistic_path, hand_path] + ([face_path] if separate_face else [])
        for path in needed:
            if not path.is_file():
                raise FileNotFoundError(f"Download the official task bundle first: {path}")
        self.holistic = self.stack.enter_context(vision.HolisticLandmarker.create_from_options(
            vision.HolisticLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(holistic_path)),
                                              running_mode=mode)))
        self.hand = self.stack.enter_context(vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
                                          running_mode=mode, num_hands=4)))
        face_mode = vision.RunningMode.IMAGE if face_crop == "pose_guided" else mode
        self.face = self.stack.enter_context(vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(face_path)),
                                          running_mode=face_mode, num_faces=1))) if separate_face else None
        self.provenance = {"name": "mediapipe_tasks_baseline", "version": mp.__version__,
                           "models": {path.name: sha256(path) for path in needed},
                           "face_source": "face_landmarker" if separate_face else "holistic_landmarker",
                           "hand_source": "hand_landmarker", "num_hands": 4}
        if face_crop == "focused":
            self.provenance["face_crop_to_frame"] = {"rect_xyxy_px": list(self.face_rect),
                                                      "transform": "x_frame=x_crop+x0; y_frame=y_crop+y0"}
        elif face_crop == "pose_guided":
            self.provenance["face_crop_to_frame"] = {
                "rule": "square centered at pose nose, y shifted up by 0.15 shoulder width; side=1.8 shoulder width, minimum 100 px",
                "pose_indices": [0, 11, 12], "fallback": "focused fixed crop",
                "face_running_mode": "IMAGE", "transform": "x_frame=x_crop+x0; y_frame=y_crop+y0"}

    def _pose_face_rect(self, pose_xy: np.ndarray | None) -> tuple[int, int, int, int]:
        if pose_xy is None or not np.isfinite(pose_xy[[0, 11, 12]]).all():
            return self.face_rect
        shoulder = float(np.linalg.norm(pose_xy[11] - pose_xy[12]))
        side = min(max(100, round(1.8 * shoulder)), self.width, self.height)
        nose_x, nose_y = pose_xy[0]
        x0 = max(0, min(round(nose_x - side / 2), self.width - side))
        y0 = max(0, min(round(nose_y - .15 * shoulder - side / 2), self.height - side))
        return x0, y0, x0 + side, y0 + side

    def close(self) -> None:
        self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> Frame:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        holistic = self.holistic.detect_for_video(image, timestamp_ms)
        independent_hands = self.hand.detect_for_video(image, timestamp_ms)
        pose_xy = pose_score = face_xy = face_score = None
        participant_hands = []
        participant_wrists = []
        if holistic.pose_landmarks:
            pose_xy, pose_score = normalized_points(holistic.pose_landmarks, self.width, self.height)
            if len(pose_xy) != 33:
                raise ValueError(f"Unexpected holistic pose count: {len(pose_xy)}")
            participant_wrists = [pose_xy[index] for index in (15, 16)]
        for points in (holistic.left_hand_landmarks, holistic.right_hand_landmarks):
            if points:
                xy, _ = normalized_points(points, self.width, self.height)
                participant_hands.append(xy[0])
        face_rect = self._pose_face_rect(pose_xy) if self.face_crop == "pose_guided" else self.face_rect
        face_result = None
        if self.face:
            if self.face_crop == "full":
                face_image = image
            else:
                x0, y0, x1, y1 = face_rect
                face_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                           data=np.ascontiguousarray(rgb[y0:y1, x0:x1]))
            face_result = (self.face.detect(face_image) if self.face_crop == "pose_guided" else
                           self.face.detect_for_video(face_image, timestamp_ms))
        face_landmarks = (face_result.face_landmarks[0] if face_result and face_result.face_landmarks
                          else holistic.face_landmarks if self.face is None else None)
        if face_landmarks:
            x0, y0, x1, y1 = face_rect
            face_xy, face_score = normalized_points(face_landmarks, x1 - x0, y1 - y0)
            face_xy += np.asarray([x0, y0], dtype=np.float32)
            if len(face_xy) != 478:
                raise ValueError(f"Expected 478 face points, got {len(face_xy)}")
        hands = []
        for index, landmarks in enumerate(independent_hands.hand_landmarks):
            xy, scores = normalized_points(landmarks, self.width, self.height)
            categories = independent_hands.handedness[index]
            handedness = categories[0].category_name.lower() if categories else "unknown"
            if handedness not in {"left", "right"}:
                handedness = "unknown"
            actor = assign_actor(xy, participant_hands, participant_wrists, self.width, self.height)
            hands.append(Hand(xy=xy, confidence=scores, actor=actor, handedness=handedness))
        self.tracker.update(hands)
        return Frame(timestamp_ms=float(timestamp_ms), pose_xy=pose_xy,
                     pose_confidence=pose_score, face_xy=face_xy,
                     face_confidence=face_score, face_crop_rect=face_rect, hands=hands)
