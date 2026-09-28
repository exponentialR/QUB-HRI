"""MediaPipe image-mode passes on explicit crops, with source-pixel coordinates."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .mediapipe_backend import normalized_points
from .schema import Hand, sha256


def square_rect(box, width: int, height: int, padding: float = 1.8, minimum: int = 80) -> tuple[int,int,int,int]:
    box=np.asarray(box,dtype=float)
    if (box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]) or
            min(width,height,minimum)<1 or not np.isfinite(padding) or padding<=0):
        raise ValueError('Invalid crop geometry')
    size=min(width,height,max(minimum,round(max(box[2:]-box[:2])*padding)))
    center=(box[2:]+box[:2])/2
    x=max(0,min(width-size,round(center[0]-size/2)))
    y=max(0,min(height-size,round(center[1]-size/2)))
    return x,y,x+size,y+size


def face_rect_from_wholebody(xy: np.ndarray | None, width: int, height: int) -> tuple | None:
    """Use the explicitly named COCO WholeBody face subset, requiring available points."""
    if xy is None:
        return None
    xy=np.asarray(xy)
    if xy.shape!=(133,2):
        raise ValueError('Expected COCO WholeBody 133 coordinates')
    face=xy[23:91]
    good=face[np.isfinite(face).all(axis=1)]
    if len(good)<10:
        return None
    lower,upper=good.min(axis=0),good.max(axis=0)
    if np.any(upper<=lower):
        return None
    return square_rect([*lower,*upper],width,height,padding=1.8)


def face_rotation_from_wholebody(xy: np.ndarray) -> float:
    """Align the two iBUG eye groups within the named COCO WholeBody face subset."""
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (133,2):
        raise ValueError('Expected COCO WholeBody 133 coordinates')
    centers = []
    for group in (xy[59:65], xy[65:71]):
        points = group[np.isfinite(group).all(axis=1)]
        if len(points) < 3:
            return 0.
        centers.append(points.mean(axis=0))
    left, right = sorted(centers, key=lambda p: p[0])
    difference = right - left
    if np.linalg.norm(difference) < 3:
        return 0.
    return float(np.degrees(np.arctan2(difference[1],difference[0])))


class CropTasks:
    def __init__(self,models_dir:Path,*,face:bool=True,hands:bool=True):
        import mediapipe as mp
        self.mp=mp
        self.stack=ExitStack()
        vision=mp.tasks.vision
        mode=vision.RunningMode.IMAGE
        self.provenance={'runtime':'mediapipe_tasks_cpu','version':mp.__version__,'mode':'IMAGE',
                         'face_crop':'COCO WholeBody face bbox, square padding 1.8, minimum 80 pixels',
                         'hand_crop':'detector bbox, square padding 2.0, minimum 80 pixels',
                         'coordinate_transform':'face uses recorded inverse affine; hand crops add ROI origin',
                         'landmark_confidence_kind':'presence_proxy_not_calibrated',
                         'handedness_policy':'unknown; unmirrored camera convention not independently verified',
                         'models':{}}
        self.face=self.hand=None
        if face:
            path=models_dir/'face_landmarker.task'
            self.provenance['models'][path.name]=sha256(path)
            self.face=self.stack.enter_context(vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                                              running_mode=mode,num_faces=1)))
        if hands:
            path=models_dir/'hand_landmarker.task'
            self.provenance['models'][path.name]=sha256(path)
            self.hand=self.stack.enter_context(vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                                              running_mode=mode,num_hands=4)))

    def _image(self,bgr,rect):
        x0,y0,x1,y1=rect
        rgb=np.ascontiguousarray(bgr[y0:y1,x0:x1,::-1])
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb)

    def face_points(self,bgr,rect,rotation_degrees: float = 0.):
        if rect is None:
            return None,None
        if self.face is None:
            raise ValueError('Face task is disabled')
        import cv2
        x0,y0,x1,y1=rect
        crop = np.ascontiguousarray(bgr[y0:y1,x0:x1,::-1])
        matrix = cv2.getRotationMatrix2D(((x1-x0)/2,(y1-y0)/2),rotation_degrees,1.)
        if rotation_degrees:
            crop = cv2.warpAffine(crop,matrix,(x1-x0,y1-y0),flags=cv2.INTER_LINEAR)
        self.last_face_crop_to_source = np.vstack([cv2.invertAffineTransform(matrix),[0,0,1]])
        self.last_face_crop_to_source[:2,2] += [x0,y0]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop))
        result=self.face.detect(image)
        if not result.face_landmarks:
            return None,None
        x0,y0,x1,y1=rect
        xy,score=normalized_points(result.face_landmarks[0],x1-x0,y1-y0)
        if xy.shape!=(478,2):
            raise ValueError('Unexpected dense face topology')
        xy = (np.column_stack([xy,np.ones(len(xy))]) @ self.last_face_crop_to_source.T)[:,:2].astype(np.float32)
        good = (np.isfinite(xy).all(axis=1) & (xy[:,0]>=0) & (xy[:,0]<bgr.shape[1]) &
                (xy[:,1]>=0) & (xy[:,1]<bgr.shape[0]))
        xy[~good] = np.nan
        score[~good] = 0
        return xy,score

    def hand_points(self,bgr,rect,*,actor='unknown'):
        if self.hand is None:
            raise ValueError('Hand task is disabled')
        result=self.hand.detect(self._image(bgr,rect))
        x0,y0,x1,y1=rect
        hands=[]
        for points in result.hand_landmarks:
            xy,score=normalized_points(points,x1-x0,y1-y0)
            if xy.shape!=(21,2):
                raise ValueError('Unexpected hand topology')
            xy += [x0,y0]
            hands.append(Hand(xy,score,actor=actor,handedness='unknown'))
        return hands

    def close(self):
        self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
