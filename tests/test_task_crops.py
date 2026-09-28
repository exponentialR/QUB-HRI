import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.task_crops import face_rect_from_wholebody,face_rotation_from_wholebody,square_rect


def test_square_crop_clips_to_image_and_never_uses_missing_face():
    assert square_rect([0,0,10,10],100,80)==(0,0,80,80)
    assert square_rect([90,70,100,80],100,80)==(20,0,100,80)
    points=np.full((133,2),np.nan)
    assert face_rect_from_wholebody(points,100,80) is None
    points[23:33]=[50,30]
    assert face_rect_from_wholebody(points,100,80) is None
    points[23]=[40,20]
    assert face_rect_from_wholebody(points,100,80)==(5,0,85,80)
    with pytest.raises(ValueError,match='133'):
        face_rect_from_wholebody(np.zeros((33,2)),100,80)


def test_eye_alignment_uses_named_groups_and_missing_eye_stays_unrotated():
    xy=np.full((133,2),np.nan)
    xy[59:65]=[10,20]
    assert face_rotation_from_wholebody(xy)==0
    xy[65:71]=[30,40]
    assert face_rotation_from_wholebody(xy)==pytest.approx(45)
