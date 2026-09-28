import cv2
import numpy as np

from preprocessing.ul_ur_landmarks.propagate_reference import flow_step


def test_flow_tracks_known_translation_and_does_not_revive_missing_point():
    previous=np.zeros((100,120),dtype=np.uint8)
    cv2.rectangle(previous,(35,30),(55,50),255,-1)
    following=cv2.warpAffine(previous,np.float32([[1,0,5],[0,1,3]]),(120,100))
    points,valid,error=flow_step(previous,following,np.float32([[35,30],[0,0]]),np.array([True,False]))
    assert valid.tolist()==[True,False]
    assert np.linalg.norm(points[0]-[40,33])<.1
    assert error[0]<.1
