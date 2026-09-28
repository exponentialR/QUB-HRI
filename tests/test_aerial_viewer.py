import json
import threading

import cv2
import h5py
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.aerial import AerialData, normalized_box, normalized_hand
from preprocessing.ul_ur_landmarks.viewer import AerialClip, Collection


def legacy_file(path, timestamps):
    count = len(timestamps)
    with h5py.File(path, 'w') as h:
        hands = np.full((count, 21, 3), .25, dtype=np.float32)
        hands[:, 0] = 0  # genuine zero point within an otherwise detected hand
        h['left_landmarks'] = hands
        h['right_landmarks'] = np.zeros_like(hands)
        h['norm_gaze'] = np.tile([.5, .75], (count, 1))
        h['rec_bboxes'] = np.tile([[[.5, .5, .2, .4], [0, 0, 0, 0]]], (count, 1, 1))
        h['surrogate_hands'] = np.tile([[[.7, .7, .1, .1], [0, 0, 0, 0]]], (count, 1, 1))
        h['timestamps'] = np.array(timestamps, dtype=np.float32)


def test_legacy_coordinates_missingness_and_unavailable_fields(tmp_path):
    path = tmp_path/'av.h5'; legacy_file(path, [0., .05])
    data = AerialData(path, [0, 50], 200, 100)
    frame = data.frame(0)
    assert frame['hands'][0]['xy'][0] == [0., 0.]
    assert frame['hands'][0]['xy'][1] == [50., 25.]
    assert frame['hands'][0]['confidence'] == [None]*21
    assert frame['hands'][0]['track_id'] is None
    assert len(frame['hands']) == 2  # one participant hand, one surrogate box
    assert not any(frame['hands'][1]['valid'])
    assert frame['objects'][0]['bbox'] == [80., 30., 120., 70.]
    assert frame['objects'][0]['confidence'] is None
    assert frame['objects'][0]['class_name'] == 'AV detection'
    assert frame['gaze'] == [100., 75.]
    assert normalized_hand(np.zeros((21, 3)), 200, 100) is None
    assert normalized_box([0, 0, 0, 0], 200, 100) is None
    assert normalized_box([np.nan, .5, .2, .2], 200, 100) is None
    json.dumps(frame, allow_nan=False)


@pytest.mark.parametrize('timestamps,match', [([0], 'frame count'), ([0, .09], 'timestamps')])
def test_rejects_misaligned_legacy_rows(tmp_path, timestamps, match):
    path = tmp_path/'av.h5'; legacy_file(path, timestamps)
    with pytest.raises(ValueError, match=match):
        AerialData(path, [0, 50], 200, 100)


def test_av_random_seek_matches_sequential_decode(tmp_path):
    video = tmp_path/'av.mp4'; path = tmp_path/'av.h5'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 20, (96, 64))
    assert writer.isOpened()
    for i in range(4):writer.write(np.full((64, 96, 3), 40*i, np.uint8))
    writer.release(); legacy_file(path, [0, .05, .10, .15])
    cap = cv2.VideoCapture(str(video)); decoded = []
    while True:
        okay, image = cap.read()
        if not okay:break
        decoded.append(image)
    cap.release()
    clip = AerialClip(path, video)
    try:
        assert clip.times.tolist() == [0, 50, 100, 150]
        for index in (3, 0, 2):
            assert np.array_equal(clip.image(index), decoded[index])
            frame = clip.frame(index)
            assert frame['frame_index'] == index and 'image' in frame
    finally:clip.close()


def test_three_view_timeline_does_not_assume_equal_frame_rates():
    class FakeClip:
        def __init__(self, times):self.times = times
        def frame(self, index, include_image):
            return {'frame_index': index, 'timestamp_ms': self.times[index]}
    collection = Collection.__new__(Collection)
    collection.lock = threading.Lock()
    clips = {'CAM_UL': FakeClip([0, 40, 80, 120, 160, 200]),
             'CAM_UR': FakeClip([0, 40, 80, 120, 160, 200])}
    collection.clip = lambda index, view: clips[view]
    collection.aerial = lambda index: FakeClip([0, 50, 100, 150])
    assert collection.frame(0, 2)['CAM_AV']['frame_index'] == 2
    assert collection.frame(0, 4)['CAM_AV']['frame_index'] == 3
    assert collection.frame(0, 5)['CAM_AV'] is None
    collection.aerial = lambda index: None
    assert collection.frame(0, 1)['CAM_UL']['frame_index'] == 1
    assert collection.frame(0, 1)['CAM_AV'] is None
