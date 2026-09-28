"""Validation at the boundary between GPU models and source-frame coordinates."""

import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.pose_models import PersonDetector, crop_interior_mask, require_cuda_provider, udp_crop_geometry, visible_points


def test_model_missingness_preserves_valid_origin_and_raw_scores() -> None:
    xy = np.array([[0, 0], [20, 30], [100, 10], [np.nan, 4], [5, 5]])
    scores = np.array([.8, .1, .9, .8, np.inf])
    points, raw, valid = visible_points(xy, scores, 100, 80, .3)
    assert valid.tolist() == [True, False, False, False, False]
    assert points[0].tolist() == [0, 0]
    assert np.isnan(points[1:]).all()
    assert raw[1] == pytest.approx(.1)
    assert xy[1].tolist() == [20, 30]


def test_gpu_session_must_not_silently_fall_back_to_cpu() -> None:
    class Session:
        def get_providers(self):
            return ["CPUExecutionProvider"]
    class Model:
        session = Session()
    with pytest.raises(RuntimeError, match="fell back to CPU"):
        require_cuda_provider(Model())


def test_udp_crop_round_trip_at_boundaries_and_center() -> None:
    matrix, center, scale = udp_crop_geometry(np.array([100, 50, 300, 450]), (768, 1024))
    source = np.array([center - scale / 2, center, center + scale / 2])
    transformed = np.column_stack([source, np.ones(3)]) @ matrix.T
    assert np.allclose(transformed, [[0, 0], [383.5, 511.5], [767, 1023]], atol=1e-4)
    restored = transformed / [767, 1023] * scale + center - scale / 2
    assert np.allclose(restored, source, atol=1e-4)
    assert scale[0] / scale[1] == pytest.approx(768 / 1024)
    with pytest.raises(ValueError, match="Invalid pose crop"):
        udp_crop_geometry(np.array([10, 10, 10, 20]), (768, 1024))


def test_person_detection_applies_recorded_threshold_and_resize_inverse() -> None:
    from types import SimpleNamespace

    detector = object.__new__(PersonDetector)
    detector.provenance = {"score_threshold": .7}
    detections = np.array([[[10, 20, 30, 40, .9], [50, 60, 80, 90, .5]]], dtype=np.float32)
    detector.model = SimpleNamespace(preprocess=lambda image: (image, .5),
                                     inference=lambda image: [detections])
    assert detector(np.zeros((100, 100, 3))).tolist() == [[20, 40, 60, 80]]


def test_crop_edge_saturation_is_separate_from_frame_validity() -> None:
    # The last point is inside the source frame but saturates the model crop.
    points = np.array([[200, 300], [200, 699]])
    center, scale = np.array([200, 300]), np.array([600, 800])
    assert visible_points(points, np.ones(2), 1000, 1000, .3)[2].tolist() == [True, True]
    assert crop_interior_mask(points, center, scale, (288, 384)).tolist() == [True, False]
def test_seated_participant_selection_rejects_larger_lower_actor():
    from preprocessing.ul_ur_landmarks.pose_models import select_participant_box
    boxes = np.array([[650,0,1000,400],[900,530,1728,972]])
    assert select_participant_box(boxes,972) == 0
    assert select_participant_box(boxes[1:],972) is None
    assert select_participant_box(np.empty((0,4)),972) is None


def test_crop_before_rgb_is_identical_to_full_image_rgb_preprocessing():
    from rtmlib.tools.pose_estimation.rtmpose import RTMPose
    from preprocessing.ul_ur_landmarks.pose_models import preprocess_pose_crop
    tool=object.__new__(RTMPose)
    tool.model_input_size=(288,384)
    tool.mean=(123.675,116.28,103.53)
    tool.std=(58.395,57.12,57.375)
    image=np.random.default_rng(7).integers(0,256,(500,700,3),dtype=np.uint8)
    box=[120,80,480,420]
    expected=tool.preprocess(np.ascontiguousarray(image[...,::-1]),box)
    actual=preprocess_pose_crop(tool,image,box)
    assert all(np.array_equal(a,b) for a,b in zip(actual,expected))
