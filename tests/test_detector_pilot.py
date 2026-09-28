"""Checks for the recovered detector's box metrics and pilot data contract."""

import pytest

from preprocessing.ul_ur_landmarks.detector_pilot import box_iou, match_boxes, score_hands


def test_box_coordinates_and_one_to_one_matches() -> None:
    assert box_iou([0, 0, 20, 10], [0, 0, 20, 10]) == 1
    assert box_iou([0, 0, 20, 10], [10, 0, 30, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    labels = [{"bbox": [0, 0, 20, 10]}, {"bbox": [0, 0, 20, 10]}]
    predictions = [{"bbox_xyxy_px": [0, 0, 20, 10], "class_name": "surrogate_hand"}]
    assert len(match_boxes(labels, predictions, .5)) == 1


def test_detection_and_actor_assignment_are_separate_and_negatives_explicit() -> None:
    frames = [
        {"id": "positive", "view": "CAM_UL", "hands_checked": True,
         "hands": [{"actor": "other_actor", "bbox": [0, 0, 20, 10]}]},
        {"id": "negative", "view": "CAM_UL", "hands_checked": True, "hands": []},
        {"id": "ambiguous", "view": "CAM_UL", "hands_checked": True, "hands": [],
         "hand_annotation_provenance": {"needs_review": True}},
        {"id": "unchecked", "view": "CAM_UL", "hands_checked": False, "hands": []},
    ]
    predictions = [
        {"id": "positive", "detections": [{"class_name": "lefthand", "bbox_xyxy_px": [0, 0, 20, 10]}]},
        {"id": "negative", "detections": [
            {"class_name": "surrogate_hand", "bbox_xyxy_px": [0, 0, 20, 10]},
            {"class_name": "two_two_block", "bbox_xyxy_px": [0, 0, 20, 10]}]},
    ]
    summary = score_hands(frames, predictions)
    row = summary["views"]["CAM_UL"]
    assert row["other_actor_recall"] == 1
    assert row["other_actor_end_to_end_recall"] == 0
    assert row["actor_accuracy_matched"] == 0
    assert row["predicted_hands"] == 2  # LEGO boxes are not hand predictions.
    assert row["false_other_actor_frame_rate"] == 1
    assert row["hand_detection_precision"] == .5
    assert set(summary["excluded_frame_ids"]) == {"ambiguous", "unchecked"}
    assert not summary["decision_ready"]
    with pytest.raises(ValueError, match="Missing prediction"):
        score_hands(frames, predictions[:1])
    with pytest.raises(ValueError, match="IoU threshold"):
        score_hands(frames, predictions, 0)
