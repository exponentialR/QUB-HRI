import numpy as np

from preprocessing.ul_ur_landmarks.compare_reference import point_score, summarize
from preprocessing.ul_ur_landmarks.hand_quality import match_boxes
from preprocessing.ul_ur_landmarks.hand_pose_variants import rotated_crop
from preprocessing.ul_ur_landmarks.object_quality import score_frame as score_objects
from preprocessing.ul_ur_landmarks.collection_backend import merge_object_detections
from preprocessing.ul_ur_landmarks.collection_backend import filter_hand_proposals


def test_missing_predictions_count_against_coverage_and_threshold_metrics():
    points = [point_score([0, 0], [0, 0], True, 100),
              point_score([20, 0], [np.nan, np.nan], True, 100),
              point_score([20, 0], [20, 0], False, 100)]
    result = summarize(points)
    assert result["labelled_points"] == 3
    assert result["available_points"] == 1
    assert result["fraction_within_30px_including_missing"] == 1 / 3
    assert result["median_error_px_when_available"] == 0


def test_pixels_and_image_diagonal_normalization_are_separate():
    scored = point_score([0, 0], [30, 40], True, 1000)
    assert scored["error_px"] == 50
    assert scored["error_by_image_diagonal"] == .05
    assert summarize([scored])["fraction_within_0_03_image_diagonal_including_missing"] == 0


def test_box_matching_keeps_observations_without_joints_and_ignores_actor():
    truth = [{"bbox": [10, 10, 20, 20], "actor": "other_actor"}]
    predictions = [{"box": np.array([10, 10, 30, 30]), "actor": "participant",
                    "valid": np.zeros(21, dtype=bool)}]
    assert match_boxes(truth, predictions) == {0: 0}
    assert match_boxes(truth * 2, predictions) == {0: 0}
    assert match_boxes(truth, [{"box": None}]) == {}


def test_rotated_hand_crop_maps_actual_pixels_back_to_source():
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    image[70, 130] = [11, 23, 47]
    for rotation in range(4):
        crop, box, inverse = rotated_crop(image, [110, 50, 150, 100], rotation, 1.)
        y, x = np.argwhere((crop == [11, 23, 47]).all(axis=-1))[0]
        assert np.allclose(inverse @ [x, y, 1], [130, 70, 1])
        assert np.all(box[2:] > box[:2])


def test_object_audit_does_not_count_unlabelled_assembly_components_as_false_boxes():
    frame = {"roi_xyxy_px": [0, 0, 400, 400], "ignore_regions_xyxy_px": [], "objects": [
        {"group": "loose_brick", "bbox_xyxy_px": [10, 10, 30, 30], "ignore_for_box_score": False},
        {"group": "assembly", "bbox_xyxy_px": [100, 100, 200, 200], "ignore_for_box_score": False}]}
    pred = [{"class_name": "two_two_block", "box": box} for box in
            ([10, 10, 30, 30], [120, 120, 140, 140], [300, 300, 320, 320])]
    pred.append({"class_name": "assembly_base", "box": [100, 100, 200, 200]})
    scores, _ = score_objects(frame, pred, .5)
    assert scores["loose_brick"] == {"labels": 1, "matched": 1, "scored_predictions": 2,
                                       "unmatched_predictions": 1, "unassessed_component_predictions": 1}
    assert scores["assembly"]["matched"] == 1


def test_object_crop_restore_and_class_aware_duplicate_removal():
    full = [{'bbox': np.array([10,110,30,130]), 'class_id': 0, 'confidence': .7}]
    crop = [{'bbox': np.array([11,11,31,31]), 'class_id': 0, 'confidence': .8},
            {'bbox': np.array([11,11,31,31]), 'class_id': 1, 'confidence': .6}]
    merged = merge_object_detections(full,crop,100)
    assert len(merged) == 2
    assert merged[0]['confidence'] == .8
    assert np.array_equal(merged[0]['bbox'],[11,111,31,131])
    assert np.array_equal(crop[0]['bbox'],[11,11,31,31])


def test_hand_suppression_keeps_distinct_touching_hands_and_enforces_actor_capacity():
    def proposal(box,score,actor):
        return {'bbox':np.array(box),'confidence':score,'class_name':actor}
    rows=[proposal([10,10,50,50],.9,'surrogate_hand'),
          proposal([12,12,52,52],.8,'surrogate_hand'),
          proposal([40,10,80,50],.7,'surrogate_hand'),
          proposal([100,10,140,50],.9,'participant_hand'),
          proposal([120,10,160,50],.8,'participant_hand'),
          proposal([200,10,220,30],.4,'participant_hand')]
    kept=filter_hand_proposals(rows,{'surrogate_hand':.3,'participant_hand':.7},2)
    assert len(kept)==4
    assert sum(p['class_name']=='surrogate_hand' for p in kept)==2
    assert any(np.array_equal(p['bbox'],[40,10,80,50]) for p in kept)
    assert all(p['confidence']>.4 for p in kept)
