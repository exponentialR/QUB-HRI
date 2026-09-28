from preprocessing.ul_ur_landmarks.hand_models import suppress_boxes, tile_rectangles
from preprocessing.ul_ur_landmarks.hand_reference import score


def test_tiles_cover_image_edges_and_deduplicate_small_frame():
    assert tile_rectangles(320, 240) == [(0, 0, 320, 240)]
    tiles = tile_rectangles(1728, 972)
    assert tiles[0] == (0, 0, 1728, 972)
    assert (864, 108, 1728, 972) in tiles
    assert all(0 <= x0 < x1 <= 1728 and 0 <= y0 < y1 <= 972 for x0,y0,x1,y1 in tiles)


def test_tile_nms_preserves_two_hands_and_best_duplicate():
    first = {'bbox_xyxy_px': [0,0,20,20], 'confidence': .9}
    second = {'bbox_xyxy_px': [1,1,21,21], 'confidence': .5}
    third = {'bbox_xyxy_px': [25,0,45,20], 'confidence': .7}
    assert suppress_boxes([second, third, first]) == [first, third]


def test_unknown_actor_is_not_a_correct_assignment_and_ambiguous_is_excluded():
    frame = {'id':'a','view':'CAM_UL','hands_checked':True,
             'hands':[{'bbox':[10,20,20,30],'actor':'other_actor'}]}
    ambiguous = dict(frame, id='b', hand_annotation_provenance={'needs_review':True})
    detection = {'bbox_xyxy_px':[10,20,30,50], 'actor':'unknown', 'valid':[True]*21}
    metrics = score([frame,ambiguous], [{'id':'a','detections':[detection]}, {'id':'b','detections':[]}])
    result = metrics['views']['CAM_UL']
    assert result['frames'] == 1
    assert result['other_actor_box_recall'] == 1
    assert result['actor_accuracy_matched'] == 0
