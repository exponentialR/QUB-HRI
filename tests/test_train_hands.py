import pytest

from preprocessing.ul_ur_landmarks.train_hands import split_participants, yolo_labels


def test_participant_split_does_not_depend_on_frame_order():
    ids=[f'P{i:02d}' for i in range(20)]
    first=split_participants(ids)
    assert first==split_participants(list(reversed(ids))+ids[:5])
    assert sum(v=='val' for v in first.values())==5


def test_labels_keep_actor_roles_and_explicit_negatives():
    frame={'width':200,'height':100,'hands':[{'actor':'other_actor','bbox':[0,0,20,40]}]}
    assert yolo_labels(frame)=='1 0.05000000 0.20000000 0.10000000 0.40000000\n'
    assert yolo_labels(dict(frame,hands=[]))==''
    with pytest.raises(ValueError,match='Unknown actor'):
        yolo_labels(dict(frame,hands=[{'actor':'unknown','bbox':[0,0,20,40]}]))
    with pytest.raises(ValueError,match='Invalid reference'):
        yolo_labels(dict(frame,hands=[{'actor':'participant','bbox':[190,0,20,40]}]))
def test_extension_preserves_validation_and_assigns_new_ids_to_training():
    from preprocessing.ul_ur_landmarks.train_hands import extend_splits
    previous={'P01':'train','P02':'val'}
    assert extend_splits(['P01','P02','P03'],previous)=={'P01':'train','P02':'val','P03':'train'}
    import pytest
    with pytest.raises(ValueError,match='retain all'):
        extend_splits(['P02','P03'],previous)
