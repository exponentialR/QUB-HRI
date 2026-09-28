"""Reference edits preserve source identity and supply training provenance."""

from copy import deepcopy
import hashlib

import pytest

from preprocessing.ul_ur_landmarks.annotate import prepare_annotations
from preprocessing.ul_ur_landmarks.train_hands import yolo_labels


@pytest.fixture
def reference(tmp_path):
    (tmp_path / 'images').mkdir()
    image = tmp_path / 'images/frame.jpg'
    image.write_bytes(b'synthetic-image-identity')
    frame = dict(id='frame', image='images/frame.jpg', pair_id='task/P01-example',
                 view='CAM_UL', frame_index=0, timestamp_ms=0, width=64, height=48,
                 hands=[], hands_checked=False, reviewed=False, body_checked=False,
                 face_checked=False, face={'visible':None, 'anchors':{}}, pose={})
    return tmp_path, {'schema_version':1, 'frames':[frame]}


@pytest.mark.parametrize('challenge', [False, True])
def test_manual_hand_labels_preserve_image_identity_and_can_be_exported(reference, challenge):
    root, old = reference
    digest = hashlib.sha256((root/'images/frame.jpg').read_bytes()).hexdigest()
    if challenge:
        old.pop('schema_version')
        old['schema'] = 'local_hand_challenge_annotations_v1'
        old['frames'][0]['image_sha256'] = digest
    saved = deepcopy(old)
    frame = saved['frames'][0]
    frame.update(hands_checked=True, hands=[{'actor':'other_actor','bbox':[10,10,20,20]}])
    prepare_annotations(old, saved, root)
    assert frame['hand_annotation_provenance']['image_sha256'] == digest
    assert frame['hand_annotation_provenance']['status'] == 'manual_visual'
    assert not frame['hand_annotation_provenance']['needs_review']
    assert yolo_labels(frame).startswith('1 ')
    assert old['frames'][0]['hands'] == []
    # Saving an unchanged checked reference retains its provenance.
    repeated = deepcopy(saved)
    prepare_annotations(saved, repeated, root)
    assert saved == repeated


def test_annotation_rejects_source_edits_and_unchecked_full_review(reference):
    root, old = reference
    for key, value in [('frame_index',1), ('image','images/other.jpg'), ('source_video_sha256','changed')]:
        new = deepcopy(old); new['frames'][0][key] = value
        with pytest.raises(ValueError, match='Immutable'):
            prepare_annotations(old, new, root)
    new = deepcopy(old); new['frames'][0]['reviewed'] = True
    with pytest.raises(ValueError, match='has not been checked'):
        prepare_annotations(old, new, root)


def test_annotation_rejects_changed_image_and_path_escape(reference):
    root, old = reference
    old['frames'][0]['image_sha256'] = 'incorrect-hash'
    new = deepcopy(old); new['frames'][0]['hands_checked'] = True
    with pytest.raises(ValueError, match='image differs'):
        prepare_annotations(old, new, root)
    old['frames'][0]['image'] = '../outside.jpg'
    new = deepcopy(old); new['frames'][0]['hands_checked'] = True
    with pytest.raises(ValueError, match='inside the local images'):
        prepare_annotations(old, new, root)


def test_checked_empty_hands_are_valid_negatives(reference):
    root, old = reference
    new = deepcopy(old); new['frames'][0]['hands_checked'] = True
    prepare_annotations(old, new, root)
    assert yolo_labels(new['frames'][0]) == ''
    assert new['frames'][0]['hand_annotation_provenance']['status'] == 'manual_visual'
