import json

import h5py
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.refine_hands import NATIVE_MODEL_ID, configuration, refine_file, refine_frame
from preprocessing.ul_ur_landmarks.schema import Frame, Hand, ObjectDetection, sha256, write_hdf5


def pose_fixture():
    pose=np.full((133,2),np.nan,dtype=np.float32)
    scores=np.zeros(133,dtype=np.float32)
    pose[91:112]=np.linspace([0,0],[20,20],21)
    scores[91:112]=5
    pose[9]=[0,0];scores[9]=5
    return pose,scores


def test_native_fallback_preserves_zero_coordinates_and_absent_detector_score():
    pose,scores=pose_fixture()
    hands=refine_frame(pose,scores,[],200,200)
    assert len(hands)==1 and hands[0].model_id==NATIVE_MODEL_ID
    assert np.array_equal(hands[0].xy[0],[0,0])
    assert hands[0].detection_confidence is None
    scores[91:112]=3.9
    assert refine_frame(pose,scores,[],200,200)==[]


def test_native_fallback_does_not_relabel_a_glove_or_duplicate_an_existing_hand():
    pose,scores=pose_fixture()
    hand=Hand(pose[91:112].copy(),scores[91:112].copy(),actor='other_actor',
              bbox_xyxy=np.array([0,0,25,25]),detection_confidence=.9)
    result=refine_frame(pose,scores,[hand],200,200)
    assert len(result)==1 and result[0].actor=='other_actor'
    assert np.array_equal(result[0].xy,hand.xy)
    assert result[0].detection_confidence==.9


def test_ambiguous_handedness_changes_without_mutating_input_points():
    pose,scores=pose_fixture()
    pose[112:133]=pose[91:112];scores[112:133]=5;pose[10]=pose[9];scores[10]=5
    hand=Hand(pose[91:112].copy(),scores[91:112].copy(),actor='participant',handedness='left',
              model_id='wholebody',bbox_xyxy=np.array([0,0,25,25]))
    result=refine_frame(pose,scores,[hand],200,200)
    assert len(result)==1 and result[0].handedness=='unknown'
    assert hand.handedness=='left' and np.array_equal(result[0].xy,hand.xy)
    assert refine_frame(pose,scores,[hand],200,200,clear_ambiguous_side=False)[0].handedness=='left'


def test_file_refinement_preserves_other_data_parent_and_safe_reruns(tmp_path):
    pose,scores=pose_fixture()
    frames=[Frame(float(i*20),pose.copy(),scores.copy(),objects=[
        ObjectDetection(np.array([50,60,70,80]),.8,3,'test_brick','test_model')]) for i in range(2)]
    model={'hand_mode':'native_hand5','hand_models':{'wholebody':{'model':'rtmw_l_384x288','checkpoint_sha256':'test'}}}
    source={'sha256':'test_source','width':200,'height':200,'declared_frames':2}
    original=tmp_path/'original.h5';output=tmp_path/'refined.h5'
    write_hdf5(original,frames,source=source,model=model,pose_count=133,
               pose_topology='coco_wholebody_133',schema_version='1.1')
    original_hash=sha256(original)
    report=refine_file(original,output)
    assert report['frames']==2 and report['added_native_observations']==2
    assert sha256(original)==original_hash
    with h5py.File(original) as a,h5py.File(output) as b:
        datasets=[]
        a.visititems(lambda name,value:datasets.append(name) if isinstance(value,h5py.Dataset) else None)
        for name in datasets:
            if name.startswith('hands/'):continue
            x,y=a[name][:],b[name][:]
            assert np.array_equal(x,y,equal_nan=True) if x.dtype.kind=='f' else np.array_equal(x,y)
        assert json.loads(b.attrs['source_json'])['parent_landmarks']['sha256']==original_hash
        assert b['hands/valid'][0,0] and np.isnan(b['hands/detection_confidence'][:]).all()
        assert np.array_equal(b['hands/track_id'][:],[0,0])
        assert b['hands/model_id'].asstr()[:].tolist()==[NATIVE_MODEL_ID]*2
    output_hash=sha256(output)
    assert refine_file(original,output)['status']=='skipped_valid'
    assert sha256(output)==output_hash
    with pytest.raises(ValueError,match='provenance'):
        refine_file(original,output,threshold=5.)
    assert sha256(output)==output_hash


def test_model_response_scale_cannot_be_silently_reused_for_sapiens():
    model={'hand_mode':'native_hand5','hand_models':{'wholebody':{'model':'sapiens_1b_coco_wholebody_133'}}}
    with pytest.raises(ValueError,match='specific to RTMW'):
        configuration(model,4.,True)
