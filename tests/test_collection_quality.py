import numpy as np
import json
import pytest

from preprocessing.ul_ur_landmarks.collection_quality import frame_flags, validated_outputs
from preprocessing.ul_ur_landmarks.schema import ACTORS


def test_collection_flags_keep_missing_predictions_separate_from_jumps_and_extra_hands():
    xy=np.full((3,133,2),20.)
    valid=np.ones((3,133),bool)
    valid[0,:11]=False
    xy[0,:11]=np.nan
    xy[2,9]=[500,500]
    face=np.zeros((3,478),bool)
    face[0]=True
    flags=frame_flags(xy,valid,face,np.array([1,1,1,2]),np.array([ACTORS['other_actor']]*3+[ACTORS['participant']]),1000,800)
    assert flags['no_body_points']==[0]
    assert flags['abrupt_body_point_change']==[2]
    assert flags['dense_face_missing_with_coarse_face']==[1,2]
    assert flags['more_than_two_other_actor_hands']==[1]
    assert flags['more_than_two_participant_hands']==[]


def test_final_audit_requires_complete_unique_successful_validation(tmp_path):
    path=tmp_path/'validation.jsonl'
    expected={('task/P01','CAM_UL'):'P01',('task/P01','CAM_UR'):'P01'}
    rows=[{'pair_id':pair,'view':view,'status':'validated','output_sha256':'a'*64} for pair,view in expected]
    path.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError,match='every expected'):validated_outputs(path,expected)
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    assert set(validated_outputs(path,expected))==set(expected)
    for changed in [rows+[rows[0]],[rows[0],{**rows[1],'status':'failed'}],
                    [rows[0],{**rows[1],'pair_id':'outside'}]]:
        path.write_text('\n'.join(json.dumps(row) for row in changed))
        with pytest.raises(ValueError,match='duplicate, failed or out-of-scope'):validated_outputs(path,expected)
