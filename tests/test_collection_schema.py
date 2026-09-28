import h5py
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.schema import Frame, Hand, ObjectDetection, validate_hdf5, write_hdf5


def test_extended_observation_roundtrip_and_empty_frame(tmp_path):
    source = {'sha256':'source','width':100,'height':80,'declared_frames':2}
    model = {'name':'combined','pose_confidence_kind':'simcc_response_not_calibrated'}
    hand = Hand(np.zeros((21,2)),np.ones(21), actor='other_actor', topology='hand5_21',
                model_id='rtmpose_hand', bbox_xyxy=np.array([0,0,20,30]), detection_confidence=.7)
    obj = ObjectDetection(np.array([30,40,50,60]), .8, 2, 'assembly_base','lego_yolo')
    frames = [Frame(0,hands=[hand],objects=[obj],person_bbox_xyxy=np.array([0,0,60,70])),Frame(25)]
    path = tmp_path / 'collection.h5'
    with pytest.raises(ValueError,match='require schema 1.1'):
        write_hdf5(path,frames,source=source,model=model,pose_topology='coco_wholebody_133',pose_count=133)
    write_hdf5(path,frames,source=source,model=model,pose_topology='coco_wholebody_133',pose_count=133,schema_version='1.1')
    result=validate_hdf5(path,source_sha256='source',model=model)
    assert result['object_observations'] == 1 and result['hand_observations'] == 1
    with h5py.File(path) as data:
        assert data['hands/topology'].asstr()[0] == 'hand5_21'
        assert data['objects/class_name'].asstr()[0] == 'assembly_base'
        assert data['objects/frame_index'][0] == 0
        assert np.isnan(data['participant/bbox_xyxy_px'][1]).all()
        assert data['hands/valid'][0,0] and np.array_equal(data['hands/xy_px'][0,0],[0,0])
    with h5py.File(path,'r+') as data:
        data['objects/frame_index'][0] = 2
    with pytest.raises(ValueError,match='object frame'):
        validate_hdf5(path,source_sha256='source',model=model)


def test_empty_objects_and_hands_are_valid_but_invalid_box_is_not_published(tmp_path):
    common = dict(source={'sha256':'source','width':100,'height':80},model={},
                  pose_topology='coco_wholebody_133',pose_count=133,schema_version='1.1')
    path=tmp_path/'empty.h5'
    write_hdf5(path,[Frame(0)],**common)
    assert validate_hdf5(path,source_sha256='source',model={})['object_observations'] == 0
    bad=tmp_path/'bad.h5'
    obj=ObjectDetection(np.array([0,0,200,30]),.8,0,'block','lego')
    with pytest.raises(ValueError,match='box coordinates'):
        write_hdf5(bad,[Frame(0,objects=[obj])],**common)
    assert not bad.exists()
    assert not list(tmp_path.glob('*.partial'))
