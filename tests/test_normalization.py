import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.normalize import normalize_clip, verify_preserved
from preprocessing.ul_ur_landmarks.normalization import PIXEL_DATASETS, normalized_name
from preprocessing.ul_ur_landmarks.schema import Frame, Hand, ObjectDetection, sha256, validate_hdf5, write_hdf5


def fixture_job(tmp_path, empty_hands=False):
    source = {'sha256': 's'*64, 'width': 200, 'height': 100, 'declared_frames': 2,
              'pair_id': 'T/P01-T-0_1', 'pid': 'P01', 'view': 'CAM_UL',
              'relpath': 'T/p01-CAM_UL-T-0_1.mp4'}
    pose = np.tile([50., 25.], (133, 1));pose[0] = [0, 0];pose[1] = [-10, 110]
    hand = Hand(np.tile([100., 50.], (21, 1)), np.ones(21), actor='other_actor',
                bbox_xyxy=np.array([50, 10, 150, 80]), track_id=7)
    obj = ObjectDetection(np.array([0, 20, 100, 90]), .8, 2, 'block', 'fixture')
    frames = [Frame(0, pose_xy=pose, hands=[] if empty_hands else [hand], objects=[obj],
                    person_bbox_xyxy=np.array([0, 0, 200, 100])), Frame(50)]
    archive=tmp_path/'archive.h5';target=tmp_path/'canonical.h5'
    write_hdf5(archive,frames,source=source,model={},pose_topology='coco_wholebody_133',pose_count=133,schema_version='1.1')
    target.write_bytes(archive.read_bytes())
    row={'pair_id':source['pair_id'],'pid':'P01','views':{'CAM_UL':source}}
    record={'output_sha256':sha256(archive),'pair_id':source['pair_id'],'view':'CAM_UL'}
    return (row,'CAM_UL',record,target,archive,'c'*64)


@pytest.mark.parametrize('empty_hands', [False, True])
def test_normalization_preserves_all_originals_and_safe_rerun(tmp_path,empty_hands):
    job=fixture_job(tmp_path,empty_hands)
    original=job[4].read_bytes();result=normalize_clip(job)
    assert result['normalization_action']=='written'
    assert job[4].read_bytes()==original
    with h5py.File(job[3]) as h:
        np.testing.assert_allclose(h['participant/pose/xy_norm'][0,:2],[[0,0],[-.05,1.1]])
        assert np.isnan(h['participant/pose/xy_norm'][1]).all()
        assert h['participant/pose/valid'][0,0]
        np.testing.assert_allclose(h['objects/bbox_xyxy_norm'][0],[0,.2,.5,.9])
        assert np.isnan(h['participant/bbox_xyxy_norm'][1]).all()
        assert h['hands/xy_norm'].shape==(0 if empty_hands else 1,21,2)
        assert h.attrs['schema_version']=='1.1'
        assert h.attrs['normalization_version']=='1.0'
    before=job[3].stat().st_mtime_ns
    again=normalize_clip(job)
    assert again['normalization_action']=='already_valid'
    assert again['output_sha256']==result['output_sha256']
    assert job[3].stat().st_mtime_ns==before


def test_conflicting_input_is_not_overwritten(tmp_path):
    job=fixture_job(tmp_path)
    with h5py.File(job[3],'r+') as h:h['hands/track_id'][0]=999
    before=job[3].read_bytes()
    with pytest.raises(ValueError,match='differs from the validated pixel input'):normalize_clip(job)
    assert job[3].read_bytes()==before


def test_corrupt_normalized_or_preserved_values_fail(tmp_path):
    job=fixture_job(tmp_path);normalize_clip(job)
    with h5py.File(job[3],'r+') as h:h['participant/pose/xy_norm'][0,0]=[.5,.5]
    with pytest.raises(ValueError,match='Normalized coordinates differ'):
        validate_hdf5(job[3],source_sha256='s'*64,model={})
    with h5py.File(job[3],'r+') as h:
        h['participant/pose/xy_norm'][0,0]=[0,0]
        h['hands/track_id'][0]=999
    with pytest.raises(ValueError,match='Original dataset changed'):verify_preserved(job[4],job[3])


def test_partial_extension_and_wrong_dimensions_fail(tmp_path):
    job=fixture_job(tmp_path);normalize_clip(job)
    with h5py.File(job[3],'r+') as h:del h[normalized_name(PIXEL_DATASETS[0])]
    with pytest.raises(ValueError,match='Incomplete'):
        validate_hdf5(job[3],source_sha256='s'*64,model={})
    job[3].write_bytes(job[4].read_bytes())
    with h5py.File(job[3],'r+') as h:
        source=json.loads(h.attrs['source_json']);source['width']=0
        h.attrs['source_json']=json.dumps(source)
    # Invalid dimensions must fail even before arithmetic can produce infinities.
    from preprocessing.ul_ur_landmarks.normalization import add_normalized
    with h5py.File(job[3],'r+') as h:
        with pytest.raises(ValueError,match='positive source dimensions'):
            add_normalized(h,input_sha256='a'*64,archive=job[4],code_sha256='c'*64)
