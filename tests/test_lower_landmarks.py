import json

import cv2
import h5py
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.lower import LowerData, TOPOLOGIES, add_normalized
from preprocessing.ul_ur_landmarks.normalize_lower import build_jobs, convert
from preprocessing.ul_ur_landmarks.schema import sha256
from preprocessing.ul_ur_landmarks.viewer import DatasetCollection


def make_lower(root, view='CAM_LL'):
    source = root/'downloads'/view/'T'/f'p01-{view}-TASK-0_1.h5'
    video = root/'videos'/'T'/f'p01-{view}-TASK-0_1.mp4'
    source.parent.mkdir(parents=True, exist_ok=True); video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25, (64,48))
    assert writer.isOpened()
    for i in range(3): writer.write(np.full((48,64,3), i*40, np.uint8))
    writer.release()
    with h5py.File(source, 'w') as h:
        h.attrs['original'] = 'preserve me'
        for name, (_, count) in TOPOLOGIES.items():
            a = np.full((3,count,2), [32,24], np.float32)
            a[0,0] = [0,0]; a[0,1] = [-8,60]; a[1] = 0; a[2,2] = [np.nan,12]
            h[name] = a; h[name].attrs['original'] = 'also preserve me'
    jobs = build_jobs(root/'downloads', root/'videos', root/'landmarks')
    return source, video, {**next(j for j in jobs if j['view']==view), 'code_sha256':'c'*64}


def test_copy_normalization_roundtrip_missing_points_and_rerun(tmp_path):
    source, video, job = make_lower(tmp_path)
    before = source.read_bytes()
    result = convert(job)
    assert result['action'] == 'written' and source.read_bytes() == before
    target = tmp_path/'landmarks/T'/source.name
    with h5py.File(target) as h:
        np.testing.assert_allclose(h['pose_landmarks_norm'][0,:2], [[0,0],[-.125,1.25]])
        assert h['pose_landmarks_valid'][0,0]
        assert not h['pose_landmarks_valid'][1].any()
        assert not h['pose_landmarks_valid'][2,2]
        assert np.isnan(h['pose_landmarks_norm'][1]).all()
        assert h['pose_landmarks'][0,1].tolist() == [-8,60]
        assert h.attrs['original'] == 'preserve me'
        assert h['pose_landmarks'].attrs['original'] == 'also preserve me'
    mtime = target.stat().st_mtime_ns
    assert convert(job)['action'] == 'already_valid'
    assert target.stat().st_mtime_ns == mtime
    data = LowerData(target, [0,40,80], 64,48, video_sha256=sha256(video))
    assert data.frame(0)['pose_topology'] == 'mediapipe_pose_33'
    assert data.frame(0)['pose']['xy'][0] == [0,0]
    assert data.frame(0)['hands'][0]['actor'] == 'unknown'
    assert data.frame(0)['hands'][0]['track_id'] is None
    assert data.frame(1)['hands'] == []
    json.dumps(data.frame(2), allow_nan=False)
    with pytest.raises(ValueError, match='hash mismatch'):
        LowerData(target, [0,40,80],64,48,video_sha256='f'*64)


def test_conflict_and_corruption_are_not_overwritten(tmp_path):
    source, _, job = make_lower(tmp_path)
    target = tmp_path/'landmarks/T'/source.name
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes()); before = target.read_bytes()
    with pytest.raises(ValueError, match='not a normalized'): convert(job)
    assert target.read_bytes() == before
    target.unlink(); convert(job)
    with h5py.File(target, 'r+') as h: h['pose_landmarks_norm'][0,0] = [1,1]
    before = target.read_bytes()
    with pytest.raises(ValueError, match='coordinates or validity'): convert(job)
    assert target.read_bytes() == before


def test_frame_mismatch_and_unknown_units_rejected(tmp_path):
    source, _, job = make_lower(tmp_path)
    with pytest.raises(ValueError, match='shape/dtype'): LowerData(source, [0,40],64,48)
    with h5py.File(source, 'r+') as h: h['pose_landmarks'][0,0] = [.5,.5]
    with pytest.raises(ValueError, match='integer-valued'): convert(job)


def test_lower_only_portable_viewer_and_timestamp_matching(tmp_path):
    for view in ('CAM_LL', 'CAM_LR'):
        _, _, job = make_lower(tmp_path, view); convert(job)
    c = DatasetCollection(tmp_path/'videos', tmp_path/'landmarks')
    try:
        assert c.info(0)['primary_view'] == 'CAM_LL'
        assert set(c.info(0)['views']) == {'CAM_LL','CAM_LR'}
        for i in (2,0,1):
            f = c.frame(0,i)
            assert f['CAM_UL'] is None and f['CAM_AV'] is None
            for view in ('CAM_LL','CAM_LR'):
                assert f[view]['frame_index'] == i
                assert f[view]['timestamp_ms'] == pytest.approx(i*40)
                assert f[view]['image'].startswith('data:image/jpeg')
    finally:
        for clip in c.cache.values(): clip.close()
