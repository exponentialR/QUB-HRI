import json
import shutil

import cv2
import h5py
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.viewer_dataset import discover_dataset
from preprocessing.ul_ur_landmarks.viewer import DatasetCollection
from preprocessing.ul_ur_landmarks.schema import Frame, sha256, write_hdf5
from preprocessing.ul_ur_landmarks.normalization import add_normalized
from preprocessing.ul_ur_landmarks.lower import TOPOLOGIES, add_normalized as add_lower_normalized


def make_dataset(root, views=('CAM_AV','CAM_UL','CAM_UR')):
    for name in ('videos','landmarks'):(root/name/'T').mkdir(parents=True)
    for view in views:
        rel=f'T/p01-{view}-TASK-0_1.mp4';video=root/'videos'/rel;out=(root/'landmarks'/rel).with_suffix('.h5')
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),25,(64,48));assert writer.isOpened()
        for i in range(3):writer.write(np.full((48,64,3),i*50,np.uint8))
        writer.release()
        if view=='CAM_AV':
            with h5py.File(out,'w') as h:
                h['timestamps']=np.array([0,.04,.08],np.float32)
                h['left_landmarks']=np.zeros((3,21,3),np.float32)
                h['right_landmarks']=np.zeros((3,21,3),np.float32)
                h['norm_gaze']=np.full((3,2),.5,np.float32)
                h['rec_bboxes']=np.zeros((3,30,4),np.float32)
                h['surrogate_hands']=np.zeros((3,2,4),np.float32)
        elif view in ('CAM_LL','CAM_LR'):
            with h5py.File(out,'w') as h:
                for name, (_, count) in TOPOLOGIES.items():
                    h[name]=np.full((3,count,2),[32,24],np.float32)
                add_lower_normalized(h, {'relpath':rel,'view':view,'width':64,'height':48,
                                        'sha256':sha256(video),'frames':3}, input_sha256='a'*64,
                                     original_path='/unavailable/legacy/file.h5',code_sha256='c'*64)
        else:
            source={'pair_id':'T/P01-TASK-0_1','pid':'P01','view':view,'relpath':rel,'width':64,'height':48,
                    'sha256':sha256(video),'declared_frames':3}
            write_hdf5(out,[Frame(i*40) for i in range(3)],source=source,model={'name':'synthetic'},
                       pose_topology='coco_wholebody_133',pose_count=133,schema_version='1.1')
            with h5py.File(out,'r+') as h:
                add_normalized(h,input_sha256='a'*64,archive='/unavailable/original-machine/file.h5',code_sha256='c'*64)
    return root


@pytest.mark.parametrize('views,primary',[
    (['CAM_AV','CAM_UL','CAM_UR'],'CAM_UL'), (['CAM_AV'],'CAM_AV'), (['CAM_UR'],'CAM_UR'),
    (['CAM_AV','CAM_UL','CAM_UR','CAM_LL','CAM_LR'],'CAM_UL'), (['CAM_LR'],'CAM_LR')])
def test_portable_moved_dataset_without_run_records(tmp_path,views,primary):
    root=make_dataset(tmp_path/'original',views);moved=tmp_path/'relocated dataset';shutil.move(root,moved)
    c=DatasetCollection(moved/'videos',moved/'landmarks')
    try:
        assert c.discovery_summary['clip_groups']==1
        assert c.info(0)['primary_view']==primary
        frame=c.frame(0,2)
        for view in ('CAM_AV','CAM_UL','CAM_UR','CAM_LL','CAM_LR'):
            if view in views:
                assert frame[view]['frame_index']==2
                assert frame[view]['image'].startswith('data:image/jpeg')
            else:assert frame[view] is None
    finally:
        for clip in c.cache.values():clip.close()


def test_bad_view_does_not_disable_other_available_views(tmp_path):
    root=make_dataset(tmp_path/'data')
    path=next((root/'landmarks').rglob('*CAM_UL*.h5'))
    with h5py.File(path,'r+') as h:
        source=json.loads(h.attrs['source_json']);source['sha256']='f'*64;h.attrs['source_json']=json.dumps(source)
    c=DatasetCollection(root/'videos',root/'landmarks')
    try:
        info=c.info(0)
        assert info['primary_view']=='CAM_AV'
        assert 'Source hash mismatch' in info['view_errors']['CAM_UL']
        assert c.frame(0,1)['CAM_UL'] is None
    finally:
        for clip in c.cache.values():clip.close()


def test_discovery_reports_missing_files_and_rejects_escapes_and_duplicates(tmp_path):
    root=make_dataset(tmp_path/'data',['CAM_AV'])
    missing=root/'videos/T/p02-CAM_UL-TASK-0_1.mp4';missing.write_bytes(b'unpaired')
    rows,summary=discover_dataset(root/'videos',root/'landmarks')
    assert len(rows)==1 and summary['videos_without_landmarks']==1
    duplicate=root/'videos/T/P01-CAM_AV-TASK-0_1.mp4';duplicate.write_bytes(b'duplicate')
    with pytest.raises(ValueError,match='Duplicate'):discover_dataset(root/'videos',root/'landmarks')
    duplicate.unlink();outside=tmp_path/'outside.mp4';outside.write_bytes(b'outside')
    (root/'videos/T/p03-CAM_UL-TASK-0_1.mp4').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes'):discover_dataset(root/'videos',root/'landmarks')
