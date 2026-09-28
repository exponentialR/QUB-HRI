import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.viewer import Collection, Clip, frame_at_time, handler_for, points
from preprocessing.ul_ur_landmarks.schema import Frame, Hand, sha256, write_hdf5


def test_timestamp_pairing_and_missing_points():
    assert frame_at_time([0,40,90],65)==1
    assert frame_at_time([0,40,90],89)==2
    assert frame_at_time([0,40,90],140) is None
    assert frame_at_time([0],5) is None
    data=points(np.array([[0.,0.],[5,6],[np.nan,np.nan]]),np.array([1.,0.,0.]),np.array([True,False,False]))
    assert data['xy']==[[0.,0.],None,None]
    json.dumps(data,allow_nan=False)


def test_video_seeking_matches_sequential_decode_and_validated_output(tmp_path):
    video=tmp_path/'source.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),25,(200,200))
    assert writer.isOpened()
    for i in range(12):writer.write(np.full((200,200,3),(i*17,i*11,i*7),dtype=np.uint8))
    writer.release()
    capture=cv2.VideoCapture(str(video));decoded=[]
    while True:
        okay,image=capture.read()
        if not okay:break
        decoded.append(image)
    capture.release()
    meta={'relpath':video.name,'width':200,'height':200,'declared_frames':12}
    row={'pair_id':'T/P01-T','pid':'P01','views':{'CAM_UL':meta}}
    source={**meta,'pair_id':row['pair_id'],'pid':'P01','view':'CAM_UL','sha256':sha256(video)}
    path=tmp_path/'landmarks.h5';model={'name':'fixture'}
    frames=[Frame(i*40.,hands=[Hand(np.zeros((21,2)),np.ones(21),actor='other_actor',track_id=3,
                                  model_id='test',bbox_xyxy=np.array([0,0,20,20]))]) for i in range(12)]
    write_hdf5(path,frames,source=source,model=model,pose_topology='coco_wholebody_133',pose_count=133,schema_version='1.1')
    clip=Clip(path,tmp_path,row,'CAM_UL',model)
    try:
        for index in [11,0,5,6,2]:assert np.array_equal(clip.image(index),decoded[index])
        data=clip.frame(5,include_image=False)
        assert data['frame_index']==5 and data['timestamp_ms']==200 and 'image' not in data
        assert data['hands'][0]['actor']=='other_actor' and data['hands'][0]['track_id']==3
        assert data['hands'][0]['xy'][0]==[0.,0.] and data['hands'][0]['detection_confidence'] is None
        json.dumps(data,allow_nan=False)
    finally:clip.close()
    with pytest.raises(ValueError,match='identity mismatch'):
        Clip(path,tmp_path,{**row,'pid':'P99'},'CAM_UL',model)


def test_local_server_only_exposes_explicit_routes_and_rejects_cross_site():
    class Fake:
        quality_root=None
        def index(self):return {'pairs':[]}
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(Fake()))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    address=f'http://127.0.0.1:{server.server_port}'
    try:
        assert json.load(urllib.request.urlopen(address+'/api/index'))=={'pairs':[]}
        for path,headers,expected in [('/api/index',{'Origin':'https://example.com'},403),('/api/index',{'Host':'example.com'},403),('/api/index',{'Sec-Fetch-Site':'cross-site'},403),('/../../etc/passwd',{},404)]:
            with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(urllib.request.Request(address+path,headers=headers))
            assert exc.value.code==expected
    finally:server.shutdown();server.server_close();thread.join()
