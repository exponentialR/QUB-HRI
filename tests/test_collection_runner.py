import h5py
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.runner import process_clip
from preprocessing.ul_ur_landmarks.schema import Frame


@pytest.mark.parametrize('batched',[False,True])
def test_collection_clip_uses_source_indices_bgr_and_validated_resume(tmp_path,monkeypatch,batched):
    import cv2
    source=tmp_path/'P01-CAM_UL-T.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,48))
    assert writer.isOpened()
    for _ in range(3):
        image=np.zeros((48,64,3),dtype=np.uint8)
        image[:,:,0]=255
        writer.write(image)
    writer.release()
    class Backend:
        provenance={'fixture':True}
        pose_topology='fixture_2'
        pose_count=2
        schema_version='1.1'
        expects_bgr=True
        def reset(self,width,height):
            assert (width,height)==(64,48)
        def process(self,bgr,timestamp):
            return Frame(timestamp,pose_xy=np.array([[int(bgr[0,0,0]>100),2],[3,4]]),pose_confidence=np.ones(2))
    if batched:
        Backend.batch_size=2
        Backend.process_batch=lambda self,images,times:[self.process(im,t) for im,t in zip(images,times)]
    row={'pair_id':'P01-T','pid':'P01','views':{'CAM_UL':{'width':64,'height':48,'declared_frames':3}}}
    output=tmp_path/'results/result.h5'
    kwargs=dict(input_root=tmp_path,row=row,view='CAM_UL',models_dir=tmp_path,backend=Backend())
    result=process_clip(source,output,**kwargs)
    assert result['status']=='written' and result['frames']==3
    with h5py.File(output) as data:
        assert data['frames/index'][:].tolist()==[0,1,2]
        assert data['frames/timestamp_ms'][:].tolist()==[0,100,200]
        assert data['participant/pose/xy_px'][0,0].tolist()==[1,2]
    before=output.stat().st_mtime_ns
    def forbidden_decode(_):
        raise AssertionError('Validated rerun must not decode the source again')
    monkeypatch.setattr('preprocessing.ul_ur_landmarks.runner.frame_pts_ms',forbidden_decode)
    assert process_clip(source,output,**kwargs)['status']=='skipped_valid'
    assert output.stat().st_mtime_ns==before
