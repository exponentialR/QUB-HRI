import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.collection_backend import point_box,restrict_to_hand_box,supported_by_box
from preprocessing.ul_ur_landmarks.schema import Hand
from preprocessing.ul_ur_landmarks.tracking import BoxHandTracker


def test_landmark_support_requires_most_points_in_proposal():
    points=np.array([[10,10],[12,12],[14,14],[100,100],[110,110],[120,120],[130,130]],dtype=float)
    assert not supported_by_box(points,[0,0,20,20])
    assert supported_by_box(points[:3],[0,0,20,20])
    assert point_box(np.full((21,2),np.nan)) is None


def test_box_tracks_survive_missing_joints_and_two_absent_frames():
    tracker=BoxHandTracker(1000,800)
    make=lambda x,side:Hand(np.full((21,2),np.nan),np.zeros(21),actor='participant',handedness=side,
                            bbox_xyxy=np.array([x,100,x+20,140]))
    first=[make(100,'left'),make(180,'right')]
    tracker.update(first)
    tracker.update([])
    tracker.update([])
    crossing=[make(130,'right'),make(150,'left')]
    tracker.update(crossing)
    assert crossing[0].track_id==first[1].track_id
    assert crossing[1].track_id==first[0].track_id


def test_hand_extent_does_not_treat_regressed_table_points_as_supported():
    xy=np.full((21,2),10.,dtype=np.float32)
    xy[0]=[0,0]
    xy[1]=[50,50]
    hand=Hand(xy,np.ones(21),bbox_xyxy=np.array([0,0,20,20]))
    restrict_to_hand_box(hand)
    assert np.array_equal(hand.xy[0],[0,0])
    assert np.isnan(hand.xy[1]).all() and hand.confidence[1]==0


@pytest.mark.parametrize('mode',['hybrid','native_hand5'])
@pytest.mark.parametrize('overlap',[False,True])
def test_batched_hand_fallback_keeps_source_order_and_tracks(mode,overlap,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from preprocessing.ul_ur_landmarks.collection_backend import CollectionBackend
    backend=CollectionBackend.__new__(CollectionBackend)
    backend.batch_size=4
    backend.overlap_face_tasks=overlap
    backend.hand_mode=mode
    backend.hand_joint_threshold=.2
    backend.hand_box_expansion=1.4
    backend.object_mode='full'
    backend.face_mode='single'
    backend.hand_nms={'participant_hand':.7,'surrogate_hand':.3}
    backend.max_hands_per_actor=2
    backend.person=lambda image:np.empty((0,4))
    monkeypatch.setattr('preprocessing.ul_ur_landmarks.collection_backend.face_rect_from_wholebody',
                        lambda *_:(0,0,100,80))
    class Proposals:
        def batch(self,images):
            return [[{'bbox':np.array([10,10,50,60]),'confidence':.9,'class_name':'surrogate_hand'}] for _ in images]
    class Objects:
        def batch(self,images):
            return [[] for _ in images]
    class Tasks:
        def face_points(self,image,*args):
            value=float(image[0,0,0])
            self.last_face_crop_to_source=np.full((2,3),value)
            return np.full((478,2),value),np.ones(478)
        def hand_points(self,*args):
            assert mode=='hybrid'
            return []
        def close(self):
            pass
    class HandPose:
        def batch(self,images,boxes):
            assert all(np.allclose(box,[2,0,58,70]) for box in boxes)
            values=np.array([np.full((21,2),im[0,0,0]) for im in images],dtype=float)
            return values,np.ones((len(images),21)),np.ones((len(images),21),dtype=bool)
    backend.detector=Proposals();backend.lego=Objects();backend.hand_pose=HandPose()
    backend.task_workers=[Tasks(),Tasks()];backend.tasks=backend.task_workers[0]
    backend.pool=ThreadPoolExecutor(max_workers=2)
    try:
        backend.reset(100,80)
        result=backend.process_batch([np.full((80,100,3),v,dtype=np.uint8) for v in (20,30,40)],[0,10,20])
        assert [f.timestamp_ms for f in result]==[0,10,20]
        assert [f.hands[0].xy[0,0] for f in result]==[20,30,40]
        assert [f.face_xy[0,0] for f in result]==[20,30,40]
        assert [f.face_crop_to_source[0,0] for f in result]==[20,30,40]
        assert len({f.hands[0].track_id for f in result})==1
        assert all(f.hands[0].actor=='other_actor' and f.hands[0].model_id=='rtmpose_hand5' for f in result)
        backend.reset(100,80)
        assert backend.timings=={}
    finally:
        backend.close()
