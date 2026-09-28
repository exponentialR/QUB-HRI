"""Reusable models for per-frame body, dense face, hand, and LEGO observations."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait
import os
import time

import numpy as np

from .detector_pilot import box_iou
from .hand_models import HandPose
from .pose_models import PersonDetector, RTMW, Sapiens, require_cuda, select_participant_box, visible_points
from .schema import Frame, Hand, ObjectDetection, sha256
from .task_crops import CropTasks,face_rect_from_wholebody,face_rotation_from_wholebody,square_rect
from .tracking import BoxHandTracker


def point_box(points: np.ndarray) -> np.ndarray | None:
    valid=np.asarray(points)[np.isfinite(points).all(axis=1)]
    if len(valid)<3:
        return None
    box=np.r_[valid.min(axis=0),valid.max(axis=0)]
    return box if np.all(box[2:]>box[:2]) else None


def supported_by_box(points: np.ndarray, box: np.ndarray, minimum: int = 3) -> bool:
    good=np.asarray(points)[np.isfinite(points).all(axis=1)]
    if len(good)<minimum:
        return False
    box=np.asarray(box)
    margin=(box[2:]-box[:2])*.1
    inside=np.all((good>=box[:2]-margin)&(good<=box[2:]+margin),axis=1)
    return int(inside.sum())>=minimum and inside.mean()>=.5


def restrict_to_hand_box(hand:Hand) -> None:
    """A proposal cannot support points far beyond its visible hand extent."""
    if hand.bbox_xyxy is None:
        return
    box=np.asarray(hand.bbox_xyxy)
    margin=(box[2:]-box[:2])*.1
    inside=(np.isfinite(hand.xy).all(axis=1)&np.all((hand.xy>=box[:2]-margin)&(hand.xy<=box[2:]+margin),axis=1))
    hand.xy[~inside]=np.nan
    hand.confidence[~inside]=0


def merge_object_detections(full, cropped, y_offset, threshold=.5):
    """Restore crop pixels and suppress duplicate observations within each original class."""
    rows = list(full) + [{**row, 'bbox': np.asarray(row['bbox']) + [0,y_offset,0,y_offset]}
                         for row in cropped]
    kept = []
    for row in sorted(rows, key=lambda p:p['confidence'], reverse=True):
        if not any(row['class_id'] == other['class_id'] and
                   box_iou(row['bbox'], other['bbox']) > threshold for other in kept):
            kept.append(row)
    return kept


def filter_hand_proposals(rows, iou_threshold=.7, max_per_actor=0):
    """Suppress same-actor duplicates, with an optional explicit dyadic hand-count prior."""
    kept=[]
    for row in sorted(rows,key=lambda p:p['confidence'],reverse=True):
        same_actor=[p for p in kept if p['class_name']==row['class_name']]
        if max_per_actor and len(same_actor)>=max_per_actor:
            continue
        threshold=iou_threshold.get(row['class_name'],.7) if isinstance(iou_threshold,dict) else iou_threshold
        if any(box_iou(row['bbox'],p['bbox'])>threshold for p in same_actor):
            continue
        kept.append(row)
    return kept


class YoloDetector:
    def __init__(self,path:Path,confidence:float,config_root:Path,*,expected_classes:set[str]):
        os.environ.update(YOLO_CONFIG_DIR=str(config_root),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true')
        require_cuda()
        from ultralytics import YOLO,settings
        settings.update({'sync':False})
        self.model=YOLO(str(path.resolve()),task='detect')
        if set(self.model.names.values())!=expected_classes:
            raise ValueError(f'Unexpected detector class names: {self.model.names}')
        self.confidence=confidence
        self.provenance={'checkpoint_sha256':sha256(path),'classes':{str(i):name for i,name in self.model.names.items()},'confidence':confidence,
                         'input_size':1280,'nms_iou':.7,'max_detections':100,'runtime':'ultralytics_cuda',
                         'confidence_kind':'detector_score_not_calibrated'}

    def __call__(self,bgr):
        return self.batch([bgr])[0]

    def batch(self,images):
        if not images:
            return []
        results=self.model.predict(images,device=0,imgsz=1280,conf=self.confidence,iou=.7,max_det=100,
                                   verbose=False,save=False)
        if len(results)!=len(images):
            raise ValueError('Detector batch length mismatch')
        return [self._decode(result,image) for result,image in zip(results,images)]

    def _decode(self,result,bgr):
        boxes=result.boxes.xyxy.cpu().numpy()
        scores=result.boxes.conf.cpu().numpy()
        classes=result.boxes.cls.cpu().numpy().astype(int)
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('Nonfinite object detector output')
        height,width=bgr.shape[:2]
        boxes=np.clip(boxes,[0,0,0,0],[width,height,width,height]).astype(np.float32)
        return [{'bbox':box,'confidence':float(score),'class_id':int(index),'class_name':self.model.names[int(index)]}
                for box,score,index in zip(boxes,scores,classes) if np.all(box[2:]>box[:2])]


class CollectionBackend:
    pose_topology='coco_wholebody_133'
    pose_count=133
    schema_version='1.1'
    expects_bgr=True

    def __init__(self,models_dir:Path,lego_weights:Path,hand_weights:Path,config_root:Path,*,pose_model='rtmw',
                 batch_size:int=8,cpu_workers:int=2,hand_mode:str='hybrid',
                 hand_joint_threshold:float=.3,hand_box_expansion:float=1.,object_mode:str='full',
                 face_mode:str='single',other_hand_nms_iou:float=.7,max_hands_per_actor:int=0,
                 overlap_face_tasks:bool=False):
        if not 1<=batch_size<=32 or not 1<=cpu_workers<=4:
            raise ValueError('Use batch size 1–32 and CPU task workers 1–4')
        if hand_mode not in {'hybrid','native_hand5'}:
            raise ValueError('Unknown hand inference mode')
        if not 0<hand_joint_threshold<1 or not 1<=hand_box_expansion<=2:
            raise ValueError('Hand5 response threshold must be in (0,1), box expansion in [1,2]')
        if object_mode not in {'full','full_lower'}:
            raise ValueError('Unknown object inference mode')
        self.object_mode=object_mode
        if face_mode not in {'single','crop_fallback'}:
            raise ValueError('Unknown face inference mode')
        self.face_mode=face_mode
        if not 0<other_hand_nms_iou<=1 or max_hands_per_actor not in {0,1,2}:
            raise ValueError('Other-hand NMS IoU must be in (0,1]; hand capacity must be 0, 1 or 2')
        self.hand_nms={'participant_hand':.7,'surrogate_hand':other_hand_nms_iou}
        self.max_hands_per_actor=max_hands_per_actor
        self.hand_mode=hand_mode
        self.hand_joint_threshold=hand_joint_threshold
        self.hand_box_expansion=hand_box_expansion
        import torch
        torch.set_num_threads(4)
        self.person=PersonDetector(models_dir/'yolox_m_humanart.onnx',threshold=.3)
        self.pose=(RTMW(models_dir/'rtmw_l_384x288.onnx') if pose_model=='rtmw' else
                   Sapiens(models_dir/'sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2',
                           models_dir.parent/'model_sources/sapiens_pose_utils.py'))
        self.hand_pose=HandPose(models_dir/'rtmpose_m_hand5_256.onnx')
        self.detector=YoloDetector(hand_weights,.25,config_root,expected_classes={'participant_hand','surrogate_hand'})
        self.lego=YoloDetector(lego_weights,.25,config_root,expected_classes={
            'two_two_block','four_two_block','assembly_base','biah_hole','lefthand','mini_stairway',
            'righthand','stacked_bridge','stacked_stairs','stacked_tower','surrogate_hand','tower_head'})
        self.lego.provenance['inference_mode']=object_mode
        if object_mode=='full_lower':
            self.lego.provenance['additional_crop']='full width; lower two-thirds; y origin rounded(height/3)'
            self.lego.provenance['merge']='class-aware confidence-ordered NMS at IoU 0.5 after restoring source pixels'
        self.tasks=CropTasks(models_dir,hands=hand_mode=='hybrid')
        self.task_workers=[self.tasks]+[CropTasks(models_dir,hands=hand_mode=='hybrid') for _ in range(cpu_workers-1)]
        self.pool=ThreadPoolExecutor(max_workers=cpu_workers) if cpu_workers>1 else None
        self.batch_size=batch_size
        self.overlap_face_tasks=overlap_face_tasks
        self.timings={}
        self.provenance={'name':'ul_ur_collection_candidate','pose_model':self.pose.provenance,
                         'person_detector':self.person.provenance,'hand_detector':self.detector.provenance,
                         'hand_pose':self.hand_pose.provenance,'lego_detector':self.lego.provenance,
                         'crop_tasks':self.tasks.provenance,'pose_confidence_kind':self.pose.provenance['confidence_kind'],
                         'person_selection':'uppermost_center_among_boxes_starting_in_upper_half',
                         'joint_threshold':.3,'face_rotation':'COCO_WholeBody_iBUG_eye_group_alignment',
                         'face_mode':face_mode,
                         'face_crop_scales_on_failed_detection':[1.,1.4,.75] if face_mode=='crop_fallback' else [1.],
                         'hand5_joint_threshold':hand_joint_threshold,
                         'hand5_detector_box_expansion_before_intrinsic_padding':hand_box_expansion,
                         'hand_proposal_filter':{'nms_iou_by_detector_class':self.hand_nms,
                                                 'max_hands_per_actor':max_hands_per_actor,
                                                 'capacity_prior':'one participant and one other actor, each with at most two hands; 0 disables capacity'},
                         'hand_point_validity':'above joint threshold, in frame, crop interior, within detector box expanded by 10 percent',
                         'hands':'independent detector proposals plus full-frame MediaPipe; proposal-supported native participant hands, then MediaPipe crop, then Hand5 fallback',
                         'hand_mode':hand_mode,
                         'actor_policy':'detector class; unmatched MediaPipe hand close to participant wrist is participant, otherwise unknown',
                         'handedness_policy':'native body hand assignment only; otherwise unknown',
                         'tracking':'box-center matching with actor/known-handedness constraints; two missing frames allowed',
                         'physical_visibility':'unmeasured; validity means an available prediction',
                         'execution':{'frame_batch_size':batch_size,'cpu_task_workers':cpu_workers,
                                      'overlap_cpu_face_with_gpu_detectors':overlap_face_tasks,
                                      'pose_batch':'official dynamic axes where supported',
                                      'tracking_order':'source frame order, after batch inference',
                                      'numerics':'GPU batch arithmetic may differ slightly from single-frame results'},
                         'hand_models':{'wholebody':self.pose.provenance,'mediapipe_hand_landmarker':self.tasks.provenance,
                                        'rtmpose_hand5':self.hand_pose.provenance},
                         'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in (
                             'collection_backend.py','pose_models.py','hand_models.py','task_crops.py','tracking.py','schema.py','runner.py')}}
        self.tracker=None
        if hand_mode=='native_hand5':
            self.provenance['hands']='independent detector proposals; proposal-supported native participant hands, then Hand5; MediaPipe hand passes disabled'
            self.provenance['actor_policy']='independent hand detector class'
            self.provenance['hand_models'].pop('mediapipe_hand_landmarker')

    def reset(self,width:int,height:int):
        self.width,self.height=width,height
        self.tracker=BoxHandTracker(width,height)
        self.timings={}

    def _native_hands(self,xy,scores):
        if xy is None:
            return []
        return [Hand(xy[start:start+21].copy(),scores[start:start+21].copy(),actor='participant',
                     handedness=side,topology='coco_wholebody_'+side+'_hand_21',model_id='wholebody')
                for start,side in ((91,'left'),(112,'right'))]

    def process(self,bgr:np.ndarray,timestamp_ms:float) -> Frame:
        return self.process_batch([bgr],[timestamp_ms])[0]

    def _record_time(self,name,started):
        self.timings[name]=self.timings.get(name,0.)+time.perf_counter()-started

    def process_batch(self,images:list[np.ndarray],timestamps:list[float]) -> list[Frame]:
        if not images or len(images)!=len(timestamps) or len(images)>self.batch_size:
            raise ValueError('Invalid collection batch length')
        if self.tracker is None or any(im.shape[:2]!=(self.height,self.width) for im in images):
            raise ValueError('Backend must be reset to this clip geometry')
        started=time.perf_counter()
        person_boxes=[]
        for bgr in images:
            boxes=self.person(bgr)
            index=select_participant_box(boxes,self.height)
            person_boxes.append(boxes[index] if index is not None else None)
        self._record_time('person_detector_s',started)
        started=time.perf_counter()
        poses=[(None,None) for _ in images]
        selected=[i for i,box in enumerate(person_boxes) if box is not None]
        if selected:
            if hasattr(self.pose,'batch'):
                raw,confidence,interior=self.pose.batch([images[i] for i in selected],[person_boxes[i] for i in selected])
            else:
                results=[]
                for i in selected:
                    points,scores=self.pose(images[i],person_boxes[i])
                    results.append((points,scores,self.pose.last_crop_valid.copy()))
                raw,confidence,interior=zip(*results)
            for i,points,scores,inside in zip(selected,raw,confidence,interior):
                xy,score,valid=visible_points(points,scores,self.width,self.height,.3)
                valid &= inside
                xy[~valid]=np.nan
                score[~valid]=0
                poses[i]=(xy,score)
        self._record_time('wholebody_s',started)
        groups=[list(range(i,len(images),len(self.task_workers))) for i in range(len(self.task_workers))]
        face_futures=[]
        if self.overlap_face_tasks and self.pool is not None:
            def face_group(worker,indices):
                return [(i,self._face_observation(worker,images[i],poses[i][0])) for i in indices]
            face_futures=[self.pool.submit(face_group,worker,indices)
                          for worker,indices in zip(self.task_workers,groups) if indices]
        started=time.perf_counter()
        proposals=self.detector.batch(images)
        proposals=[filter_hand_proposals(rows,self.hand_nms,self.max_hands_per_actor) for rows in proposals]
        objects=self.lego.batch(images)
        if self.object_mode=='full_lower':
            origin=round(self.height/3)
            cropped=self.lego.batch([im[origin:] for im in images])
            objects=[merge_object_detections(full,lower,origin) for full,lower in zip(objects,cropped)]
        self._record_time('hand_and_object_detectors_s',started)
        started=time.perf_counter()
        faces={i:value for future in face_futures for i,value in future.result()}
        def prepare_group(worker,indices):
            return [(i,self._prepare_frame(worker,images[i],timestamps[i],person_boxes[i],*poses[i],proposals[i],objects[i],faces.get(i)))
                    for i in indices]
        if self.pool is None:
            prepared=prepare_group(self.tasks,groups[0])
        else:
            futures=[self.pool.submit(prepare_group,worker,indices) for worker,indices in zip(self.task_workers,groups) if indices]
            wait(futures)
            prepared=[item for future in futures for item in future.result()]
        prepared.sort(key=lambda item:item[0])
        self._record_time('mediapipe_and_hand_assignment_s',started)
        pending=[(i,hand,box) for i,(_,requests) in prepared for hand,box in requests]
        started=time.perf_counter()
        for offset in range(0,len(pending),self.batch_size):
            part=pending[offset:offset+self.batch_size]
            boxes=[]
            for _,_,box in part:
                center=(box[:2]+box[2:])/2
                half=(box[2:]-box[:2])*self.hand_box_expansion/2
                boxes.append(np.r_[center-half,center+half])
            raw,confidence,interior=self.hand_pose.batch([images[i] for i,_,_ in part],boxes)
            for (_,hand,_),points,scores,inside in zip(part,raw,confidence,interior):
                xy,score,valid=visible_points(points,scores,self.width,self.height,self.hand_joint_threshold)
                valid &= inside
                xy[~valid]=np.nan
                score[~valid]=0
                hand.xy,hand.confidence=xy,score
        self._record_time('hand5_s',started)
        frames=[]
        for _,(frame,_) in prepared:
            for hand in frame.hands:
                restrict_to_hand_box(hand)
            self.tracker.update(frame.hands)
            frames.append(frame)
        return frames

    def _face_observation(self,tasks,bgr,pose_xy):
        rect=face_rect_from_wholebody(pose_xy,self.width,self.height)
        angle=face_rotation_from_wholebody(pose_xy) if pose_xy is not None else 0.
        face_xy,face_scores=tasks.face_points(bgr,rect,angle)
        if self.face_mode=='crop_fallback' and rect is not None and face_xy is None:
            original_rect=rect
            for scale in (1.4,.75):
                rect=square_rect(original_rect,self.width,self.height,padding=scale)
                face_xy,face_scores=tasks.face_points(bgr,rect,angle)
                if face_xy is not None:
                    break
        return (face_xy,face_scores,rect,
                tasks.last_face_crop_to_source.copy() if rect is not None else None)

    def _prepare_frame(self,tasks,bgr,timestamp_ms,person_box,pose_xy,pose_scores,proposals,objects,face=None):
        face_xy,face_scores,rect,face_transform=(self._face_observation(tasks,bgr,pose_xy) if face is None else face)
        native=self._native_hands(pose_xy,pose_scores)
        independent=tasks.hand_points(bgr,(0,0,self.width,self.height)) if self.hand_mode=='hybrid' else []
        observations=[]
        pending=[]
        used_native=set()
        for proposal in proposals:
            box=proposal['bbox']
            actor='participant' if proposal['class_name']=='participant_hand' else 'other_actor'
            hand=None
            if actor=='participant':
                candidates=[(i,h) for i,h in enumerate(native) if i not in used_native and supported_by_box(h.xy,box)]
                if candidates:
                    center=(box[:2]+box[2:])/2
                    i,hand=min(candidates,key=lambda item:np.linalg.norm(np.nanmedian(item[1].xy,axis=0)-center))
                    used_native.add(i)
            if hand is None:
                candidates=[h for h in independent if supported_by_box(h.xy,box) and
                            (point_box(h.xy) is not None and box_iou(point_box(h.xy),box)>=.5)]
                if not candidates and self.hand_mode=='hybrid':
                    crop=square_rect(box,self.width,self.height,padding=2.)
                    candidates=[h for h in tasks.hand_points(bgr,crop) if supported_by_box(h.xy,box) and
                                (point_box(h.xy) is not None and box_iou(point_box(h.xy),box)>=.5)]
                if candidates:
                    hand=max(candidates,key=lambda h:box_iou(point_box(h.xy),box))
                else:
                    hand=Hand(np.full((21,2),np.nan,dtype=np.float32),np.zeros(21,dtype=np.float32),
                              topology='hand5_21',model_id='rtmpose_hand5')
            # Copy arrays: reused independent detections must not share mutable metadata.
            hand=Hand(hand.xy.copy(),hand.confidence.copy(),actor=actor,handedness=hand.handedness,
                      topology=hand.topology,model_id=hand.model_id,bbox_xyxy=box.copy(),
                      detection_confidence=proposal['confidence'])
            observations.append(hand)
            if hand.model_id=='rtmpose_hand5':
                pending.append((hand,box))
        for hand in independent:
            if any(supported_by_box(hand.xy,p['bbox']) for p in proposals):
                continue
            box=point_box(hand.xy)
            if box is None:
                continue
            if pose_xy is not None and np.isfinite(hand.xy[0]).all():
                wrists=pose_xy[[9,10]]
                distances=np.linalg.norm(wrists-hand.xy[0],axis=1)
                if np.isfinite(distances).any() and np.nanmin(distances)<=.06*np.hypot(self.width,self.height):
                    hand.actor='participant'
            hand.bbox_xyxy=box
            observations.append(hand)
        objects=[ObjectDetection(d['bbox'],d['confidence'],d['class_id'],d['class_name'],'lego_yolo') for d in objects]
        return (Frame(timestamp_ms,pose_xy,pose_scores,face_xy,face_scores,rect,observations,objects,person_box,
                      face_transform),pending)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        for worker in self.task_workers:
            worker.close()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
