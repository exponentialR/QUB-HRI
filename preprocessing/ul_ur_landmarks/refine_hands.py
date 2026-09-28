"""Refine hand observations from cached landmarks into a separate, atomic HDF5 file.

No video or model inference is performed. Original body, face, object and timing
datasets are copied exactly. The input HDF5 remains the provenance record.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .collection_backend import point_box, supported_by_box
from .schema import ACTORS, HANDEDNESS, Hand, _points, sha256, validate_hdf5
from .tracking import BoxHandTracker


NATIVE_MODEL_ID = 'wholebody_native_fallback'
HAND_FIELDS = {'frame_index','track_id','actor','handedness','xy_px','confidence','valid',
               'topology','model_id','bbox_xyxy_px','bbox_valid','detection_confidence'}


def refine_frame(pose, scores, hands, width, height, threshold=4., clear_ambiguous_side=True):
    """Keep existing points; add strong, unsupported native participant hand hypotheses."""
    result=copy.deepcopy(hands)
    native=[(pose[start:start+21],scores[start:start+21],wrist,side)
            for start,wrist,side in ((91,9,'left'),(112,10,'right'))]
    if clear_ambiguous_side:
        for hand in result:
            if hand.actor=='participant' and hand.model_id=='wholebody' and hand.bbox_xyxy is not None:
                compatible=[side for points,_,_,side in native if supported_by_box(points,hand.bbox_xyxy)]
                if len(compatible)>1:
                    hand.handedness='unknown'
    candidates=[]
    for points,confidence,wrist,side in native:
        valid=np.isfinite(points).all(axis=1)
        if valid.sum()<10 or not valid[0] or not np.isfinite(pose[wrist]).all():
            continue
        response=float(np.median(confidence[valid]))
        if response<threshold or np.linalg.norm(points[0]-pose[wrist])>.03*np.hypot(width,height):
            continue
        if any(h.bbox_xyxy is not None and supported_by_box(points,h.bbox_xyxy) for h in result):
            continue
        box=point_box(points)
        if box is None:
            continue
        center=(box[:2]+box[2:])/2
        half=(box[2:]-box[:2])*.6
        box=np.clip(np.r_[center-half,center+half],[0,0,0,0],[width,height,width,height])
        if np.any(box[2:]-box[:2]<10) or np.any(box[2:]-box[:2]>np.array([width,height])*.25):
            continue
        candidates.append((response,Hand(points.copy(),confidence.copy(),actor='participant',handedness=side,
            topology='coco_wholebody_'+side+'_hand_21',model_id=NATIVE_MODEL_ID,bbox_xyxy=box)))
    for _,hand in sorted(candidates,key=lambda row:-row[0]):
        if sum(h.actor=='participant' for h in result)>=2:
            break
        if any(h.bbox_xyxy is not None and supported_by_box(hand.xy,h.bbox_xyxy) for h in result):
            continue
        result.append(hand)
    return result


def configuration(model, threshold, clear_ambiguous_side):
    result=copy.deepcopy(model)
    if 'hand_refinement' in result:
        raise ValueError('Input has already undergone hand refinement; use original collection outputs')
    if result.get('hand_mode')!='native_hand5' or 'wholebody' not in result.get('hand_models',{}):
        raise ValueError('Expected a native_hand5 collection with documented wholebody hand provenance')
    if result['hand_models']['wholebody'].get('model')!='rtmw_l_384x288':
        raise ValueError('The response threshold is specific to RTMW SimCC; other models require a separate calibration')
    result['hand_refinement']={
        'version':1,'median_native_simcc_response_threshold':threshold,'minimum_native_points':10,
        'native_to_body_wrist_max_image_diagonal_fraction':.03,'extent_expansion':1.2,
        'minimum_extent_px':10,'maximum_extent_fraction_xy':.25,'max_participant_observations':2,
        'duplicate_guard':'at least 3 points and at least half of available points in any existing hand box expanded 10 percent',
        'clear_ambiguous_handedness':clear_ambiguous_side,
        'ambiguity':'both native left and right hand point groups support one detector box',
        'coordinates':'existing observations unchanged; added points copied from cached 133-point pose',
        'tracking':'recompute BoxHandTracker in source order after refinement; no interpolation',
        'bbox_kind_for_added_rows':'native point extent expanded 20 percent; not an independent detection',
        'physical_visibility':'unmeasured; high uncalibrated response is not a visibility probability',
        'input_model_sha256':hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest(),
        'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                       ('refine_hands.py','collection_backend.py','tracking.py','schema.py')}}
    result['hand_models'][NATIVE_MODEL_ID]={**model['hand_models']['wholebody'],
        'observation_role':'fallback from cached native participant pose',
        'bbox_kind':'expanded native landmark extent; no independent detector confidence'}
    result['hands']=model.get('hands','')+'; strong unsupported native participant hand hypotheses added from cached pose'
    result['actor_policy']=model.get('actor_policy','')+'; added native hypotheses use the selected participant pose identity'
    if clear_ambiguous_side:
        result['handedness_policy']=model.get('handedness_policy','')+'; ambiguous native-to-box side assignments become unknown during refinement'
    return result


def write_hands(group, frames):
    """Write the existing 1.1 hand contract, including explicit absent detector scores."""
    import h5py
    rows=[(i,h) for i,hands in enumerate(frames) for h in hands]
    for key,values,dtype in (
        ('frame_index',[i for i,_ in rows],np.int32),('track_id',[h.track_id for _,h in rows],np.int32),
        ('actor',[ACTORS[h.actor] for _,h in rows],np.uint8),
        ('handedness',[HANDEDNESS[h.handedness] for _,h in rows],np.uint8)):
        group.create_dataset(key,data=np.asarray(values,dtype=dtype))
    processed=[_points(h.xy,h.confidence,21) for _,h in rows]
    for key,index,empty,dtype in (('xy_px',0,(0,21,2),np.float32),('confidence',1,(0,21),np.float32),('valid',2,(0,21),bool)):
        values=np.stack([p[index] for p in processed]) if processed else np.empty(empty,dtype=dtype)
        group.create_dataset(key,data=values,compression='gzip',compression_opts=1)
    for key in ('topology','model_id'):
        group.create_dataset(key,data=[getattr(h,key) for _,h in rows],dtype=h5py.string_dtype('utf-8'))
    group.create_dataset('bbox_xyxy_px',data=np.asarray([h.bbox_xyxy if h.bbox_xyxy is not None else [np.nan]*4 for _,h in rows],dtype=np.float32).reshape(-1,4))
    group.create_dataset('bbox_valid',data=np.asarray([h.bbox_xyxy is not None for _,h in rows],dtype=bool))
    group.create_dataset('detection_confidence',data=np.asarray([h.detection_confidence if h.detection_confidence is not None else np.nan for _,h in rows],dtype=np.float32))
    group['detection_confidence'].attrs['missing']='NaN means no separate detector score'


def refine_file(input_path, output_path, threshold=4., clear_ambiguous_side=True):
    import h5py
    input_path,output_path=input_path.resolve(),output_path.resolve()
    if input_path==output_path or not np.isfinite(threshold) or threshold<=0:
        raise ValueError('Use a separate output and a positive finite response threshold')
    input_hash=sha256(input_path)
    with h5py.File(input_path) as original:
        if original.attrs['schema_version']!='1.1' or original.attrs['pose_topology']!='coco_wholebody_133':
            raise ValueError('Expected collection schema 1.1 / coco_wholebody_133')
        if set(original['hands'])!=HAND_FIELDS:
            raise ValueError('Unrecognized hand fields; do not silently drop an extended schema')
        source=json.loads(original.attrs['source_json'])
        input_model=json.loads(original.attrs['model_json'])
        model=configuration(input_model,threshold,clear_ambiguous_side)
        validate_hdf5(input_path,source_sha256=source['sha256'],model=input_model)
        source['parent_landmarks']={'path':str(input_path),'sha256':input_hash}
        if output_path.exists():
            result=validate_hdf5(output_path,source_sha256=source['sha256'],model=model)
            with h5py.File(output_path) as existing:
                if json.loads(existing.attrs['source_json'])!=source:
                    raise ValueError('Existing refinement has a different parent input')
            return {'status':'skipped_valid',**result}
        data={key:(value.asstr()[:] if key in ('model_id','topology') else value[:]) for key,value in original['hands'].items()}
        poses=original['participant/pose/xy_px'][:];scores=original['participant/pose/confidence'][:]
        hand_attrs=dict(original['hands'].attrs)
    actors={value:key for key,value in ACTORS.items()};sides={value:key for key,value in HANDEDNESS.items()}
    input_frames=[[] for _ in poses]
    for row,index in enumerate(data['frame_index']):
        score=data['detection_confidence'][row]
        input_frames[index].append(Hand(data['xy_px'][row].copy(),data['confidence'][row].copy(),
            actor=actors[int(data['actor'][row])],handedness=sides[int(data['handedness'][row])],
            track_id=int(data['track_id'][row]),topology=str(data['topology'][row]),model_id=str(data['model_id'][row]),
            bbox_xyxy=data['bbox_xyxy_px'][row].copy() if data['bbox_valid'][row] else None,
            detection_confidence=float(score) if np.isfinite(score) else None))
    frames=[];added=ambiguous=0
    tracker=BoxHandTracker(source['width'],source['height'])
    for pose,score,hands in zip(poses,scores,input_frames):
        refined=refine_frame(pose,score,hands,source['width'],source['height'],threshold,clear_ambiguous_side)
        added+=len(refined)-len(hands)
        ambiguous+=sum(a.handedness!=b.handedness for a,b in zip(hands,refined))
        tracker.update(refined);frames.append(refined)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+output_path.name+'.',suffix='.partial',dir=output_path.parent)
    os.close(fd);temporary=Path(name)
    try:
        shutil.copyfile(input_path,temporary)
        if sha256(temporary)!=input_hash:
            raise ValueError('Input changed while it was being copied')
        with h5py.File(temporary,'r+') as result:
            result.attrs['source_json']=json.dumps(source,sort_keys=True)
            result.attrs['model_json']=json.dumps(model,sort_keys=True)
            del result['hands']
            group=result.create_group('hands')
            group.attrs.update(hand_attrs)
            write_hands(group,frames)
        validation=validate_hdf5(temporary,source_sha256=source['sha256'],model=model)
        os.link(temporary,output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'status':'written','input_sha256':input_hash,'output_sha256':sha256(output_path),
            'added_native_observations':added,'ambiguous_side_labels_cleared':ambiguous,**validation}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-file',type=Path,required=True)
    p.add_argument('--output-file',type=Path,required=True)
    p.add_argument('--native-response-threshold',type=float,default=4.)
    p.add_argument('--retain-ambiguous-side',action='store_true')
    a=p.parse_args()
    if a.output_file.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        p.error('Keep refined research outputs outside the repository')
    print(json.dumps(refine_file(a.input_file,a.output_file,a.native_response_threshold,not a.retain_ambiguous_side)))


if __name__=='__main__':main()
