"""Propagate independently placed seed labels for visual review, never as final truth."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .detector_pilot import write_json
from .schema import sha256


def flow_step(previous, following, points: np.ndarray, valid: np.ndarray):
    import cv2
    updated=np.asarray(points,dtype=np.float32).copy()
    okay=np.asarray(valid,dtype=bool).copy()
    errors=np.full(len(points),np.nan,dtype=np.float32)
    indices=np.flatnonzero(okay)
    if not len(indices):
        return updated,okay,errors
    start=updated[indices].reshape(-1,1,2)
    options=dict(winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    forward,status,_=cv2.calcOpticalFlowPyrLK(previous,following,start,None,**options)
    if forward is None:
        okay[:]=False
        return updated,okay,errors
    backward,back_status,_=cv2.calcOpticalFlowPyrLK(following,previous,forward,None,**options)
    if backward is None:
        okay[:]=False
        return updated,okay,errors
    error=np.linalg.norm(backward.reshape(-1,2)-start.reshape(-1,2),axis=1)
    xy=forward.reshape(-1,2)
    height,width=following.shape[:2]
    passed=(status.ravel().astype(bool)&back_status.ravel().astype(bool)&np.isfinite(xy).all(axis=1)&
            np.isfinite(error)&(error<=1.5)&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height))
    updated[indices]=xy
    okay[indices]=passed
    errors[indices]=error
    return updated,okay,errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-root','reference','selection','seeds','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    reference,root=args.reference.resolve(),args.input_root.resolve()
    if args.output.resolve().is_relative_to(root) or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error('Drafts must remain outside the source video tree and repository')
    if args.output.exists():
        raise FileExistsError('Existing propagated drafts are preserved')
    frames=json.loads((reference/'annotations.json').read_text())['frames']
    seeds=json.loads(args.seeds.read_text())['frames']
    rows=json.loads(args.selection.read_text())
    if len(rows)>20 or len(frames)>200:
        parser.error('This annotation helper is bounded to the comparison reference set')
    selected={row['pair_id']:row for row in rows}
    grouped=defaultdict(list)
    for frame in frames:
        grouped[(frame['pair_id'],frame['view'])].append(frame)
    import cv2
    cv2.setNumThreads(2)
    output={}
    for (pair,view),group in grouped.items():
        group.sort(key=lambda f:f['frame_index'])
        seed_frame=group[0]
        seed=seeds[seed_frame['id']]
        image=reference/seed_frame['image']
        if sha256(image)!=seed['provenance']['image_sha256']:
            raise ValueError('Seed image differs from annotation provenance')
        names=[('pose',name) for name in seed['pose']]+[('face',name) for name in seed['face']['anchors']]
        points=np.array([seed['pose'][name] if group_name=='pose' else seed['face']['anchors'][name]
                         for group_name,name in names],dtype=np.float32)
        valid=np.ones(len(points),dtype=bool)
        cumulative=np.zeros(len(points),dtype=np.float32)
        path=(root/selected[pair]['views'][view]['relpath']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Invalid source video path')
        capture=cv2.VideoCapture(str(path))
        targets={frame['frame_index']:frame for frame in group}
        previous=None
        try:
            for index in range(group[-1]['frame_index']+1):
                okay,bgr=capture.read()
                if not okay:
                    raise ValueError('Video ended before a reference frame')
                if index<seed_frame['frame_index']:
                    continue
                gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                if previous is not None:
                    points,valid,error=flow_step(previous,gray,points,valid)
                    cumulative+=np.nan_to_num(error)
                previous=gray
                if index in targets:
                    frame=targets[index]
                    pose={};anchors={};quality={}
                    for i,(area,name) in enumerate(names):
                        if valid[i]:
                            (pose if area=='pose' else anchors)[name]=points[i].tolist()
                        quality[area+'/'+name]={'track_valid':bool(valid[i]),'cumulative_fb_error_px':float(cumulative[i])}
                    output[frame['id']]={'pose':pose,'face':{'visible':seed['face']['visible'] if index==seed_frame['frame_index'] else None,
                                                           'anchors':anchors},
                                         'reviewed':False,'seed_frame_id':seed_frame['id'],
                                         'image_sha256':sha256(reference/frame['image']),
                                         'method':'manual_seed' if index==seed_frame['frame_index'] else 'pyramidal_LK_forward_backward',
                                         'point_tracking_diagnostics':quality}
        finally:
            capture.release()
        print(json.dumps({'view':view,'completed_groups':len(output)//5}),flush=True)
    if set(output)!={f['id'] for f in frames}:
        raise ValueError('Draft reference coverage differs')
    result={'schema':'propagated_reference_drafts_v1','frames':output,
            'provenance':{'seed_labels_sha256':sha256(args.seeds),'selection_sha256':sha256(args.selection),
                          'code_sha256':sha256(Path(__file__)),'opencv':cv2.__version__,
                          'flow_fb_threshold_px':1.5,'status':'requires_visual_review',
                          'note':'No compared landmark model supplies these drafts. Optical flow may drift; tracked points are not automatically reviewed or physically visible.'}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)


if __name__=='__main__':
    main()
