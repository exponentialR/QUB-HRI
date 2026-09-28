"""Evaluate dense MediaPipe face recovery using saved whole-body pose crops."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .detector_pilot import write_json
from .pose_reference import reference_frames
from .schema import sha256
from .task_crops import CropTasks,face_rect_from_wholebody,face_rotation_from_wholebody


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','pose-predictions','models-dir','output-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--align-eyes',action='store_true')
    args=parser.parse_args()
    reference,output=args.reference.resolve(),args.output_root.resolve()
    if any(output.is_relative_to(p) for p in (reference,Path(__file__).resolve().parents[2])):
        parser.error('Output must be outside reference directory and repository')
    frames,identity=reference_frames(reference,200)
    pose=json.loads(args.pose_predictions.read_text())
    if pose['topology']!='coco_wholebody_133' or pose['provenance']['reference']!=identity:
        raise ValueError('Pose source topology or reference images differ')
    if [f['id'] for f in pose['frames']] != [f['id'] for f in frames]:
        raise ValueError('Pose and reference order differs')
    output.mkdir(parents=True,exist_ok=True)
    (output/'overlays').mkdir(exist_ok=True)
    import cv2
    cv2.setNumThreads(2)
    with CropTasks(args.models_dir,hands=False) as tasks:
        provenance={'reference':identity,'pose_predictions_sha256':sha256(args.pose_predictions),
                    'align_eyes':args.align_eyes,
                    'tasks':tasks.provenance,'code_sha256':sha256(Path(__file__)),
                    'crop_adapter_sha256':sha256(Path(__file__).with_name('task_crops.py'))}
        destination=output/'predictions.json'
        if destination.exists():
            saved=json.loads(destination.read_text())
            if saved['provenance']!=provenance:
                raise ValueError('Existing face predictions differ; choose a new directory')
            print(json.dumps({'status':'skipped_valid','frames':len(saved['frames'])}))
            return
        predictions=[]
        counts=Counter()
        for frame,body in zip(frames,pose['frames']):
            image=cv2.imread(str(reference/frame['image']))
            if image is None or image.shape[:2]!=(frame['height'],frame['width']):
                raise ValueError('Unexpected reference image dimensions')
            xy=np.asarray(body['xy_px'],dtype=np.float32)
            rect=face_rect_from_wholebody(xy,frame['width'],frame['height'])
            angle=face_rotation_from_wholebody(xy) if args.align_eyes else 0.
            started=time.perf_counter()
            points,confidence=tasks.face_points(image,rect,angle)
            elapsed=time.perf_counter()-started
            valid=np.zeros(478,dtype=bool) if points is None else np.isfinite(points).all(axis=1)&(confidence>0)
            counts[frame['view']+'_frames']+=1
            counts[frame['view']+'_face_returned']+=bool(valid.any())
            if points is not None:
                for point in points[valid]:
                    cv2.circle(image,tuple(map(round,point)),1,(40,230,50),-1)
            if rect:
                cv2.rectangle(image,tuple(rect[:2]),tuple(rect[2:]),(255,150,0),2)
            if not cv2.imwrite(str(output/'overlays'/Path(frame['image']).name),image):
                raise OSError('Cannot save face overlay')
            predictions.append({'id':frame['id'],'view':frame['view'],'crop_xyxy_px':rect,
                                'rotation_degrees':angle,
                                'crop_to_source_3x3':tasks.last_face_crop_to_source.tolist() if rect else None,
                                'xy_px':[[float(x),float(y)] if v else [None,None] for (x,y),v in zip(points,valid)] if points is not None else [[None,None]]*478,
                                'confidence':confidence.tolist() if confidence is not None else [0.]*478,
                                'valid':valid.tolist(),'inference_s':elapsed})
        saved={'schema':'face_reference_v1','topology':'mediapipe_478','provenance':provenance,
               'frames':predictions,'coverage_counts':dict(counts),
               'coverage_note':'Returned predictions are not measured recall or accuracy; use visible labelled anchors.'}
        write_json(destination,saved)
        print(json.dumps({'frames':len(predictions),'coverage_counts':dict(counts),
                          'face_inference_s':sum(f['inference_s'] for f in predictions)},indent=2))


if __name__=='__main__':
    main()
