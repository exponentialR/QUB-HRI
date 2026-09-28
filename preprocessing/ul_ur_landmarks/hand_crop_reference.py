"""Measure MediaPipe hand availability on known boxes; this is an oracle diagnostic."""

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .collection_backend import point_box,supported_by_box
from .detector_pilot import box_iou,write_json
from .pose_reference import reference_frames
from .schema import sha256
from .task_crops import CropTasks,square_rect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','models-dir','output-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    reference=args.reference.resolve();output=args.output_root.resolve()
    if output.exists() or any(output.is_relative_to(p) for p in (reference,Path(__file__).resolve().parents[2])):
        parser.error('Choose a new output root outside reference and repository')
    frames,identity=reference_frames(reference,200)
    output.mkdir(parents=True)
    import cv2
    cv2.setNumThreads(2)
    predictions=[];counts=defaultdict(Counter);started=time.perf_counter();inference_s=0.
    with CropTasks(args.models_dir,face=False) as tasks:
        for frame in frames:
            bgr=cv2.imread(str(reference/frame['image']));detections=[]
            if bgr is None or bgr.shape[:2]!=(frame['height'],frame['width']):
                raise ValueError('Invalid reference image')
            for index,label in enumerate(frame['hands']):
                x,y,w,h=label['bbox'];box=np.array([x,y,x+w,y+h]);rect=square_rect(box,frame['width'],frame['height'],padding=2.)
                begin=time.perf_counter();hands=tasks.hand_points(bgr,rect);inference_s+=time.perf_counter()-begin
                candidates=[hand for hand in hands if supported_by_box(hand.xy,box) and
                            point_box(hand.xy) is not None and box_iou(point_box(hand.xy),box)>=.5]
                selected=max(candidates,key=lambda hand:box_iou(point_box(hand.xy),box)) if candidates else None
                detections.append({'reference_hand_index':index,'actor':label['actor'],'bbox_xyxy_px':box.tolist(),
                    'crop_xyxy_px':list(rect),'all_returned_hands':len(hands),'supported_hand':selected is not None,
                    'xy_px':[[float(v) if np.isfinite(v) else None for v in point] for point in selected.xy] if selected else None,
                    'confidence':selected.confidence.tolist() if selected else None})
                if not frame.get('hand_annotation_provenance',{}).get('needs_review'):
                    c=counts[frame['view']];c['labelled_'+label['actor']]+=1;c['supported_'+label['actor']]+=selected is not None
            predictions.append({'id':frame['id'],'detections':detections})
        provenance={'reference':identity,'annotations_sha256':sha256(reference/'annotations.json'),
            'oracle_boxes':True,'tasks':tasks.provenance,'box_support_iou':.5,
            'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                           ('hand_crop_reference.py','task_crops.py','collection_backend.py')},
            'inference_s':inference_s,'elapsed_s':time.perf_counter()-started}
    write_json(output/'predictions.json',{'schema':'mediapipe_hand_oracle_v1','provenance':provenance,'frames':predictions})
    metrics={'provenance':provenance,'views':dict(counts),'decision_ready':False,
        'limitations':['Known boxes cannot measure detector recall.','Returned joints are availability, not joint accuracy.',
                       'Assistant boxes are provisional; ambiguous frames excluded from counts.']}
    write_json(output/'metrics.json',metrics)
    print(json.dumps({'views':dict(counts),'inference_s':inference_s},indent=2))


if __name__=='__main__':
    main()
