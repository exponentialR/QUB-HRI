"""Bounded diagnostic of fixed hand-detector crops and contrast transforms."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from .collection_backend import YoloDetector
from .detector_pilot import box_iou, score_hands, write_json
from .schema import sha256


def tile_rects(width, height):
    """Four overlapping two-thirds-size tiles, independent of annotations."""
    tw,th=round(width*2/3),round(height*2/3)
    return [(x,y,x+tw,y+th) for y in (0,height-th) for x in (0,width-tw)]


def merge_detections(detections, threshold=.5):
    kept=[]
    for item in sorted(detections,key=lambda d:d['confidence'],reverse=True):
        if not any(item['class_name']==other['class_name'] and
                   box_iou(item['bbox_xyxy_px'],other['bbox_xyxy_px'])>threshold for other in kept):
            kept.append(item)
    return kept


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--weights',required=True,type=Path)
    parser.add_argument('--baseline',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    args=parser.parse_args()
    reference=args.reference.resolve(); output=args.output_root.resolve()
    if output.is_relative_to(reference) or output.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error('Use an output root outside reference and repository')
    if output.exists():
        parser.error('Choose a new diagnostic output directory')
    annotations=reference/'annotations.json'
    frames=json.loads(annotations.read_text())['frames']
    if not 1<=len(frames)<=200 or not all(f.get('hands_checked') for f in frames):
        parser.error('Use 1–200 visually annotated local frames')
    baseline=json.loads(args.baseline.read_text())
    if baseline['provenance']['annotations_sha256']!=sha256(annotations):
        raise ValueError('Baseline annotation identity mismatch')
    if baseline['provenance']['weights_sha256']!=sha256(args.weights):
        raise ValueError('Baseline checkpoint identity mismatch')
    baseline_frames={f['id']:f for f in baseline['frames']}
    if set(baseline_frames)!={f['id'] for f in frames}:
        raise ValueError('Baseline frame identity mismatch')
    output.mkdir(parents=True)
    import cv2
    import torch
    torch.set_num_threads(4)
    detector=YoloDetector(args.weights,.25,output/'yolo_config',
                          expected_classes={'participant_hand','surrogate_hand'})
    gamma=np.round(255*(np.arange(256,dtype=np.float32)/255)**.5).astype(np.uint8)
    variants={name:[] for name in ('gamma_full','tiles','gamma_tiles','baseline_plus_tiles','baseline_plus_gamma_tiles')}
    started=time.perf_counter()
    inference_s=0.
    for frame in frames:
        image_path=(reference/frame['image']).resolve()
        if not image_path.is_relative_to(reference/'images') or sha256(image_path)!=frame['image_sha256']:
            raise ValueError('Reference image identity mismatch')
        bgr=cv2.imread(str(image_path))
        if bgr is None or bgr.shape[:2]!=(frame['height'],frame['width']):
            raise ValueError('Reference image dimensions mismatch')
        height,width=bgr.shape[:2]
        bright=cv2.LUT(bgr,gamma)
        local={}
        for mode,image,rects in (
            ('gamma_full',bright,[(0,0,width,height)]),
            ('tiles',bgr,tile_rects(width,height)),
            ('gamma_tiles',bright,tile_rects(width,height))):
            crops=[np.ascontiguousarray(image[y0:y1,x0:x1]) for x0,y0,x1,y1 in rects]
            begin=time.perf_counter();results=detector.batch(crops);inference_s+=time.perf_counter()-begin
            detections=[]
            for rect,items in zip(rects,results):
                x0,y0,_,_=rect
                for item in items:
                    detections.append({'bbox_xyxy_px':(item['bbox']+[x0,y0,x0,y0]).tolist(),
                        'confidence':item['confidence'],'class_id':item['class_id'],
                        'class_name':item['class_name'],'origin':mode,'crop':list(rect)})
            local[mode]=merge_detections(detections)
        for suffix in ('tiles','gamma_tiles'):
            local['baseline_plus_'+suffix]=merge_detections(baseline_frames[frame['id']]['detections']+local[suffix])
        for mode,items in local.items():
            variants[mode].append({'id':frame['id'],'detections':items})
        if len(variants['tiles'])%10==0:
            print(json.dumps({'frames':len(variants['tiles'])}),flush=True)
    all_frames=copy.deepcopy(frames)
    for frame in all_frames:
        frame['hand_annotation_provenance']['needs_review']=False
    metrics={mode:{'primary':score_hands(frames,items,.5),'including_ambiguous':score_hands(all_frames,items,.5)}
             for mode,items in variants.items()}
    metrics['baseline']={'primary':score_hands(frames,baseline['frames'],.5),
                         'including_ambiguous':score_hands(all_frames,baseline['frames'],.5)}
    provenance={'annotations_sha256':sha256(annotations),'weights_sha256':sha256(args.weights),
        'baseline_sha256':sha256(args.baseline),'detector':detector.provenance,
        'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                       ('detector_variants.py','collection_backend.py','detector_pilot.py')},
        'gamma':.5,'tiles':'2x2 overlapping two-thirds-size crops','merge_iou':.5,
        'status':'development diagnostic on previously evaluated challenge; not a fresh final test',
        'inference_s':inference_s,'elapsed_s':time.perf_counter()-started}
    write_json(output/'predictions.json',{'provenance':provenance,'variants':variants})
    write_json(output/'metrics.json',{'provenance':provenance,'variants':metrics})
    print(json.dumps({mode:{view:row['other_actor_recall'] for view,row in m['primary']['views'].items()}
                      for mode,m in metrics.items()},indent=2))


if __name__=='__main__':
    main()
