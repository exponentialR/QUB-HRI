"""Compare sequential and native dynamic-batch pose inference on local reference crops."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .detector_pilot import write_json
from .hand_models import HandPose
from .pose_models import RTMW
from .schema import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','pose-predictions','models-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep the previous benchmark; use a new output path')
    import cv2
    cv2.setNumThreads(2)
    all_frames=json.loads((args.reference/'annotations.json').read_text())['frames']
    reference={f['id']:f for f in all_frames}
    poses=json.loads(args.pose_predictions.read_text())['frames']
    selected=[f for f in poses if f['bbox_xyxy_px'] is not None][::12][:16]
    images=[cv2.imread(str(args.reference/reference[f['id']]['image'])) for f in selected]
    boxes=[f['bbox_xyxy_px'] for f in selected]
    hand_samples=[(frame,hand) for frame in all_frames for hand in frame['hands']][::29][:16]
    hand_images=[cv2.imread(str(args.reference/frame['image'])) for frame,_ in hand_samples]
    hand_boxes=[[h['bbox'][0],h['bbox'][1],h['bbox'][0]+h['bbox'][2],h['bbox'][1]+h['bbox'][3]] for _,h in hand_samples]
    result={}
    for name,model,image_list,box_list in (
        ('rtmw',RTMW(args.models_dir/'rtmw_l_384x288.onnx'),images,boxes),
        ('hand5',HandPose(args.models_dir/'rtmpose_m_hand5_256.onnx'),hand_images,hand_boxes)):
        model(image_list[0],np.asarray(box_list[0]))
        start=time.perf_counter()
        singles=[model(image,np.asarray(box)) for image,box in zip(image_list,box_list)]
        sequential_s=time.perf_counter()-start
        single_xy=np.stack([x[0] for x in singles]);single_scores=np.stack([x[1] for x in singles])
        sizes={}
        for batch_size in (1,2,4,8,16):
            model.batch(image_list[:batch_size],box_list[:batch_size])
            times=[]
            for repeat in range(3):
                start=time.perf_counter()
                batches=[model.batch(image_list[i:i+batch_size],box_list[i:i+batch_size])
                         for i in range(0,len(image_list),batch_size)]
                times.append(time.perf_counter()-start)
            xy=np.concatenate([x[0] for x in batches]);scores=np.concatenate([x[1] for x in batches])
            available=(single_scores>=.3)&(scores>=.3)
            delta=np.linalg.norm(xy-single_xy,axis=-1)
            sizes[str(batch_size)]={'median_total_s':float(np.median(times)),
                                    'median_ms_per_crop':float(np.median(times)*1000/len(image_list)),
                                    'max_point_delta_px':float(np.max(delta)),
                                    'max_confident_point_delta_px':float(np.max(delta[available])) if available.any() else None,
                                    'max_score_delta':float(np.max(np.abs(single_scores-scores))),
                                    'confidence_threshold_disagreements':int(np.count_nonzero((single_scores>=.3)!=(scores>=.3)))}
        result[name]={'crops':len(image_list),'sequential_s':sequential_s,'batches':sizes,'model':model.provenance}
    saved={'schema':'native_batch_benchmark_v1','models':result,
           'reference_sha256':sha256(args.reference/'annotations.json'),
           'pose_predictions_sha256':sha256(args.pose_predictions),'code_sha256':sha256(Path(__file__)),
           'note':'Official ONNX files already have dynamic batch axes. No checkpoint bytes, weights, or graph operators were changed. Timings include crop preprocessing and decoding.'}
    write_json(args.output,saved)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
