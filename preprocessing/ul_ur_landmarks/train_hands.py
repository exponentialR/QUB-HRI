"""Prepare a participant-separated, local hand detector experiment from drafts."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .detector_pilot import write_json
from .runner import freeze_run_scope
from .schema import sha256


CLASSES = {'participant': 0, 'other_actor': 1}


def split_participants(ids: list[str], seed: int = 42, val_count: int = 5) -> dict[str,str]:
    unique = sorted(set(ids), key=lambda pid: hashlib.sha256(f'{seed}:{pid}'.encode()).hexdigest())
    if not 0 < val_count < len(unique):
        raise ValueError('Both participant splits must be nonempty')
    return {pid: ('val' if i < val_count else 'train') for i,pid in enumerate(unique)}


def extend_splits(ids: list[str], previous: dict[str,str]) -> dict[str,str]:
    """Preserve the earlier development split; newly labelled IDs train only."""
    if not set(previous)<=set(ids) or set(previous.values())!={'train','val'}:
        raise ValueError('Extended reference must retain all previous train/val participants')
    return {pid:previous.get(pid,'train') for pid in sorted(set(ids))}


def yolo_labels(frame: dict) -> str:
    lines = []
    width,height = frame['width'],frame['height']
    for hand in frame['hands']:
        if hand['actor'] not in CLASSES:
            raise ValueError('Unknown actor cannot become a training label or a negative')
        x,y,w,h = hand['bbox']
        if not np.isfinite([x,y,w,h]).all() or min(x,y) < 0 or min(w,h) <= 0 or x+w > width or y+h > height:
            raise ValueError('Invalid reference hand bounds')
        lines.append(f"{CLASSES[hand['actor']]} {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}")
    return '\n'.join(lines) + ('\n' if lines else '')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','selection','weights','output-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--extend-experiment',type=Path,
                        help='Preserve the completed earlier experiment split and add new IDs to training')
    args=parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error('Epochs and batch size must be positive')
    reference,output=args.reference.resolve(),args.output_root.resolve()
    if any(output.is_relative_to(p) for p in (reference,Path(__file__).resolve().parents[2])):
        parser.error('Training artifacts must be outside reference data and repository')
    annotations=reference/'annotations.json'
    rows=json.loads(args.selection.read_text())
    participants={r['pair_id']:r['pid'] for r in rows}
    previous=None
    if args.extend_experiment:
        previous=json.loads((args.extend_experiment/'experiment.json').read_text())
        completion=json.loads((args.extend_experiment/'training_completion.json').read_text())
        if (completion['experiment_sha256']!=sha256(args.extend_experiment/'experiment.json') or
                completion['sha256']!=sha256(args.weights)):
            raise ValueError('Parent experiment or initialization checkpoint identity mismatch')
        splits=extend_splits(list(participants.values()),previous['participant_splits'])
    else:
        splits=split_participants(list(participants.values()))
    frames=json.loads(annotations.read_text())['frames']
    accepted=[]
    excluded=[]
    for frame in frames:
        if (not frame.get('hands_checked') or frame.get('hand_annotation_provenance',{}).get('needs_review') or
                any(h['actor'] not in CLASSES for h in frame['hands'])):
            excluded.append(frame['id'])
            continue
        image=(reference/frame['image']).resolve()
        if not image.is_relative_to(reference/'images') or not image.is_file():
            raise ValueError('Invalid local reference path')
        image_hash=sha256(image)
        if image_hash != frame['hand_annotation_provenance']['image_sha256']:
            raise ValueError('Reference image differs from annotation provenance')
        labels=yolo_labels(frame)
        pid=participants[frame['pair_id']]
        accepted.append({'id':frame['id'],'image':str(image),'image_sha256':image_hash,'pid':pid,
                         'view':frame['view'],'pair_id':frame['pair_id'],'split':splits[pid],
                         'labels':labels})
    config={'epochs':args.epochs,'batch':args.batch,'imgsz':1280,'seed':42,'optimizer':'AdamW',
            'lr0':.001,'lrf':.01,'patience':30,'close_mosaic':10,'workers':2,'cache':False,
            'device':0,'deterministic':True,'amp':False,'plots':True,'save_period':25,
            'fliplr':.5,'flipud':0.,'degrees':10.,'translate':.1,'scale':.3,'mosaic':1.,
            'mixup':0.,'perspective':0.}
    identity={'schema':'hand_training_experiment_v1','annotations_sha256':sha256(annotations),
              'selection_sha256':sha256(args.selection),'initial_weights_sha256':sha256(args.weights),
              'code_sha256':sha256(Path(__file__)),'classes':CLASSES,'participant_splits':splits,
              'frames':accepted,'excluded_frame_ids':excluded,'configuration':config,
              'limitations':['Provisional assistant labels. All reference frames have informed model development.',
                             'Participant separation prevents training/validation leakage within this experiment.',
                             'Prior checkpoint training overlap is unknown; this is not an independent final test.']}
    if previous is not None:
        earlier={f['id']:f for f in previous['frames']}
        now={f['id']:f for f in accepted}
        for fid,old in earlier.items():
            if fid not in now or any(old[k]!=now[fid][k] for k in ('image_sha256','pid','view','pair_id','split','labels')):
                raise ValueError('Extended experiment altered an earlier accepted frame')
        identity['parent_experiment_sha256']=sha256(args.extend_experiment/'experiment.json')
        identity['added_participant_role']='training; earlier challenge is now development data'
    output.mkdir(parents=True,exist_ok=True)
    freeze_run_scope(output/'experiment.json',identity)
    for item in accepted:
        images=output/'dataset/images'/item['split'];labels=output/'dataset/labels'/item['split']
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        target=images/(item['id']+'.jpg')
        if target.exists():
            if sha256(target)!=item['image_sha256']:
                raise ValueError('Existing training image differs')
        else:
            # Independent copies prevent training caches from touching source frames.
            shutil.copyfile(item['image'],target)
        label_path=labels/(item['id']+'.txt')
        if label_path.exists() and label_path.read_text()!=item['labels']:
            raise ValueError('Existing training labels differ')
        label_path.write_text(item['labels'])
    dataset=output/'dataset/data.yaml'
    dataset.write_text(json.dumps({'path':str(dataset.parent),'train':'images/train','val':'images/val',
                                   'names':{0:'participant_hand',1:'surrogate_hand'}},indent=2)+'\n')
    summary={'frames':dict(Counter(item['split'] for item in accepted)),
             'participants':dict(Counter(splits.values())),'excluded_frames':len(excluded),'config':config}
    write_json(output/'preparation_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    if not args.train:
        return
    if (output/'fit').exists():
        raise FileExistsError('Training directory already exists; resume its saved checkpoint explicitly')
    os.environ.update(YOLO_CONFIG_DIR=str(output/'yolo_config'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      WANDB_MODE='disabled',COMET_MODE='DISABLED')
    from .pose_models import require_cuda
    require_cuda()
    import torch
    from ultralytics import YOLO,settings
    torch.set_num_threads(4)
    settings.update({key:False for key in ('sync','wandb','comet','clearml','neptune','mlflow','dvc','hub') if key in settings})
    model=YOLO(str(args.weights.resolve()),task='detect')
    model.train(data=str(dataset),project=str(output),name='fit',exist_ok=False,**config)
    best=output/'fit/weights/best.pt'
    write_json(output/'training_completion.json',{'best_checkpoint':str(best),'sha256':sha256(best),
                                                'experiment_sha256':sha256(output/'experiment.json')})


if __name__=='__main__':
    main()
