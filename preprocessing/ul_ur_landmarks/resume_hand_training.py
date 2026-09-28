"""Resume a verified local hand experiment after a process interruption."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

from .detector_pilot import write_json
from .schema import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    root=args.experiment.resolve()
    identity=json.loads((root/'experiment.json').read_text())
    if identity['schema']!='hand_training_experiment_v1':
        raise ValueError('Unknown experiment format')
    if (root/'training_completion.json').exists():
        saved=json.loads((root/'training_completion.json').read_text())
        if sha256(Path(saved['best_checkpoint']))!=saved['sha256']:
            raise ValueError('Completed checkpoint changed')
        print('Training already completed')
        return
    for frame in identity['frames']:
        image=root/'dataset/images'/frame['split']/(frame['id']+'.jpg')
        label=root/'dataset/labels'/frame['split']/(frame['id']+'.txt')
        if sha256(image)!=frame['image_sha256'] or label.read_text()!=frame['labels']:
            raise ValueError('Frozen training data changed')
    checkpoint=root/'fit/weights/last.pt'
    os.environ.update(YOLO_CONFIG_DIR=str(root/'yolo_config'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',
                      WANDB_MODE='disabled',COMET_MODE='DISABLED',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    from .pose_models import require_cuda
    runtime=require_cuda()
    import torch
    from ultralytics import YOLO,settings
    torch.set_num_threads(4)
    settings.update({key:False for key in ('sync','wandb','comet','clearml','neptune','mlflow','dvc','hub') if key in settings})
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    train_args=saved['train_args']
    if (Path(train_args['data']).resolve()!=root/'dataset/data.yaml' or
            Path(train_args['project']).resolve()!=root or train_args['name']!='fit' or saved['epoch']<0):
        raise ValueError('Checkpoint belongs to another run or has no resumable state')
    stage={'started_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':sha256(checkpoint),
           'last_completed_epoch':saved['epoch']+1,'runtime':runtime,
           'packages':{name:version(name) for name in ('torch','torchvision','ultralytics','numpy')},
           'code_sha256':sha256(Path(__file__)),'cublas_workspace_config':':4096:8',
           'note':'Original stage warned that CuBLAS workspace determinism was unset; enabled for resume. Not a bitwise replay of uninterrupted training.'}
    del saved
    with (root/'resume_stages.jsonl').open('a') as stream:
        stream.write(json.dumps(stage)+'\n')
        stream.flush()
    model=YOLO(str(checkpoint),task='detect')
    model.train(resume=True)
    best=root/'fit/weights/best.pt'
    write_json(root/'training_completion.json',{'best_checkpoint':str(best),'sha256':sha256(best),
                                               'experiment_sha256':sha256(root/'experiment.json'),
                                               'completed_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':
    main()
