"""Freeze fresh participant clips for a local hand-detector challenge set."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .detector_pilot import write_json
from .runner import frame_pts_ms
from .schema import sha256


def choose_rows(inventory:list[dict],excluded:set[str],participants:int=5) -> list[dict]:
    eligible=defaultdict(list)
    for row in inventory:
        if row['pid'] in excluded:
            continue
        if all(3<=m['duration_s']<=8 and (m['declared_frames'] or 0)>=30 for m in row['views'].values()):
            eligible[row['pid']].append(row)
    def rank(value):
        return hashlib.sha256(('qub_hand_holdout_20260926:'+value).encode()).hexdigest()
    ids=sorted([pid for pid,rows in eligible.items() if len({r['subtask_dir'] for r in rows})>=2],key=rank)[:participants]
    if len(ids)!=participants:
        raise ValueError('Insufficient fresh participants with two eligible subtasks')
    result=[]
    for pid in ids:
        candidates=sorted(eligible[pid],key=lambda r:rank(r['pair_id']))
        first=candidates[0]
        second=next(r for r in candidates if r['subtask_dir']!=first['subtask_dir'])
        result.extend([first,second])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input-root','inventory','existing-selection','trained-checkpoint','output-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    root,output=args.input_root.resolve(),args.output_root.resolve()
    if output.is_relative_to(root) or output.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error('Challenge frames must be stored privately outside source videos and repository')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Challenge set already exists; do not reselect or overwrite it')
    scope=json.loads(args.inventory.with_name('inventory_scope.json').read_text())
    if scope['scope']!='all_local_ul_ur' or scope['input_root']!=str(root):
        raise ValueError('Expected the frozen all-participant inventory')
    excluded={r['pid'] for r in json.loads(args.existing_selection.read_text())}
    selected=choose_rows(json.loads(args.inventory.read_text()),excluded)
    output.mkdir(parents=True,exist_ok=True)
    (output/'reference/images').mkdir(parents=True)
    write_json(output/'selection.json',selected)
    write_json(output/'holdout_scope.json',{'scope':'local_hand_challenge','input_root':str(root),
        'parent_inventory_sha256':sha256(args.inventory),'excluded_development_ids':sorted(excluded),
        'selected_ids':sorted({r['pid'] for r in selected}),'checkpoint_frozen_before_annotation_sha256':sha256(args.trained_checkpoint),
        'selection_rule':'deterministic hash order; five fresh participants, two distinct 3–8 second subtasks each',
        'frames_per_clip':'10%, 50%, 90% of decoded indices','code_sha256':sha256(Path(__file__)),
        'limitations':'Excluded from every participant in the supplied development selection. Initial LEGO checkpoint training overlap is unknown.'})
    import cv2
    cv2.setNumThreads(2)
    frames=[]
    for number,row in enumerate(selected):
        for view,meta in row['views'].items():
            path=(root/meta['relpath']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Invalid source path')
            pts=frame_pts_ms(path)
            if meta['declared_frames'] is not None and len(pts)!=meta['declared_frames']:
                raise ValueError('Decoded reference frame count differs from inventory')
            positions=sorted({round((len(pts)-1)*fraction) for fraction in (.1,.5,.9)})
            capture=cv2.VideoCapture(str(path))
            source_hash=sha256(path)
            try:
                for index in range(positions[-1]+1):
                    okay,bgr=capture.read()
                    if not okay or bgr.shape[:2]!=(meta['height'],meta['width']):
                        raise ValueError('Source decoding mismatch')
                    if index not in positions:
                        continue
                    fid=f'holdout{number:02d}_{view}_{index:05d}'
                    image=output/'reference/images'/f'{fid}.jpg'
                    if not cv2.imwrite(str(image),bgr,[cv2.IMWRITE_JPEG_QUALITY,95]):
                        raise OSError('Cannot write local challenge image')
                    frames.append({'id':fid,'image':f'images/{fid}.jpg','pair_id':row['pair_id'],'view':view,
                                   'frame_index':index,'timestamp_ms':pts[index],'width':meta['width'],'height':meta['height'],
                                   'reviewed':False,'hands_checked':False,'hands':[],'pose':{},
                                   'face':{'visible':None,'anchors':{}},'body_checked':False,'face_checked':False,
                                   'source_video_sha256':source_hash,'decoded_bgr_sha256':hashlib.sha256(bgr.tobytes()).hexdigest(),
                                   'image_sha256':sha256(image)})
            finally:
                capture.release()
    write_json(output/'reference/annotations.json',{'schema':'local_hand_challenge_annotations_v1','frames':frames})
    print(json.dumps({'pairs':len(selected),'participants':len({r['pid'] for r in selected}),'frames':len(frames),
                      'output_root':str(output),'status':'awaiting_visual_annotation'}))


if __name__=='__main__':
    main()
