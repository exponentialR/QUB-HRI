"""Measure dense-face crop and detector-threshold variants on saved body predictions."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from .compare_reference import point_score, read_hdf5, summarize
from .evaluate import FACE_INDEX
from .schema import sha256
from .task_crops import CropTasks, face_rect_from_wholebody, face_rotation_from_wholebody, square_rect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('annotations', 'reference', 'results', 'models-dir', 'output-root'):
        parser.add_argument('--' + name, required=True, type=Path)
    a = parser.parse_args()
    reference = json.loads(a.annotations.read_text())
    if not 1 <= len(reference['frames']) <= 200:
        raise ValueError('Use 1–200 reference frames')
    a.output_root.mkdir(parents=True, exist_ok=False)
    import cv2
    import mediapipe as mp
    frames, totals, hashes, image_hashes = [], defaultdict(list), {}, {}
    start = time.perf_counter()
    with CropTasks(a.models_dir, face=False, hands=False) as task:
        for threshold in (.5, .2):
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(a.models_dir / 'face_landmarker.task')),
                running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1,
                min_face_detection_confidence=threshold)
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as model:
                task.face = model
                for frame in reference['frames']:
                    path = a.reference / frame['image']
                    image = cv2.imread(str(path))
                    digest = sha256(path)
                    if frame.get('image_sha256', digest) != digest:
                        raise ValueError('Reference image changed')
                    image_hashes[str(path)] = digest
                    _, pose, _, _, _, h5 = read_hdf5(frame, a.results)
                    hashes[str(h5)] = sha256(h5)
                    rect = face_rect_from_wholebody(pose, frame['width'], frame['height'])
                    angle = face_rotation_from_wholebody(pose)
                    variants = {}
                    for scale in (.75, 1., 1.4):
                        crop = square_rect(rect, frame['width'], frame['height'], padding=scale) if rect else None
                        xy, confidence = task.face_points(image, crop, angle)
                        returned = xy is not None and bool(np.isfinite(xy).all(axis=1).any())
                        points = []
                        for joint, truth in frame['face'].get('anchors', {}).items():
                            indices = list(FACE_INDEX[joint])
                            valid = returned and bool(np.isfinite(xy[indices]).all())
                            predicted = xy[indices].mean(axis=0) if valid else [np.nan, np.nan]
                            points.append(point_score(truth, predicted, valid, math.hypot(frame['width'], frame['height'])))
                        name = f'detect{threshold}_scale{scale}'
                        totals[(name, frame['view'])].extend(points)
                        variants[name] = {'returned': bool(returned), 'crop_xyxy_px': crop,
                                          'rotation_degrees': angle, 'anchor_scores': points,
                                          'xy_px': [[float(x), float(y)] if np.isfinite([x,y]).all() else [None,None] for x,y in xy] if returned else None}
                    frames.append({'id': frame['id'], 'view': frame['view'], 'variants': variants})
    summary = {name: {view: summarize(points) for (n,view),points in totals.items() if n == name}
               for name, _ in totals}
    result = {'schema': 'face_variants_v1', 'annotations_sha256': sha256(a.annotations),
              'source_output_sha256': hashes, 'image_sha256': image_hashes, 'code_sha256': sha256(Path(__file__)),
              'face_model_sha256': sha256(a.models_dir/'face_landmarker.task'),
              'presence_threshold': .5, 'face_scales_relative_to_original_crop': [.75,1.,1.4],
              'elapsed_s': time.perf_counter()-start, 'summary': summary, 'frames': frames,
              'limitations': 'Development comparison using assistant-labelled visible anchors. No face-absent negatives in this reference; false detections are not fully assessed.'}
    (a.output_root/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
