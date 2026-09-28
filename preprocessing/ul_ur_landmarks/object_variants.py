"""Bounded LEGO inference diagnostics on the same local object reference images."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .collection_backend import YoloDetector
from .detector_pilot import box_iou
from .object_quality import GROUPS, score_frame
from .schema import sha256


def merge(rows, threshold=.5):
    kept = []
    for row in sorted(rows, key=lambda p: p['confidence'], reverse=True):
        if not any(row['class_name'] == other['class_name'] and
                   box_iou(row['box'], other['box']) > threshold for other in kept):
            kept.append(row)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('annotations', 'reference', 'weights', 'output-root'):
        parser.add_argument('--' + name, required=True, type=Path)
    a = parser.parse_args()
    reference = json.loads(a.annotations.read_text())
    if not 1 <= len(reference['frames']) <= 200:
        raise ValueError('Use 1–200 reference frames')
    a.output_root.mkdir(parents=True, exist_ok=False)
    import cv2
    import torch
    torch.set_num_threads(4)
    model = YoloDetector(a.weights, .1, a.output_root / 'yolo_config', expected_classes={
        *GROUPS, 'lefthand', 'righthand', 'surrogate_hand'})
    totals, frames = defaultdict(Counter), []
    elapsed = Counter()
    for frame in reference['frames'].values():
        path = a.reference / frame['image']
        if sha256(path) != frame['image_sha256']:
            raise ValueError('Reference image changed')
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        # Full width and lower two-thirds: fixed geometry, independent of labels/view.
        origin = round(height / 3)
        variants = {}
        for name, source, offset in [('full', image, 0), ('lower', image[origin:], origin)]:
            started = time.perf_counter()
            detected = model(source)
            elapsed[name] += time.perf_counter() - started
            variants[name] = [{'box': (p['bbox'] + [0, offset, 0, offset]).tolist(),
                               'confidence': p['confidence'], 'class_name': p['class_name'],
                               'class_id': p['class_id']} for p in detected]
        variants['merged'] = merge(variants['full'] + variants['lower'])
        frames.append({'id': frame['id'], 'variants': variants})
        for variant, rows in variants.items():
            for confidence in (.1, .15, .25):
                predictions = [p for p in rows if p['confidence'] >= confidence]
                for iou in (.3, .5):
                    scores, _ = score_frame(frame, predictions, iou)
                    for group, counts in scores.items():
                        totals[(variant, confidence, iou, frame['view'], group)].update(counts)
    summary = []
    for (variant, confidence, iou, view, group), row in totals.items():
        summary.append({'variant': variant, 'confidence': confidence, 'iou': iou, 'view': view,
                        'group': group, **row,
                        'recall': row['matched'] / row['labels'] if row['labels'] else None,
                        'precision': row['matched'] / row['scored_predictions'] if row['scored_predictions'] else None})
    result = {'schema': 'lego_inference_variants_v1', 'annotations_sha256': sha256(a.annotations),
              'weights_sha256': sha256(a.weights), 'code_sha256': sha256(Path(__file__)),
              'model': model.provenance, 'crop': 'full width, lower two-thirds, origin rounded(height/3)',
              'merge': 'class-aware confidence-ordered NMS, IoU 0.5', 'inference_s': dict(elapsed),
              'frames': frames, 'summary': summary,
              'limitations': 'Development diagnostic on 20 assistant-labelled frames, not independent validation. Original class semantics and pretraining overlap remain unverified.'}
    (a.output_root / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps([r for r in summary if r['iou'] == .5 and r['group'] == 'loose_brick'], indent=2))


if __name__ == '__main__':
    main()
