"""Bounded hand-model diagnostics with independent detection and known-box modes."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .detector_pilot import HAND_ACTORS, match_boxes, write_json
from .hand_models import HandDetector, HandPose
from .pose_models import visible_points
from .pose_reference import reference_frames
from .schema import sha256


def score(frames: list[dict], predicted: list[dict], iou: float = .5) -> dict:
    """Box recall is independent of actor and of whether landmarks were returned."""
    counts = defaultdict(Counter)
    for frame, prediction in zip(frames, predicted):
        if frame['id'] != prediction['id']:
            raise ValueError("Frame order differs")
        if not frame.get('hands_checked') or frame.get('hand_annotation_provenance', {}).get('needs_review'):
            continue
        c = counts[frame['view']]
        hands, detections = frame['hands'], prediction['detections']
        pairs = match_boxes(hands, detections, iou)
        c['frames'] += 1
        c['predicted_hands'] += len(detections)
        c['matched_hands'] += len(pairs)
        for hand in hands:
            c['labelled_' + hand['actor']] += 1
        for i, j in pairs:
            c['detected_' + hands[i]['actor']] += 1
            c['with_three_valid_joints_' + hands[i]['actor']] += sum(detections[j]['valid']) >= 3
            c['actor_correct'] += hands[i]['actor'] == detections[j]['actor'] and hands[i]['actor'] != 'unknown'
        c['hand_absent_frames'] += not hands
        c['false_hand_absent_frames'] += not hands and bool(detections)
    views = {}
    for view, c in counts.items():
        ratio = lambda n, d: c[n] / c[d] if c[d] else None
        views[view] = {**c, 'other_actor_box_recall': ratio('detected_other_actor', 'labelled_other_actor'),
                       'participant_box_recall': ratio('detected_participant', 'labelled_participant'),
                       'hand_box_precision': ratio('matched_hands', 'predicted_hands'),
                       'actor_accuracy_matched': ratio('actor_correct', 'matched_hands')}
    return {'iou': iou, 'views': views, 'decision_ready': False,
            'limitations': ['Assistant hand-box drafts, ambiguous frames excluded.',
                            'Returned joints do not establish joint accuracy or physical visibility.',
                            'Unknown actors count as incorrect assignments; native RTMDet has no actor classifier.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'models-dir', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('rtmdet', 'rtmdet_tiled', 'yolo_boxes', 'reference_boxes'), required=True)
    parser.add_argument('--detections', type=Path)
    parser.add_argument('--max-frames', type=int, default=200)
    parser.add_argument('--confidence', type=float, default=.05)
    parser.add_argument('--joint-threshold', type=float, default=.3)
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 200 or not 0 < args.confidence < 1 or not 0 < args.joint_threshold < 1:
        parser.error('Invalid frame count or confidence')
    if (args.mode == 'yolo_boxes') != bool(args.detections):
        parser.error('--detections is required only for yolo_boxes mode')
    reference, output = args.reference.resolve(), args.output_root.resolve()
    if any(output.is_relative_to(p) for p in (reference, Path(__file__).resolve().parents[2])):
        parser.error('Output must be outside reference frames and repository')
    output.mkdir(parents=True, exist_ok=True)
    frames, identity = reference_frames(reference, args.max_frames)
    source = None
    if args.detections:
        source = json.loads(args.detections.read_text())
        if source['provenance']['image_sha256'] != {f['id']: f['image_sha256'] for f in identity['frames']}:
            raise ValueError('Detection image identities differ from reference')
        source = {f['id']: f for f in source['frames']}
    detector = HandDetector(args.models_dir / 'rtmdet_nano_hand_320.onnx', args.confidence,
                            args.mode == 'rtmdet_tiled') if args.mode.startswith('rtmdet') else None
    model = HandPose(args.models_dir / 'rtmpose_m_hand5_256.onnx')
    provenance = {'reference': identity, 'mode': args.mode, 'pose': model.provenance,
                  'detector': detector.provenance if detector else None,
                  'detection_source_sha256': sha256(args.detections) if args.detections else None,
                  'annotations_sha256': sha256(reference / 'annotations.json'),
                  'code_sha256': sha256(Path(__file__)),
                  'adapter_sha256': sha256(Path(__file__).with_name('hand_models.py')),
                  'confidence': args.confidence, 'joint_threshold': args.joint_threshold,
                  'oracle_boxes': args.mode == 'reference_boxes'}
    destination = output / 'predictions.json'
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing['provenance'] != provenance:
            raise ValueError('Existing hand outputs differ; choose a new directory')
        print(json.dumps({'status': 'skipped_valid', 'frames': len(existing['frames'])}))
        return
    import cv2
    cv2.setNumThreads(2)
    (output / 'overlays').mkdir(exist_ok=True)
    predicted = []
    started = time.perf_counter()
    for frame in frames:
        image = cv2.imread(str(reference / frame['image']))
        if image is None or image.shape[:2] != (frame['height'], frame['width']):
            raise ValueError('Reference image dimensions differ')
        start = time.perf_counter()
        if detector:
            detections = detector(image)
        elif source is not None:
            detections = [dict(d, actor=HAND_ACTORS[d['class_name']]) for d in source[frame['id']]['detections']
                          if d['class_name'] in HAND_ACTORS and d['confidence'] >= args.confidence]
        else:
            detections = [{'bbox_xyxy_px': [h['bbox'][0], h['bbox'][1],
                                             h['bbox'][0] + h['bbox'][2], h['bbox'][1] + h['bbox'][3]],
                           'confidence': 1., 'actor': h['actor'], 'class_name': 'reference_hand',
                           'reference_hand_index': i} for i, h in enumerate(frame['hands'])]
        detection_s = time.perf_counter() - start
        pose_start = time.perf_counter()
        for detection in detections:
            points, confidence = model(image, detection['bbox_xyxy_px'])
            xy, raw, valid = visible_points(points, confidence, frame['width'], frame['height'], args.joint_threshold)
            valid &= model.last_crop_valid
            detection.update({'xy_px': [[float(x), float(y)] if v else [None, None] for (x,y),v in zip(xy,valid)],
                              'raw_xy_px': [[float(x),float(y)] if np.isfinite([x,y]).all() else [None,None] for x,y in points],
                              'joint_confidence': [float(s) if np.isfinite(s) else None for s in raw],
                              'valid': valid.tolist(), 'crop_interior': model.last_crop_valid.tolist(),
                              'handedness': 'unknown'})
        pose_s = time.perf_counter() - pose_start
        overlay = image.copy()
        for d in detections:
            x0, y0, x1, y1 = map(round, d['bbox_xyxy_px'])
            color = (255, 150, 0) if d['actor'] == 'other_actor' else (40, 220, 40)
            cv2.rectangle(overlay, (x0,y0), (x1,y1), color, 2)
            cv2.putText(overlay, f"{d['actor']} {d['confidence']:.2f}", (x0,max(15,y0-4)), cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            for xy, v in zip(d['xy_px'], d['valid']):
                if v:
                    cv2.circle(overlay, tuple(map(round,xy)), 2, color, -1)
        if not cv2.imwrite(str(output / 'overlays' / Path(frame['image']).name), overlay):
            raise OSError('Cannot write hand overlay')
        predicted.append({'id': frame['id'], 'detections': detections, 'detection_s': detection_s, 'pose_s': pose_s})
        if len(predicted) % 20 == 0:
            print(json.dumps({'mode':args.mode,'completed_frames':len(predicted)}), flush=True)
    saved = {'schema': 'hand_reference_v1', 'topology': model.topology, 'provenance': provenance,
             'frames': predicted, 'wall_s_including_overlays': time.perf_counter() - started}
    write_json(destination, saved)
    metrics = {'predictions_sha256': sha256(destination), 'annotations_sha256': provenance['annotations_sha256'],
               'box_metrics': None if args.mode == 'reference_boxes' else score(frames, predicted),
               'box_metrics_reason': 'Known reference boxes cannot measure detector quality' if args.mode == 'reference_boxes' else None}
    write_json(output / 'metrics.json', metrics)
    print(json.dumps({'mode': args.mode,'frames': len(predicted), 'wall_s': saved['wall_s_including_overlays'],
                      'box_metrics': metrics['box_metrics']}, indent=2))


if __name__ == '__main__':
    main()
