"""Summarize collection coverage and identify frames for visual audit, without accuracy claims."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .collection_status import ledger_rows
from .schema import ACTORS, sha256
from .locations import Locations


def frame_flags(pose, pose_valid, face_valid, hand_frames, actors, width, height):
    """Return diagnostic indices; flags require visual interpretation, not automatic rejection."""
    count = len(pose)
    upper = pose_valid[:, :11].sum(axis=1)
    common = pose_valid[1:, :11] & pose_valid[:-1, :11]
    delta = np.linalg.norm(pose[1:, :11] - pose[:-1, :11], axis=-1)
    jumps = np.any(common & (delta > .1 * np.hypot(width, height)), axis=1)
    flags = {'no_body_points': np.flatnonzero(upper == 0).tolist(),
             'fewer_than_four_body_points': np.flatnonzero(upper < 4).tolist(),
             'abrupt_body_point_change': (np.flatnonzero(jumps) + 1).tolist(),
             'dense_face_missing_with_coarse_face': np.flatnonzero(
                 ~face_valid.any(axis=1) & (pose_valid[:, 23:91].sum(axis=1) >= 10)).tolist()}
    for name, actor in ACTORS.items():
        observations = np.bincount(hand_frames[actors == actor], minlength=count)
        flags['more_than_two_' + name + '_hands'] = np.flatnonzero(observations > 2).tolist()
    return flags


def clip_summary(path):
    import h5py
    with h5py.File(path) as h:
        if str(h.attrs['pose_topology']) != 'coco_wholebody_133':
            raise ValueError('Collection audit expects explicitly named 133-point pose')
        source = json.loads(h.attrs['source_json'])
        pose, valid = h['participant/pose/xy_px'][:], h['participant/pose/valid'][:]
        face_valid = h['participant/face/valid'][:]
        hand_frames, actors = h['hands/frame_index'][:], h['hands/actor'][:]
        hand_valid = h['hands/valid'][:]
        tracks = h['hands/track_id'][:]
        flags = frame_flags(pose, valid, face_valid, hand_frames, actors, source['width'], source['height'])
        counts = {'clips': 1, 'frames': len(pose), 'body_frames': int(valid[:, :11].any(axis=1).sum()),
                  'body_points_available': int(valid[:, :11].sum()), 'body_point_slots': len(pose)*11,
                  'dense_face_frames': int(face_valid.any(axis=1).sum()),
                  'dense_face_points_available': int(face_valid.sum()), 'dense_face_point_slots': len(pose)*478,
                  'hand_observations': len(hand_frames), 'hand_points_available': int(hand_valid.sum()),
                  'hand_point_slots': len(hand_frames)*21, 'object_observations': len(h['objects/frame_index']),
                  'hdf5_bytes': path.stat().st_size}
        hand_counts = {}
        for name, actor in ACTORS.items():
            chosen = actors == actor
            track_counts = Counter(tracks[chosen].tolist())
            hand_counts[name] = {'observations': int(chosen.sum()),
                                'frames_with_observations': len(np.unique(hand_frames[chosen])),
                                'observations_with_no_points': int((~hand_valid[chosen].any(axis=1)).sum()),
                                'points_available': int(hand_valid[chosen].sum()),
                                'tracks_within_clip': len(track_counts),
                                'tracks_with_only_one_observation': sum(n == 1 for n in track_counts.values())}
        objects = Counter(h['objects/class_name'].asstr()[:].tolist())
    return {'source': source, 'output': str(path), 'output_sha256': sha256(path), 'counts': counts,
            'hands_by_actor': hand_counts, 'objects_by_original_class': dict(objects), 'flagged_frames': flags}


def validated_outputs(path, expected):
    """Require one successful final-validation record for every in-scope view."""
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = row['pair_id'], row['view']
        if key not in expected or key in result or row['status'] != 'validated':
            raise ValueError('Validation manifest has a duplicate, failed or out-of-scope clip')
        if len(row.get('output_sha256', '')) != 64:
            raise ValueError('Validation manifest lacks an output digest')
        result[key] = row
    if set(result) != set(expected):
        raise ValueError('Validation manifest does not cover every expected clip')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'output-root', 'report-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--validated-manifest', type=Path,
                        help='Require complete final validation and unchanged output hashes')
    a = parser.parse_args()
    locations = Locations(a.output_root, a.inventory)
    if locations.landmarks_root and a.report_root.resolve().is_relative_to(locations.landmarks_root):
        raise ValueError('Keep reports outside the canonical landmark tree')
    if a.report_root.resolve().is_relative_to(a.output_root.resolve()/'results'):
        raise ValueError('Keep reports outside the HDF5 results tree')
    a.report_root.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(a.inventory.read_text())
    scope = json.loads((a.output_root/'run_scope.json').read_text())
    if scope['manifest_sha256'] != sha256(a.inventory):
        raise ValueError('Requested inventory differs from frozen collection scope')
    expected = {(r['pair_id'], v): r['pid'] for r in inventory for v in r['views']}
    tasks = {r['pair_id']: r.get('subtask_dir', Path(r['pair_id']).parent.as_posix()) for r in inventory}
    inventory_rows = {r['pair_id']: r for r in inventory}
    validated = validated_outputs(a.validated_manifest, expected) if a.validated_manifest else None
    ledger, partial = ledger_rows(a.output_root/'run_ledger.jsonl')
    latest = {(r['pair_id'], r['view']): r for r in ledger}
    counters, hand_counters, object_counters, flags = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    errors, candidates = [], []
    task_counters, participant_counters = defaultdict(Counter), defaultdict(Counter)
    metrics = a.report_root/'clip_quality.jsonl'
    with metrics.open('x') as stream:
        for key, row in latest.items():
            if key not in expected:
                raise ValueError('Ledger clip is outside inventory')
            if row['status'] not in {'written','skipped_valid'}:
                continue
            path = locations.output(inventory_rows[row['pair_id']], row['view'], row['model_key'])
            try:
                result = clip_summary(path)
                source = result['source']
                if (source['pair_id'],source['view']) != key or source['pid'] != expected[key]:
                    raise ValueError('Source identity does not match inventory')
                if validated is not None:
                    prior = validated[key]
                    if (Path(prior['output']).resolve() != path.resolve()
                            or prior['output_sha256'] != result['output_sha256']
                            or prior['frames'] != result['counts']['frames']):
                        raise ValueError('Output differs from its final validation record')
            except Exception as exc:
                errors.append({'pair_id':key[0], 'view':key[1], 'error':str(exc)})
                continue
            stream.write(json.dumps(result)+'\n')
            view = source['view']
            counters[view].update(result['counts'])
            task_counters[(tasks[key[0]], view)].update(result['counts'])
            participant_counters[(source['pid'], view)].update(result['counts'])
            object_counters[view].update(result['objects_by_original_class'])
            for actor, counts in result['hands_by_actor'].items():
                hand_counters[(view,actor)].update(counts)
            for name, indices in result['flagged_frames'].items():
                flags[view][name] += len(indices)
                if indices:
                    candidates.append({'pair_id':key[0], 'view':key[1], 'pid':source['pid'],
                                       'flag':name, 'frames_flagged':len(indices),
                                       'fraction_of_clip':len(indices)/result['counts']['frames'],
                                       'example_frame_indices':indices[:2]+indices[-1:] if len(indices)>2 else indices,
                                       'output':str(path)})
    views = {}
    for view, counts in counters.items():
        views[view] = {'counts':dict(counts),
                       'body_prediction_frame_fraction':counts['body_frames']/counts['frames'],
                       'dense_face_prediction_frame_fraction':counts['dense_face_frames']/counts['frames'],
                       'hands':{actor:dict(hand_counters[(view,actor)]) for actor in ACTORS},
                       'objects':dict(object_counters[view]), 'flag_counts':dict(flags[view])}
    result = {'schema':'collection_coverage_audit_v1', 'as_of_utc':datetime.now(timezone.utc).isoformat(),
              'inventory_sha256':sha256(a.inventory), 'code_sha256':sha256(Path(__file__)),
              'expected_clips':len(expected), 'audited_clips':sum(v['clips'] for v in counters.values()),
              'partial_ledger_line_ignored':partial, 'views':views, 'read_errors':errors,
              'clip_metrics_sha256':sha256(metrics),
              'validated_manifest_sha256':sha256(a.validated_manifest) if a.validated_manifest else None,
              'by_task_and_view':[{ 'task':task, 'view':view, 'counts':dict(counts)}
                                  for (task,view),counts in sorted(task_counters.items())],
              'by_participant_and_view':[{ 'pid':pid, 'view':view, 'counts':dict(counts)}
                                         for (pid,view),counts in sorted(participant_counters.items())],
              'complete_collection_coverage':sum(v['clips'] for v in counters.values())==len(expected) and not errors,
              'limitations':'Prediction availability is not accuracy or physical visibility. Diagnostic flags require visual review; abrupt changes can be real motion. Track counts/one-observation tracks do not measure identity switches. Model classes are unaltered. This audit does not replace final source/schema/configuration validation.'}
    (a.report_root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.report_root/'audit_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if a.validated_manifest and not result['complete_collection_coverage']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
