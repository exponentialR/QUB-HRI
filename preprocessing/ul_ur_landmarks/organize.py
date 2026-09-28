"""Copy validated UL/UR outputs to landmarks/<task>/<video-stem>.h5.

Dry run by default. Preserve original HDF5 bytes and frozen provenance. Existing
destinations must be byte-identical; independent copies never overwrite files.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py

from .collection_quality import validated_outputs
from .locations import contained, filename_path
from .schema import sha256


def copy_verified(source, destination, digest):
    if sha256(source) != digest:
        raise ValueError('Original output changed: ' + str(source))
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError('Destination differs; refusing overwrite: ' + str(destination))
        return 'already_identical'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.ul_ur_copy_', dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output, source.open('rb') as input_stream:
            shutil.copyfileobj(input_stream, output)
            output.flush()
            os.fsync(output.fileno())
        if sha256(temporary) != digest:
            raise ValueError('Copied output digest differs')
        # Link the independent temporary copy, never the archived original.
        # link() fails atomically if another process created the destination.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return 'copied'


def build_plan(collection, inventory, validated_manifest, validated_digest,
               video_root, landmarks_root, model_key):
    scope = json.loads((collection / 'run_scope.json').read_text())
    if scope['manifest_sha256'] != sha256(inventory):
        raise ValueError('Inventory differs from frozen collection')
    if sha256(validated_manifest) != validated_digest:
        raise ValueError('Validation manifest digest differs')
    rows = json.loads(inventory.read_text())
    expected = {(r['pair_id'], v) for r in rows for v in r['views']}
    if len(expected) != len(rows) * 2 or any(set(r['views']) != {'CAM_UL', 'CAM_UR'} for r in rows):
        raise ValueError('Inventory must contain unique complete UL/UR pairs')
    prior = validated_outputs(validated_manifest, expected)
    plan, seen = [], set()
    archive = collection / 'results' / model_key
    for row in rows:
        for view, meta in row['views'].items():
            record = prior[row['pair_id'], view]
            source = contained(archive, Path(row['pair_id']) / (view + '.h5'))
            if source != Path(record['output']).resolve() or not source.is_file():
                raise ValueError('Validation record does not identify the original output')
            video = contained(video_root, meta['relpath'])
            if not video.is_file():
                raise FileNotFoundError(video)
            target = filename_path(landmarks_root, row, view)
            if target in seen or target == source:
                raise ValueError('Duplicate or original destination')
            seen.add(target)
            if target.exists() and sha256(target) != record['output_sha256']:
                raise ValueError('Destination differs; refusing overwrite: ' + str(target))
            plan.append((row, view, record, source, video, target))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('collection-root', 'inventory', 'validated-manifest', 'video-root', 'landmarks-root', 'audit-root'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--validated-manifest-sha256', required=True)
    p.add_argument('--model-key', default='rtmw_collection_v1_1')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    collection, videos, landmarks, audit = (x.resolve() for x in
        (a.collection_root, a.video_root, a.landmarks_root, a.audit_root))
    repo = Path(__file__).resolve().parents[2]
    for root in (videos, collection, repo):
        if landmarks.is_relative_to(root) or root.is_relative_to(landmarks):
            raise ValueError('Landmarks must be separate from videos, archive and repository')
    for root in (videos, landmarks, repo):
        if audit.is_relative_to(root) or root.is_relative_to(audit):
            raise ValueError('Audit must be separate from videos, landmarks and repository')
    descriptor = {'layout': 'video_filename_v1', 'manifest_sha256': sha256(a.inventory),
                  'video_root': str(videos), 'landmarks_root': str(landmarks), 'model_key': a.model_key}
    location_file = collection / 'data_locations.json'
    if location_file.exists() and json.loads(location_file.read_text()) != descriptor:
        raise ValueError('Existing location descriptor differs')
    plan = build_plan(collection, a.inventory, a.validated_manifest, a.validated_manifest_sha256,
                      videos, landmarks, a.model_key)
    print(json.dumps({'clips': len(plan), 'destination': str(landmarks), 'apply': a.apply}), flush=True)
    if not a.apply:
        return
    audit.mkdir(parents=True, exist_ok=False)
    # Includes all existing files, so both AV originals and resumed copies are protected.
    before = {str(path.relative_to(landmarks)): sha256(path)
              for path in sorted(landmarks.rglob('*')) if path.is_file()}
    (audit / 'existing_files_before.json').write_text(json.dumps(before, indent=2) + '\n')
    original_bytes = sum(source.stat().st_size for _, _, _, source, _, target in plan if not target.exists())
    if shutil.disk_usage(landmarks.parent).free < original_bytes + 100_000_000:
        raise ValueError('Insufficient disk space for independent copies')
    counts = {'copied': 0, 'already_identical': 0}
    manifest = audit / 'validated_outputs.jsonl'
    with manifest.open('x') as output:
        for i, (row, view, record, source, video, target) in enumerate(plan):
            with h5py.File(source, 'r') as h:
                identity = json.loads(h.attrs['source_json'])
            meta = row['views'][view]
            for key, value in {'pair_id': row['pair_id'], 'pid': row['pid'], 'view': view,
                               'relpath': meta['relpath'], 'width': meta['width'], 'height': meta['height']}.items():
                if identity[key] != value:
                    raise ValueError('Source identity mismatch: ' + key)
            if sha256(video) != identity['sha256']:
                raise ValueError('Renamed video content differs: ' + str(video))
            action = copy_verified(source, target, record['output_sha256'])
            counts[action] += 1
            output.write(json.dumps({**record, 'output': str(target), 'original_output': str(source),
                                     'source_video': str(video), 'source_sha256': identity['sha256'],
                                     'relocation_action': action}) + '\n')
            if (i + 1) % 1000 == 0:
                output.flush()
                print(f'{i + 1}/{len(plan)} verified', flush=True)
    changed = [relative for relative, digest in before.items()
               if not (landmarks / relative).is_file() or sha256(landmarks / relative) != digest]
    if changed:
        raise ValueError('Existing files changed during organization: ' + repr(changed))
    summary = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'clips': len(plan), **counts,
               'existing_files_preserved': len(before), 'existing_cam_av_preserved':
                   sum('CAM_AV' in Path(name).stem.split('-') for name in before),
               'manifest_sha256': sha256(a.inventory), 'original_validated_manifest': str(a.validated_manifest),
               'original_validated_manifest_sha256': a.validated_manifest_sha256,
               'validated_output_manifest': str(manifest), 'validated_output_manifest_sha256': sha256(manifest),
               'locations': descriptor, 'code_sha256': sha256(Path(__file__)),
               'note': 'Original validated HDF5 bytes copied; all renamed source hashes checked; no inference or schema conversion.'}
    (audit / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not location_file.exists():
        with location_file.open('x') as stream:
            stream.write(json.dumps(descriptor, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
