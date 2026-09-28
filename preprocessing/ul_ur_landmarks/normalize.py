"""Add normalized coordinates to delivered UL/UR files, preserving archived originals.

Dry run by default. --apply publishes independently copied, fully checked HDF5
files by atomic replacement. The pinned archived input must remain available.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import h5py
import numpy as np

from .collection_quality import validated_outputs
from .locations import Locations, contained
from .normalization import PIXEL_DATASETS, add_normalized, normalized_name
from .schema import sha256, validate_hdf5


def atomic_json(path, data):
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(data, indent=2)+'\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a, b)


def verify_preserved(archive, candidate):
    """Compare every original dataset and attribute, including masks and strings."""
    with h5py.File(archive, 'r') as old, h5py.File(candidate, 'r') as new:
        old_names, new_names = [], []
        old.visit(old_names.append); new.visit(new_names.append)
        added = {normalized_name(name) for name in PIXEL_DATASETS}
        if set(new_names) != set(old_names) | added:
            raise ValueError('Unexpected change to the original HDF5 hierarchy')
        for name in [''] + old_names:
            before, after = (old, new) if name == '' else (old[name], new[name])
            expected_attrs = set(before.attrs) | ({'normalization_version', 'normalization_json'} if name == '' else set())
            if set(after.attrs) != expected_attrs or any(not equal(value, after.attrs[key]) for key, value in before.attrs.items()):
                raise ValueError('Original attributes changed: ' + name)
            if isinstance(before, h5py.Dataset):
                if before.shape != after.shape or before.dtype != after.dtype:
                    raise ValueError('Original dataset shape/dtype changed: ' + name)
                for start in range(0, len(before), 256):
                    if not equal(before[start:start+256], after[start:start+256]):
                        raise ValueError('Original dataset changed: ' + name)


def normalize_clip(job):
    row, view, record, path, archive, code_digest = job
    start = time.monotonic()
    digest = record['output_sha256']
    if path.is_symlink() or archive.is_symlink() or path.resolve() == archive.resolve():
        raise ValueError('Canonical and archived paths must be distinct regular files')
    if sha256(archive) != digest:
        raise ValueError('Archived input differs from the pinned validation manifest: ' + str(archive))
    with h5py.File(archive, 'r') as old:
        source = json.loads(old.attrs['source_json']); model = json.loads(old.attrs['model_json'])
        if 'normalization_version' in old.attrs:
            raise ValueError('Archive must be the original pixel-coordinate file')
    for key, expected in {'pair_id': row['pair_id'], 'pid': row['pid'], 'view': view,
                          'relpath': row['views'][view]['relpath'],
                          'width': row['views'][view]['width'], 'height': row['views'][view]['height']}.items():
        if source[key] != expected:
            raise ValueError('Archived source identity differs: '+key)
    with h5py.File(path, 'r') as current:
        normalized = 'normalization_version' in current.attrs
        if normalized:
            prior = json.loads(current.attrs['normalization_json'])
            if prior['input_sha256'] != digest or prior['archive'] != str(archive):
                raise ValueError('Existing normalization derives from a different input')
    temporary = None
    try:
        if normalized:
            candidate = path
        else:
            if sha256(path) != digest:
                raise ValueError('Current output differs from the validated pixel input: '+str(path))
            fd, name = tempfile.mkstemp(prefix='.normalize_', suffix='.partial', dir=path.parent)
            os.close(fd); temporary = Path(name)
            shutil.copyfile(path, temporary)
            with h5py.File(temporary, 'r+') as h:
                add_normalized(h, input_sha256=digest, archive=archive, code_sha256=code_digest)
            candidate = temporary
        validated = validate_hdf5(candidate, source_sha256=source['sha256'], model=model)
        verify_preserved(archive, candidate)
        output_digest = sha256(candidate)
        if temporary is not None:
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
            if sha256(path) != digest:
                raise ValueError('Output changed while normalization was being prepared')
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:os.fsync(directory_fd)
            finally:os.close(directory_fd)
        return {**record, **validated, 'output': str(path), 'output_sha256': output_digest,
                'input_sha256': digest, 'original_output': str(archive), 'bytes': path.stat().st_size,
                'status': 'validated', 'normalization_version': '1.0',
                'normalization_action': 'already_valid' if normalized else 'written',
                'elapsed_s': time.monotonic()-start, 'all_original_datasets_and_attributes_preserved': True}
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def run_parallel(jobs, workers):
    """Bound queued writes so a failure cancels work that has not started."""
    iterator = iter(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(normalize_clip, job) for _, job in zip(range(workers*2), iterator)}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                    job = next(iterator, None)
                    if job is not None:pending.add(pool.submit(normalize_clip, job))
        except BaseException:
            for future in pending:future.cancel()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('collection-root', 'inventory', 'validated-manifest', 'audit-root'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--validated-manifest-sha256', required=True)
    p.add_argument('--model-key', default='rtmw_collection_v1_1')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, help='Bound a real-data check; omit for the full collection')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    if a.workers < 1 or (a.limit is not None and a.limit < 1):
        raise ValueError('Workers and limit must be positive')
    locations = Locations(a.collection_root, a.inventory)
    if locations.landmarks_root is None:
        raise ValueError('Requires the canonical video-filename layout')
    audit = a.audit_root.resolve()
    for root in (locations.landmarks_root, locations.video_root, Path(__file__).resolve().parents[2]):
        if audit.is_relative_to(root) or root.is_relative_to(audit):
            raise ValueError('Keep audit output separate from landmarks, videos and repository')
    if sha256(a.validated_manifest) != a.validated_manifest_sha256:
        raise ValueError('Validation manifest digest differs')
    rows = json.loads(a.inventory.read_text())
    expected = {(r['pair_id'], view) for r in rows for view in r['views']}
    if len(expected) != len(rows)*2 or any(set(r['views']) != {'CAM_UL', 'CAM_UR'} for r in rows):
        raise ValueError('Inventory must contain unique complete UL/UR pairs')
    prior = validated_outputs(a.validated_manifest, expected)
    code_digest = sha256(Path(__file__))
    jobs, targets = [], set()
    archive_root = a.collection_root.resolve()/'results'/a.model_key
    for row in rows:
        for view in row['views']:
            record = prior[row['pair_id'], view]
            target = locations.output(row, view, a.model_key)
            archive = contained(archive_root, Path(row['pair_id'])/(view+'.h5'))
            if (Path(record['output']).resolve() != target or Path(record['original_output']).resolve() != archive
                    or not archive.is_file() or not target.is_file() or target in targets):
                raise ValueError('Invalid, duplicate or missing input path')
            targets.add(target); jobs.append((row, view, record, target, archive, code_digest))
    if a.limit is not None:jobs = jobs[:a.limit]
    print(json.dumps({'planned_clips': len(jobs), 'total_clips': len(expected), 'apply': a.apply}), flush=True)
    if not a.apply:return
    audit.mkdir(parents=True, exist_ok=True)
    with (audit/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {'inventory_sha256': sha256(a.inventory), 'input_manifest_sha256': a.validated_manifest_sha256,
                  'landmarks_root': str(locations.landmarks_root), 'archive_root': str(archive_root),
                  'code_sha256': code_digest, 'extension_code_sha256': sha256(Path(__file__).with_name('normalization.py')),
                  'schema_code_sha256': sha256(Path(__file__).with_name('schema.py'))}
        config_path = audit/'configuration.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Existing normalization run configuration differs')
        atomic_json(config_path, config)
        av = {str(path.relative_to(locations.landmarks_root)): sha256(path)
              for path in locations.landmarks_root.rglob('*.h5') if '-CAM_AV-' in path.name}
        av_path = audit/'av_before.json'
        if av_path.exists() and json.loads(av_path.read_text()) != av:
            raise ValueError('Aerial files changed since normalization began')
        atomic_json(av_path, av)
        started = time.monotonic(); stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        manifest = audit/('validated_outputs_'+stamp+'.jsonl')
        frames = written = skipped = output_bytes = 0
        status = {'state': 'running', 'planned_clips': len(jobs), 'total_clips': len(expected), 'verified_clips': 0}
        atomic_json(audit/'status.json', status)
        try:
            with manifest.open('x') as stream:
                for i, result in enumerate(run_parallel(jobs, a.workers), 1):
                    stream.write(json.dumps(result)+'\n')
                    frames += result['frames']; output_bytes += result['bytes']
                    written += result['normalization_action'] == 'written'
                    skipped += result['normalization_action'] == 'already_valid'
                    if i % 100 == 0 or i == len(jobs):
                        stream.flush()
                        status.update(verified_clips=i, frames=frames, written=written, already_valid=skipped,
                                      elapsed_s=time.monotonic()-started)
                        atomic_json(audit/'status.json', status)
                        print(json.dumps(status), flush=True)
            for relative, digest in av.items():
                if sha256(locations.landmarks_root/relative) != digest:
                    raise ValueError('An aerial file changed during normalization')
            status.update(state='complete' if len(jobs) == len(expected) else 'bounded_check_complete',
                          collection_complete=len(jobs) == len(expected), bytes=output_bytes,
                          av_files_preserved=len(av), elapsed_s=time.monotonic()-started,
                          validated_output_manifest=str(manifest), validated_output_manifest_sha256=sha256(manifest))
            atomic_json(audit/'status.json', status)
            print(json.dumps(status), flush=True)
        except Exception as exc:
            status.update(state='failed', error=str(exc)); atomic_json(audit/'status.json', status)
            raise


if __name__ == '__main__':main()
