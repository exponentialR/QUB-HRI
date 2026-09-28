"""Copy legacy LL/LR pixels into the dataset layout with normalized arrays.

Dry run by default; --apply adds independent copies, never overwrites files.
Every applied clip is checked against its video header and first decoded frame.
Full video PTS/frame count is checked by the viewer when a clip is opened.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import h5py
import numpy as np

from .locations import contained
from .lower import TOPOLOGIES, add_normalized, read_pixels, validate_extension, validity
from .schema import sha256
from .viewer_dataset import NAME


def build_jobs(input_root, video_root, output_root):
    jobs, seen = [], set()
    for view in ('CAM_LL', 'CAM_LR'):
        for path in sorted((input_root/view).rglob('*.h5')):
            relative = path.relative_to(input_root/view)
            match = NAME.fullmatch(path.stem)
            if match is None or match[2].upper() != view:
                raise ValueError('Unexpected lower-view filename: ' + str(path))
            key = (str(relative.parent).lower(), path.stem.lower())
            if key in seen:
                raise ValueError('Duplicate lower-view identity: ' + str(path))
            seen.add(key)
            original = contained(input_root, path.relative_to(input_root))
            video = contained(video_root, relative.with_suffix('.mp4'))
            target = contained(output_root, relative)
            if not video.is_file():
                raise ValueError('Missing matching video: ' + str(video))
            if original == target or target.is_symlink():
                raise ValueError('Output must be a separate regular file')
            jobs.append({'input': str(original), 'video': str(video), 'output': str(target),
                         'relpath': relative.with_suffix('.mp4').as_posix(), 'view': view})
    if not jobs:
        raise ValueError('No lower-view HDF5 files found under CAM_LL/ and CAM_LR/')
    return jobs


def convert(job):
    cv2.setNumThreads(1)
    original, video, target = (Path(job[k]) for k in ('input', 'video', 'output'))
    digest = sha256(original)
    capture = cv2.VideoCapture(str(video))
    try:
        width, height, count = (int(capture.get(prop)) for prop in
            (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        if not ok or count <= 0 or frame.shape[:2] != (height, width):
            raise ValueError('Unreadable video or inconsistent dimensions: ' + str(video))
    finally:
        capture.release()
    source = {'relpath': job['relpath'], 'view': job['view'], 'width': width, 'height': height,
              'frames': count, 'sha256': sha256(video)}
    with h5py.File(original, 'r') as h:
        if set(h) != set(TOPOLOGIES):
            raise ValueError('Expected original four-dataset layout: ' + str(original))
        arrays = read_pixels(h, count)
        attrs = dict(h.attrs)
        dataset_attrs = {name: dict(h[name].attrs) for name in TOPOLOGIES}
    temporary = None
    exists = target.exists()
    try:
        if exists:
            candidate = target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='.lower_', suffix='.partial', dir=target.parent)
            os.close(fd); temporary = Path(name)
            shutil.copyfile(original, temporary)
            with h5py.File(temporary, 'r+') as h:
                add_normalized(h, source, input_sha256=digest, original_path=original,
                               code_sha256=job['code_sha256'])
            candidate = temporary
        with h5py.File(candidate, 'r') as h:
            if 'lower_normalization_json' not in h.attrs:
                raise ValueError('Existing output is not a normalized lower-view copy')
            copied = read_pixels(h, count)
            meta = validate_extension(h, copied, width, height, source)
            if meta['input_sha256'] != digest:
                raise ValueError('Existing output derives from another input')
            if set(h.attrs) != set(attrs) | {'lower_normalization_json'}:
                raise ValueError('Original root attributes changed')
            for key, value in attrs.items():
                if not np.array_equal(h.attrs[key], value):
                    raise ValueError('Original root attribute changed: '+key)
            for key, value in arrays.items():
                if not np.array_equal(copied[key], value, equal_nan=True):
                    raise ValueError('Original pixels changed: '+key)
                if set(h[key].attrs) != set(dataset_attrs[key]) or any(
                        not np.array_equal(h[key].attrs[a], v) for a, v in dataset_attrs[key].items()):
                    raise ValueError('Original dataset attributes changed: '+key)
        output_digest = sha256(candidate)
        if sha256(original) != digest:
            raise ValueError('Original input changed during conversion')
        if temporary is not None:
            with temporary.open('rb') as stream: os.fsync(stream.fileno())
            os.link(temporary, target)  # Exclusive atomic publication; no replacement.
        return {**job, 'source': source, 'input_sha256': digest, 'output_sha256': output_digest,
                'action': 'already_valid' if exists else 'written', 'bytes': target.stat().st_size,
                'originals_preserved': True,
                'present_frames': {n: int(validity(a).any(axis=1).sum()) for n, a in arrays.items()},
                'nonfinite_coordinates': sum(int((~np.isfinite(a)).sum()) for a in arrays.values())}
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(temporary, path)


def results(jobs, workers):
    iterator = iter(jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(convert, j) for _, j in zip(range(workers*2), iterator)}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                    job = next(iterator, None)
                    if job is not None: pending.add(pool.submit(convert, job))
        except BaseException:
            for future in pending: future.cancel()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input-root', 'video-root', 'output-root', 'audit-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    if a.workers < 1 or (a.limit is not None and a.limit < 1):
        p.error('Workers and limit must be positive')
    roots = [a.input_root.resolve(), a.video_root.resolve(), a.output_root.resolve(), a.audit_root.resolve()]
    repo = Path(__file__).resolve().parents[2]
    for i, root in enumerate(roots):
        if root.is_relative_to(repo) or repo.is_relative_to(root):
            p.error('Keep inputs and outputs outside the repository')
        if any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots[i+1:]):
            p.error('Input, video, output and audit trees must be separate')
    if not roots[0].is_dir() or not roots[1].is_dir():
        p.error('Input and video directories must exist')
    jobs = build_jobs(*roots[:3])
    total = len(jobs)
    if a.limit is not None: jobs = jobs[:a.limit]
    print(json.dumps({'total_files': total, 'selected_files': len(jobs), 'apply': a.apply}), flush=True)
    if not a.apply: return
    audit = roots[3]; audit.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    sources = [Path(__file__), Path(__file__).with_name('lower.py')]
    code_digest = hashlib.sha256(b''.join(f.read_bytes() for f in sources)).hexdigest()
    for f in sources: shutil.copyfile(f, audit/(stamp+'_'+f.name))
    for job in jobs: job['code_sha256'] = code_digest
    manifest = audit/(stamp+'_manifest.jsonl')
    start = time.monotonic()
    status = {'state': 'running', 'expected_files': len(jobs), 'completed_files': 0,
              'views': {}, 'frames': 0, 'bytes': 0, 'dimensions': {}, 'nonfinite_coordinates': 0,
              'manifest': str(manifest), 'code_sha256': code_digest}
    try:
        with manifest.open('x') as stream:
            for record in results(jobs, a.workers):
                stream.write(json.dumps(record)+'\n'); stream.flush()
                status['completed_files'] += 1
                view = record['view']; status['views'][view] = status['views'].get(view, 0)+1
                status['frames'] += record['source']['frames']; status['bytes'] += record['bytes']
                dims = f"{record['source']['width']}x{record['source']['height']}"
                status['dimensions'][dims] = status['dimensions'].get(dims, 0)+1
                status['nonfinite_coordinates'] += record['nonfinite_coordinates']
                if status['completed_files'] % 100 == 0:
                    status['elapsed_s'] = time.monotonic()-start
                    atomic_json(audit/'status.json', status)
        status['state'] = 'complete'
        status['manifest_sha256'] = sha256(manifest)
    except BaseException as exc:
        status.update(state='failed', error=str(exc)); raise
    finally:
        status['elapsed_s'] = time.monotonic()-start
        atomic_json(audit/'status.json', status)
        atomic_json(audit/(stamp+'_summary.json'), status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
