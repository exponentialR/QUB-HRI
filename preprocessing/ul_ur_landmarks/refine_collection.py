"""Resume cached hand refinement, optionally following a live systemd producer.

Each separate output is checked against its source video, parent HDF5, and frozen
configuration. Every original non-hand dataset and existing hand coordinate is
audited before recording success. No video decoding or model inference occurs.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np

from .collection_status import ledger_rows, progress
from .refine_hands import configuration, refine_file
from .runner import freeze_run_scope, selected_rows
from .schema import sha256, validate_hdf5


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def audit_preservation(parent, output):
    """Existing rows remain the prefix of each frame's refined observations."""
    def equal(a, b):
        return np.array_equal(a, b, equal_nan=True) if a.dtype.kind == 'f' else np.array_equal(a, b)

    with h5py.File(parent) as a, h5py.File(output) as b:
        paths = []
        a.visititems(lambda key, value: paths.append(key) if isinstance(value, h5py.Dataset) else None)
        for key in paths:
            if not key.startswith('hands/') and not equal(a[key][:], b[key][:]):
                raise ValueError('Preservation audit failed: '+key)
        old_frames, new_frames = a['hands/frame_index'][:], b['hands/frame_index'][:]
        old_counts = np.bincount(old_frames, minlength=len(a['frames/index']))
        new_counts = np.bincount(new_frames, minlength=len(a['frames/index']))
        if np.any(new_counts < old_counts):
            raise ValueError('Refinement removed an existing hand observation')
        starts = np.r_[0, np.cumsum(new_counts)[:-1]]
        indices = np.concatenate([np.arange(start, start+count) for start, count in zip(starts, old_counts)])
        for key in a['hands']:
            if key in {'track_id', 'handedness'}:
                continue
            if not equal(a['hands/'+key][:], b['hands/'+key][:][indices]):
                raise ValueError('Existing hand observation changed: '+key)
        source = json.loads(b.attrs['source_json'])
        if source['parent_landmarks'] != {'path': str(parent.resolve()), 'sha256': sha256(parent)}:
            raise ValueError('Refinement parent provenance mismatch')
        return {'preservation_audited': True, 'added_native_observations': int(len(new_frames)-len(old_frames)),
                'ambiguous_side_labels_cleared': int(np.sum(a['hands/handedness'][:] != b['hands/handedness'][:][indices]))}


def producer_state(service):
    """Unknown service-query results never mean the producer has completed."""
    try:
        result = subprocess.run(['systemctl', '--user', 'show', service, '--property=ActiveState', '--value'],
                                capture_output=True, text=True, timeout=5)
        state = result.stdout.strip()
        return state if result.returncode == 0 and state in {'active', 'activating', 'deactivating', 'inactive', 'failed'} else 'unknown'
    except (OSError, subprocess.TimeoutExpired):
        return 'unknown'


class CollectionRefiner:
    def __init__(self, inventory, input_root, output_root, model_key='rtmw_collection_v1_1', threshold=4.):
        self.input_root, self.output_root = input_root.resolve(), output_root.resolve()
        self.scope = json.loads((self.input_root/'run_scope.json').read_text())
        if self.scope['manifest_sha256'] != sha256(inventory):
            raise ValueError('Parent collection and requested inventory differ')
        self.video_root = Path(self.scope['input_root']).resolve()
        repository = Path(__file__).resolve().parents[2]
        if any(self.output_root.is_relative_to(root) or root.is_relative_to(self.output_root)
               for root in (self.input_root, self.video_root, repository)):
            raise ValueError('Use a separate output tree outside the parent collection, repository and source videos')
        self.inventory = selected_rows(inventory, None, self.video_root, max_pairs=None)
        self.expected = {(row['pair_id'], view): (row, meta)
                         for row in self.inventory for view, meta in row['views'].items()}
        self.model_key, self.threshold = model_key, threshold
        if Path(model_key).name != model_key or model_key in {'.', '..'}:
            raise ValueError('Model key must be a directory name')
        self.model = json.loads((self.input_root/(model_key+'_configuration.json')).read_text())
        self.refined_model = configuration(self.model, threshold, True)
        self.identity = {'input_root': str(self.input_root), 'output_root': str(self.output_root),
                         'manifest_sha256': self.scope['manifest_sha256'], 'model_key': model_key,
                         'refined_model': self.refined_model, 'runner_sha256': sha256(Path(__file__))}
        self.attempted = set()

    def prepare(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output_root/'refinement.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            freeze_run_scope(self.output_root/'refinement_configuration.json', self.identity)
            freeze_run_scope(self.output_root/'run_scope.json', self.scope)
            freeze_run_scope(self.output_root/(self.model_key+'_configuration.json'), self.refined_model)
            ledger = self.output_root/'run_ledger.jsonl'
            if ledger.exists():
                data = ledger.read_bytes()
                if data and not data.endswith(b'\n'):
                    split = data.rfind(b'\n')+1
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    (self.output_root/('interrupted_ledger_tail_'+stamp+'.bin')).write_bytes(data[split:])
                    with ledger.open('r+b') as stream:
                        stream.truncate(split)
                records, _ = ledger_rows(ledger)
                progress(self.inventory, records)
                if any(row['model_key'] != self.model_key for row in records):
                    raise ValueError('Unexpected model key in refinement ledger')
        except Exception:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def process_available(self):
        records, partial = ledger_rows(self.input_root/'run_ledger.jsonl')
        progress(self.inventory, records)  # Reject unrelated parent ledger identities.
        latest = {}
        for record in records:
            if record['model_key'] != self.model_key:
                raise ValueError('Unexpected model key in parent ledger')
            latest[record['pair_id'], record['view']] = record
        processed = 0
        for key, record in latest.items():
            if key in self.attempted or record['status'] not in {'written', 'skipped_valid'}:
                continue
            self.attempted.add(key)
            pair_id, view = key
            row, meta = self.expected[key]
            suffix = Path('results')/self.model_key/pair_id/(view+'.h5')
            parent, output = self.input_root/suffix, self.output_root/suffix
            started = time.perf_counter()
            entry = {'pair_id': pair_id, 'view': view, 'model_key': self.model_key,
                     'input': str(parent), 'output': str(output)}
            try:
                if not parent.resolve().is_relative_to(self.input_root) or not output.resolve().is_relative_to(self.output_root):
                    raise ValueError('Result path escapes the requested tree')
                source_hash = sha256(self.video_root/meta['relpath'])
                validation = validate_hdf5(parent, source_sha256=source_hash, model=self.model)
                with h5py.File(parent) as h:
                    identity = json.loads(h.attrs['source_json'])
                    for name, value in {'pair_id': pair_id, 'pid': row['pid'], 'view': view,
                                        'relpath': meta['relpath'], 'width': int(meta['width']), 'height': int(meta['height'])}.items():
                        if identity[name] != value:
                            raise ValueError('Parent source identity mismatch: '+name)
                if meta.get('declared_frames') is not None and validation['frames'] != meta['declared_frames']:
                    raise ValueError('Parent frame count differs from inventory')
                entry.update(refine_file(parent, output, self.threshold))
                entry.update(audit_preservation(parent, output), output_sha256=sha256(output))
            except Exception as exc:
                entry.update(status='failed', error=str(exc))
            entry.update(elapsed_s=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
            with (self.output_root/'run_ledger.jsonl').open('a') as stream:
                stream.write(json.dumps(entry)+'\n')
                stream.flush()
            processed += 1
        return processed, partial

    def status(self, state, producer=None):
        records, partial = ledger_rows(self.output_root/'run_ledger.jsonl')
        result = progress(self.inventory, records)
        result.update(run_state=state, producer_state=producer, as_of_utc=datetime.now(timezone.utc).isoformat(),
                      partial_ledger_line_ignored=partial, manifest_sha256=self.scope['manifest_sha256'],
                      fresh_validation_performed=False, goal_complete=False,
                      note='Refinement includes per-file source, schema, parent and preservation checks. Runtime estimates count active CPU work only; waiting for inference is excluded.')
        atomic_json(self.output_root/'collection_status.json', result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--input-root', type=Path, required=True, help='Parent collection outputs')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--model-key', default='rtmw_collection_v1_1')
    p.add_argument('--native-response-threshold', type=float, default=4.)
    p.add_argument('--follow-service', help='User systemd inference unit; wait for new completed parents')
    p.add_argument('--poll-seconds', type=float, default=30.)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not np.isfinite(a.native_response_threshold) or a.native_response_threshold <= 0 or not 1 <= a.poll_seconds <= 60:
        p.error('Use a positive finite threshold and a poll interval from 1 to 60 seconds')
    refiner = CollectionRefiner(a.inventory, a.input_root, a.output_root, a.model_key, a.native_response_threshold)
    if a.dry_run:
        records, _ = ledger_rows(a.input_root/'run_ledger.jsonl')
        print(json.dumps({'dry_run': True, 'parent_progress': progress(refiner.inventory, records),
                          'output_root': str(refiner.output_root), 'no_inference': True}, indent=2))
        return
    refiner.prepare()
    try:
        while True:
            refiner.process_available()
            state = producer_state(a.follow_service) if a.follow_service else None
            report = refiner.status('following_producer' if a.follow_service else 'processed_available', state)
            if report['all_expected_outputs_logged_valid']:
                break
            if not a.follow_service or state in {'inactive', 'failed'}:
                # The last parent ledger append may have raced with the first scan.
                if refiner.process_available()[0]:
                    continue
                refiner.status('incomplete', state)
                raise SystemExit(2)
            time.sleep(a.poll_seconds)
        refiner.status('validating_all', state)
        command = [sys.executable, '-m', 'preprocessing.ul_ur_landmarks.collection_status',
                   '--inventory', str(a.inventory), '--output-root', str(a.output_root),
                   '--model-key', a.model_key, '--validate-all']
        with (a.output_root/'final_validation.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        final = json.loads((a.output_root/'collection_status.json').read_text())
        final.update(run_state='complete' if result.returncode == 0 else 'validation_failed', goal_complete=False)
        atomic_json(a.output_root/'collection_status.json', final)
        raise SystemExit(result.returncode)
    finally:
        refiner.close()


if __name__ == '__main__':
    main()
