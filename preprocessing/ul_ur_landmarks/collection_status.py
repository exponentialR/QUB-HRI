"""Report collection progress; optionally verify every expected file against its source."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .schema import sha256, validate_hdf5
from .locations import Locations, contained


def ledger_rows(path):
    """Ignore an unfinished final write while reading a live append-only ledger."""
    if not path.exists():
        return [], False
    data = path.read_bytes()
    partial = bool(data and not data.endswith(b'\n'))
    lines = data.splitlines()
    if partial:
        lines = lines[:-1]
    return [json.loads(line) for line in lines], partial


def progress(inventory, records):
    expected = {(row['pair_id'], view): meta for row in inventory for view, meta in row['views'].items()}
    latest = {}
    for row in records:
        key = row['pair_id'], row['view']
        if key not in expected:
            raise ValueError('Ledger contains a clip outside the expected inventory')
        latest[key] = row
    completed = {key: row for key, row in latest.items() if row['status'] in {'written','skipped_valid'}}
    frames = sum(row['frames'] for row in completed.values())
    total_frames = sum(meta.get('declared_frames') or 0 for meta in expected.values())
    elapsed = sum(row.get('elapsed_s', 0.) for row in records if row['status'] == 'written')
    return {'expected_clips': len(expected), 'logged_valid_clips': len(completed),
            'logged_frames': frames, 'expected_declared_frames': total_frames,
            'remaining_clips': len(expected)-len(completed),
            'latest_failures': [{k: row.get(k) for k in ('pair_id','view','error')}
                                for row in latest.values() if row['status']=='failed'],
            'logged_processing_s': elapsed,
            'estimated_remaining_hours_from_logged_rate':
                max(0,total_frames-frames)*elapsed/frames/3600 if frames else None,
            'all_expected_outputs_logged_valid': len(completed)==len(expected),
            'note': 'Ledger progress is not a fresh filesystem audit or an accuracy measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model-key', default='rtmw_collection_v1_1')
    parser.add_argument('--validate-all', action='store_true')
    a = parser.parse_args()
    inventory = json.loads(a.inventory.read_text())
    locations = Locations(a.output_root, a.inventory)
    scope = json.loads((a.output_root/'run_scope.json').read_text())
    if scope['manifest_sha256'] != sha256(a.inventory):
        raise ValueError('Run and requested inventory differ')
    records, partial = ledger_rows(a.output_root/'run_ledger.jsonl')
    if any(row['model_key'] != a.model_key for row in records):
        raise ValueError('Unexpected model key in ledger')
    result = progress(inventory, records)
    result.update(as_of_utc=datetime.now(timezone.utc).isoformat(), partial_ledger_line_ignored=partial,
                  manifest_sha256=scope['manifest_sha256'], fresh_validation_performed=False)
    if a.validate_all:
        import h5py
        import numpy as np
        configuration = json.loads((a.output_root/(a.model_key+'_configuration.json')).read_text())
        input_root = locations.video_root
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        manifest = a.output_root/('validated_outputs_'+stamp+'.jsonl')
        totals, errors, expected_paths = Counter(), [], set()
        with manifest.open('x') as stream:
            for row in inventory:
                for view, meta in row['views'].items():
                    path = locations.output(row, view, a.model_key)
                    expected_paths.add(path)
                    entry = {'pair_id': row['pair_id'], 'view': view, 'output': str(path)}
                    try:
                        if not path.is_file():
                            raise FileNotFoundError('Expected HDF5 output is missing')
                        source = contained(input_root, meta['relpath'])
                        validated = validate_hdf5(path, source_sha256=sha256(source), model=configuration)
                        with h5py.File(path) as h:
                            identity = json.loads(h.attrs['source_json'])
                            for key, value in {'pair_id':row['pair_id'], 'pid':row['pid'], 'view':view,
                                               'relpath':meta['relpath'], 'width':int(meta['width']),
                                               'height':int(meta['height'])}.items():
                                if identity[key] != value:
                                    raise ValueError('Source identity mismatch: '+key)
                            if meta['declared_frames'] is not None and validated['frames'] != meta['declared_frames']:
                                raise ValueError('Frame count differs from inventory')
                            coverage = {name+'_frames': int(np.any(h['participant/'+name+'/valid'][:],axis=1).sum())
                                        for name in ('pose','face')}
                            coverage['hand_observations'] = len(h['hands/frame_index'])
                            coverage['object_observations'] = len(h['objects/frame_index'])
                        entry.update(status='validated', **validated, output_sha256=sha256(path),
                                     bytes=path.stat().st_size, coverage=coverage)
                        totals.update(clips=1, frames=validated['frames'], bytes=entry['bytes'], **coverage)
                    except Exception as exc:
                        entry.update(status='failed', error=str(exc))
                        errors.append(entry)
                    stream.write(json.dumps(entry)+'\n')
        unexpected = sorted(str(p) for p in locations.outputs_on_disk(a.model_key)-expected_paths)
        result.update(fresh_validation_performed=True, validation_totals=dict(totals),
                      validation_errors=errors, unexpected_outputs=unexpected,
                      validated_output_manifest=str(manifest), validated_output_manifest_sha256=sha256(manifest),
                      collection_files_complete=not errors and not unexpected and totals['clips']==len(expected_paths))
        (a.output_root/('validation_'+stamp+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    (a.output_root/'collection_status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if a.validate_all and not result['collection_files_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
