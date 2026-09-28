"""Current data locations, separate from immutable inference provenance."""

import json
from pathlib import Path

from .schema import sha256


def contained(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Expected a relative path within the data root')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Path escapes the data root')
    return result


def filename_path(root, row, view):
    """Preserve the exact source filename, including case and decimal timestamps."""
    relative = Path(row['views'][view]['relpath'])
    if view not in ('CAM_UL', 'CAM_UR') or relative.stem.split('-').count(view) != 1:
        raise ValueError('Source filename does not identify the requested UL/UR view')
    return contained(root, relative.with_suffix('.h5'))


class Locations:
    def __init__(self, collection_root, inventory):
        self.collection_root = Path(collection_root).resolve()
        scope = json.loads((self.collection_root / 'run_scope.json').read_text())
        digest = sha256(inventory)
        if scope['manifest_sha256'] != digest:
            raise ValueError('Inventory does not match the collection')
        self.video_root = Path(scope['input_root']).resolve()
        self.landmarks_root = None
        self.model_key = None
        path = self.collection_root / 'data_locations.json'
        if path.exists():
            data = json.loads(path.read_text())
            if data.get('layout') != 'video_filename_v1' or data.get('manifest_sha256') != digest:
                raise ValueError('Unrecognized layout or mismatched location inventory')
            self.video_root = Path(data['video_root']).resolve()
            self.landmarks_root = Path(data['landmarks_root']).resolve()
            self.model_key = data['model_key']

    def output_root(self, model_key):
        if self.landmarks_root is not None:
            if model_key != self.model_key:
                raise ValueError('Model key differs from relocated collection')
            return self.landmarks_root
        return self.collection_root / 'results' / model_key

    def output(self, row, view, model_key):
        root = self.output_root(model_key)
        if self.landmarks_root is not None:
            return filename_path(root, row, view)
        return contained(root, Path(row['pair_id']) / (view + '.h5'))

    def outputs_on_disk(self, model_key):
        paths = self.output_root(model_key).rglob('*.h5')
        if self.landmarks_root is None:
            return set(paths)
        # A shared landmark tree also contains historical CAM_AV files.
        return {p for p in paths if {'CAM_UL', 'CAM_UR'} & set(p.stem.upper().split('-'))}
