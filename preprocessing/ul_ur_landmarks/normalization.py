"""Additive normalized-coordinate extension to HDF5 schema 1.1."""

import json
import numpy as np


POINTS = ('participant/pose/xy_px', 'participant/face/xy_px', 'hands/xy_px')
BOXES = ('participant/bbox_xyxy_px', 'hands/bbox_xyxy_px', 'objects/bbox_xyxy_px')
PIXEL_DATASETS = POINTS + BOXES
VERSION = '1.0'


def normalized_name(name):
    return name.removesuffix('_px') + '_norm'


def scale_for(source, name):
    scale = np.asarray([source['width'], source['height']], dtype=np.float32)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Normalization requires positive source dimensions')
    return scale if name in POINTS else np.tile(scale, 2)


def dataset_attrs(name):
    return {'coordinate_units': 'image_fraction', 'pixel_dataset': name,
            'coordinate_order': 'xy' if name in POINTS else 'x1_y1_x2_y2',
            'normalization': 'x/source_width; y/source_height; no clipping'}


def add_normalized(h, *, input_sha256, archive, code_sha256):
    if h.attrs.get('schema_version') != '1.1':
        raise ValueError('Normalized extension requires schema 1.1')
    if 'normalization_version' in h.attrs or any(normalized_name(name) in h for name in PIXEL_DATASETS):
        raise ValueError('Normalized extension already exists')
    source = json.loads(h.attrs['source_json'])
    for name in PIXEL_DATASETS:
        values = h[name][:] / scale_for(source, name)
        out = h.create_dataset(normalized_name(name), data=values.astype(np.float32),
                               compression='gzip', compression_opts=1)
        for key, value in dataset_attrs(name).items():
            out.attrs[key] = value
    h.attrs['normalization_version'] = VERSION
    h.attrs['normalization_json'] = json.dumps({
        'version': VERSION, 'input_sha256': input_sha256, 'archive': str(archive),
        'code_sha256': code_sha256, 'source_width': source['width'], 'source_height': source['height'],
        'formula': 'x/source_width; y/source_height', 'clipped': False,
        'validity': 'reuse the existing masks; NaNs are preserved',
    }, sort_keys=True)


def validate_normalized(h):
    """Optional extension: fail on partial, inconsistent or mislabeled arrays."""
    present = [normalized_name(name) in h for name in PIXEL_DATASETS]
    if not any(present) and 'normalization_version' not in h.attrs and 'normalization_json' not in h.attrs:
        return
    if (h.attrs.get('schema_version') != '1.1' or h.attrs.get('normalization_version') != VERSION
            or not all(present)):
        raise ValueError('Incomplete or unsupported normalized-coordinate extension')
    source = json.loads(h.attrs['source_json'])
    info = json.loads(h.attrs['normalization_json'])
    if (info['version'] != VERSION or info['source_width'] != source['width']
            or info['source_height'] != source['height'] or info['clipped'] is not False
            or info['formula'] != 'x/source_width; y/source_height'
            or not info['archive'] or any(len(info[key]) != 64 for key in ('input_sha256', 'code_sha256'))):
        raise ValueError('Invalid normalization provenance')
    for name in PIXEL_DATASETS:
        out = h[normalized_name(name)]
        if out.shape != h[name].shape or out.dtype != np.dtype('float32'):
            raise ValueError('Normalized dataset shape or dtype differs: ' + name)
        if any(out.attrs.get(key) != value for key, value in dataset_attrs(name).items()):
            raise ValueError('Normalized coordinate metadata differs: ' + name)
        scale = scale_for(source, name)
        for start in range(0, len(out), 256):
            pixel = h[name][start:start+256]
            norm = out[start:start+256]
            if not np.array_equal(norm, pixel / scale, equal_nan=True):
                raise ValueError('Normalized coordinates differ from pixel coordinates: ' + name)
