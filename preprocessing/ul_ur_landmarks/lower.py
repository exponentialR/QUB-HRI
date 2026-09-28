"""Legacy LL/LR MediaPipe pixels and their additive normalized extension.

The historical sideview_keyextraction.py writer stores integer-valued pixels
as float32, with entirely zero-filled groups for missing observations. It
does not save confidence, actor identity, tracks, or inference timestamps.
"""

import json

import h5py
import numpy as np


TOPOLOGIES = {'face_landmarks': ('mediapipe_478', 478),
              'pose_landmarks': ('mediapipe_pose_33', 33),
              'left_hand_landmarks': ('mediapipe_hand_21', 21),
              'right_hand_landmarks': ('mediapipe_hand_21', 21)}
VERSION = 'legacy_lower_normalized_1.0'


def validity(xy):
    """A zero point is valid within an otherwise nonzero observation."""
    return np.isfinite(xy).all(axis=-1) & np.any(xy != 0, axis=(-2, -1))[..., None]


def read_pixels(h, count):
    arrays = {}
    for name, (_, points) in TOPOLOGIES.items():
        a = h[name][:]
        if a.shape != (count, points, 2) or a.dtype != np.dtype('float32'):
            raise ValueError('Unexpected lower-view landmark shape/dtype: ' + name)
        # This particular legacy producer truncates normalized positions to
        # integer pixels. Reject fractional/unknown layouts rather than guess units.
        finite = a[np.isfinite(a)]
        if not np.array_equal(finite, np.trunc(finite)):
            raise ValueError('Expected integer-valued legacy pixel coordinates: ' + name)
        arrays[name] = a
    return arrays


def normalized(xy, width, height):
    mask = validity(xy)
    result = xy / np.array([width, height], dtype=np.float32)
    result[~mask] = np.nan
    return result, mask


def add_normalized(h, source, *, input_sha256, original_path, code_sha256):
    if source['width'] <= 0 or source['height'] <= 0:
        raise ValueError('Invalid video dimensions')
    arrays = read_pixels(h, source['frames'])
    if set(h) != set(TOPOLOGIES) or 'lower_normalization_json' in h.attrs:
        raise ValueError('Expected the unmodified four-dataset lower-view layout')
    meta = {'version': VERSION, 'source': source, 'input_sha256': input_sha256,
            'original_path': str(original_path), 'code_sha256': code_sha256,
            'formula': 'x / width, y / height; no clipping',
            'missing': 'Whole zero-filled observation absent/unknown; finite points in other observations valid; invalid normalized points are NaN',
            'timing': 'Legacy row index; no saved inference timestamps. Viewer uses matching video relative PTS.',
            'actor_identity': 'unknown; handedness is the historical detector label, not an actor ID',
            'precision': 'Original integer pixel quantization retained; original MediaPipe subpixel precision cannot be recovered',
            'historical_writer': 'reconstruction/sideview_keyextraction.py'}
    for name, xy in arrays.items():
        norm, mask = normalized(xy, source['width'], source['height'])
        ds = h.create_dataset(name+'_norm', data=norm, compression='gzip', compression_opts=1, shuffle=True)
        ds.attrs['topology'] = TOPOLOGIES[name][0]
        ds.attrs['coordinates'] = 'normalized_image_xy'
        h.create_dataset(name+'_valid', data=mask, compression='gzip', compression_opts=1)
    h.attrs['lower_normalization_json'] = json.dumps(meta, sort_keys=True)
    return meta


def validate_extension(h, arrays, width, height, source=None):
    meta = json.loads(h.attrs['lower_normalization_json'])
    if meta['version'] != VERSION:
        raise ValueError('Unknown lower-view normalization version')
    stored = meta['source']
    if (stored['width'], stored['height'], stored['frames']) != (width, height, len(next(iter(arrays.values())))):
        raise ValueError('Lower-view normalization dimensions/frame count differ from video')
    if source is not None and stored != source:
        raise ValueError('Lower-view source identity mismatch')
    expected_names = set(TOPOLOGIES) | {n+s for n in TOPOLOGIES for s in ('_norm', '_valid')}
    if set(h) != expected_names:
        raise ValueError('Unexpected normalized lower-view hierarchy')
    for name, xy in arrays.items():
        norm, mask = normalized(xy, width, height)
        ds, valid = h[name+'_norm'], h[name+'_valid']
        if (ds.dtype != np.dtype('float32') or valid.dtype != np.dtype('bool')
                or ds.attrs.get('topology') != TOPOLOGIES[name][0]
                or not np.array_equal(ds[:], norm, equal_nan=True)
                or not np.array_equal(valid[:], mask)):
            raise ValueError('Normalized lower-view coordinates or validity differ: ' + name)
    return meta


class LowerData:
    def __init__(self, path, times_ms, width, height, *, video_sha256=None):
        self.width, self.height = width, height
        self.times = np.asarray(times_ms, dtype=float)
        if not len(self.times) or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('Lower-view video timing is invalid')
        with h5py.File(path, 'r') as h:
            self.arrays = read_pixels(h, len(self.times))
            if 'lower_normalization_json' in h.attrs:
                meta = validate_extension(h, self.arrays, width, height)
                if video_sha256 is not None and meta['source']['sha256'] != video_sha256:
                    raise ValueError('Lower-view source hash mismatch')
            elif set(h) != set(TOPOLOGIES):
                raise ValueError('Unsupported lower-view landmark format')

    def frame(self, index):
        if not 0 <= index < len(self.times):
            raise IndexError('Frame outside lower-view clip')

        def group(name):
            xy = self.arrays[name][index]
            mask = validity(xy)
            return {'xy': [p.tolist() if ok else None for p, ok in zip(xy, mask)],
                    'valid': mask.tolist(), 'confidence': [None]*len(xy)}

        result = {'kind': 'legacy_lower', 'pose_topology': 'mediapipe_pose_33',
                  'frame_index': index, 'timestamp_ms': float(self.times[index]),
                  'width': self.width, 'height': self.height,
                  'pose': group('pose_landmarks'), 'face': group('face_landmarks'),
                  'hands': [], 'objects': []}
        for side in ('left', 'right'):
            hand = group(side+'_hand_landmarks')
            if any(hand['valid']):
                result['hands'].append({**hand, 'actor': 'unknown', 'handedness': side,
                    'track_id': None, 'topology': 'mediapipe_hand_21', 'model_id': 'legacy_lower',
                    'bbox': None, 'detection_confidence': None})
        return result
