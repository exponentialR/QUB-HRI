"""Read the historical aerial HDF5 layout without changing its stored arrays.

AV hands/gaze are normalized x/y; boxes use normalized center-x, center-y,
width, height (the legacy YOLO writer's xywhn convention). No confidence,
object class, validity mask or persistent track ID is supplied by this layout.
"""

import json
import subprocess

import h5py
import numpy as np


def video_timing(path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
               '-show_frames', '-show_entries',
               'stream=width,height:frame=best_effort_timestamp_time', '-of', 'json', str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    data = json.loads(result.stdout)
    times = np.array([float(f['best_effort_timestamp_time']) for f in data['frames']]) * 1000
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Video has missing or non-increasing presentation timestamps')
    stream = data['streams'][0]
    return times - times[0], int(stream['width']), int(stream['height'])


def normalized_box(value, width, height):
    cx, cy, w, h = np.asarray(value, dtype=float)
    if not np.isfinite([cx, cy, w, h]).all() or w <= 0 or h <= 0:
        return None
    return [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]


def normalized_hand(value, width, height):
    value = np.asarray(value, dtype=float)
    if value.shape != (21, 3):
        raise ValueError('Expected AV hand shape (21, 3)')
    # Only an entirely zero-filled legacy hand is a missing observation.
    # A zero coordinate within a returned hand can be a genuine image edge.
    if np.all(value == 0):
        return None
    xy = value[:, :2] * [width, height]
    valid = np.isfinite(xy).all(axis=1)
    if not valid.any():
        return None
    return {'xy': [p.tolist() if ok else None for p, ok in zip(xy, valid)],
            'valid': valid.tolist(), 'confidence': [None] * 21}


class AerialData:
    def __init__(self, path, times_ms, width, height):
        self.width, self.height = width, height
        self.times = np.asarray(times_ms, dtype=float)
        if not len(self.times) or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('AV video timing is invalid')
        with h5py.File(path, 'r') as h:
            self.arrays = {name: h[name][:] for name in
                           ('left_landmarks', 'right_landmarks', 'norm_gaze',
                            'rec_bboxes', 'surrogate_hands', 'timestamps')}
        a, count = self.arrays, len(self.times)
        for name, tail in [('left_landmarks', (21, 3)), ('right_landmarks', (21, 3)),
                           ('norm_gaze', (2,)), ('timestamps', ())]:
            if a[name].shape != (count, *tail):
                raise ValueError('AV frame count or shape differs from video: ' + name)
        for name in ('rec_bboxes', 'surrogate_hands'):
            if a[name].ndim != 3 or a[name].shape[0] != count or a[name].shape[2] != 4:
                raise ValueError('Unexpected AV box shape: ' + name)
        stored_ms = a['timestamps'].astype(float) * 1000
        if (not np.isfinite(stored_ms).all() or np.any(np.diff(stored_ms) <= 0)
                or np.max(np.abs(stored_ms - self.times)) > 2):
            raise ValueError('AV HDF5 timestamps differ from relative video PTS by more than 2 ms')
        self.maximum_timing_error_ms = float(np.max(np.abs(stored_ms - self.times)))

    def frame(self, index):
        if not 0 <= index < len(self.times):
            raise IndexError('Frame outside AV clip')
        result = {'kind': 'legacy_aerial', 'frame_index': index,
                  'timestamp_ms': float(self.times[index]),
                  'landmark_timestamp_ms': float(self.arrays['timestamps'][index]) * 1000,
                  'width': self.width, 'height': self.height, 'hands': [], 'objects': [], 'gaze': None}
        for side in ('left', 'right'):
            hand = normalized_hand(self.arrays[side+'_landmarks'][index], self.width, self.height)
            if hand is not None:
                result['hands'].append({**hand, 'actor': 'participant', 'handedness': side,
                    'track_id': None, 'topology': 'legacy_av_21', 'model_id': 'legacy_av',
                    'bbox': None, 'detection_confidence': None})
        for value in self.arrays['surrogate_hands'][index]:
            box = normalized_box(value, self.width, self.height)
            if box is not None:
                result['hands'].append({'xy': [None]*21, 'valid': [False]*21, 'confidence': [None]*21,
                    'actor': 'other_actor', 'handedness': 'unknown', 'track_id': None,
                    'topology': 'bbox_only', 'model_id': 'legacy_av', 'bbox': box, 'detection_confidence': None})
        for value in self.arrays['rec_bboxes'][index]:
            box = normalized_box(value, self.width, self.height)
            if box is not None:
                result['objects'].append({'bbox': box, 'class_name': 'AV detection',
                                         'confidence': None, 'model_id': 'legacy_av'})
        gaze = self.arrays['norm_gaze'][index].astype(float)
        # No validity field exists: omit zero-filled gaze, whose meaning is ambiguous.
        if np.isfinite(gaze).all() and np.any(gaze != 0):
            result['gaze'] = (gaze * [self.width, self.height]).tolist()
        return result
