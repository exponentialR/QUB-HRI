"""Discover portable dataset copies using relative filenames, without run logs."""

from collections import Counter, defaultdict
from pathlib import Path
import re

from .locations import contained


NAME = re.compile(r'^(P\d+)-(CAM_AV|CAM_UL|CAM_UR|CAM_LL|CAM_LR)-(.+)$', re.IGNORECASE)


def scan(root, suffix):
    found = {}
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() != suffix or not path.is_file():
            continue
        match = NAME.fullmatch(path.stem)
        if match is None:
            continue
        relative = path.relative_to(root)
        # Reject even an apparently matching filename if a symlink escapes the root.
        contained(root, relative)
        pid, view, rest = match.groups()
        key = f'{relative.parent.as_posix()}/{pid.upper()}-{rest}', view.upper()
        if key in found:
            raise ValueError(f'Duplicate {key[1]} clip identity: {key[0]}')
        found[key] = relative.as_posix()
    return found


def discover_dataset(video_root, landmarks_root):
    videos, landmarks = scan(video_root, '.mp4'), scan(landmarks_root, '.h5')
    rows = defaultdict(dict)
    for pair_id, view in sorted(videos.keys() & landmarks.keys()):
        rows[pair_id][view] = {'relpath': videos[pair_id, view],
                               'landmark_relpath': landmarks[pair_id, view]}
    if not rows:
        raise ValueError('No matching AV/UL/UR/LL/LR .mp4 and .h5 files found. Expected videos/<task>/<video-stem>.mp4 and landmarks/<task>/<video-stem>.h5.')
    result = [{'pair_id': pair_id, 'pid': Path(pair_id).name.split('-')[0],
               'subtask_dir': Path(pair_id).parent.as_posix(), 'views': views}
              for pair_id, views in sorted(rows.items())]
    summary = {'clip_groups': len(result), 'matched_views': dict(Counter(v for row in result for v in row['views'])),
               'videos_without_landmarks': len(videos.keys()-landmarks.keys()),
               'landmarks_without_videos': len(landmarks.keys()-videos.keys())}
    return result, summary
