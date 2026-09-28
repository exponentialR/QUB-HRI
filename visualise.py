#!/usr/bin/env python3
"""Start the repository's local visualiser using .env or explicit CLI paths."""

import json
import shutil
import sys

from preprocessing.ul_ur_landmarks.viewer_config import configuration


def main(argv=None):
    try:
        config = configuration(argv)
        if shutil.which('ffprobe') is None:
            raise ValueError('ffprobe is required. Install FFmpeg and make ffprobe available on PATH; see docs/visualiser.md.')
        try:
            from preprocessing.ul_ur_landmarks.viewer import DatasetCollection, serve
        except ImportError as exc:
            raise ValueError('Viewer dependencies are missing. Run: python -m pip install -r requirements-viewer.txt') from exc
        collection = DatasetCollection(config.video_root, config.landmarks_root, config.quality_root)
        print(json.dumps(collection.discovery_summary, indent=2), flush=True)
        if config.check:
            print('Configuration and file discovery passed. Selected clips are validated when opened; no files changed.')
        else:
            serve(collection, config.port)
        return 0
    except (ValueError, OSError) as exc:
        print('Visualiser: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
