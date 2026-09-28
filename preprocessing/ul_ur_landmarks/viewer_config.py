"""Viewer configuration: CLI overrides environment, then a literal .env file."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_env(path):
    """Read KEY=value entries without executing commands or expanding variables."""
    values = {}
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        key = key.strip()
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError(f'{path.name}:{number}: expected KEY=value')
        try:
            parts = shlex.split(value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f'{path.name}:{number}: {exc}') from exc
        if len(parts) > 1:
            raise ValueError(f'{path.name}:{number}: quote values containing spaces')
        if key in values:
            raise ValueError(f'{path.name}:{number}: duplicate setting {key}')
        values[key] = parts[0] if parts else ''
    return values


@dataclass(frozen=True)
class ViewerConfig:
    video_root: Path
    landmarks_root: Path
    quality_root: Path | None
    port: int
    check: bool


def configuration(argv=None, *, environ=None, repo_root=REPO_ROOT):
    p = argparse.ArgumentParser(description='QUB-PHEO local AV/UL/UR/LL/LR visualiser')
    p.add_argument('--env-file', type=Path, help='Defaults to .env beside visualise.py')
    p.add_argument('--dataset-root', help='Directory containing videos/ and landmarks/')
    p.add_argument('--video-root', help='Override the video directory')
    p.add_argument('--landmarks-root', help='Override the landmark directory')
    p.add_argument('--quality-root', help='Optional UL/UR coverage report directory')
    p.add_argument('--port', type=int, help='Local port; default 8767')
    p.add_argument('--check', action='store_true', help='Check configuration and discovery, then exit')
    a = p.parse_args(argv)
    env = os.environ if environ is None else environ
    env_file = a.env_file.expanduser().resolve() if a.env_file else repo_root/'.env'
    if a.env_file and not env_file.is_file():
        raise ValueError('Requested environment file does not exist: '+str(env_file))
    values = read_env(env_file) if env_file.exists() else {}

    def setting(name, cli, default=None):
        if cli is not None:
            return cli, Path.cwd()
        if name in env:
            return env[name], Path.cwd()
        return values.get(name, default), env_file.parent

    def path_setting(name, cli, default=None):
        value, base = setting(name, cli, default)
        if not value:
            return None
        path = Path(value).expanduser()
        return (base/path).resolve() if not path.is_absolute() else path.resolve()

    root = path_setting('QUB_PHEO_DATASET_ROOT', a.dataset_root)
    videos = path_setting('QUB_PHEO_VIDEO_ROOT', a.video_root, str(root/'videos') if root else None)
    landmarks = path_setting('QUB_PHEO_LANDMARKS_ROOT', a.landmarks_root, str(root/'landmarks') if root else None)
    quality = path_setting('QUB_PHEO_QUALITY_ROOT', a.quality_root)
    if videos is None or landmarks is None:
        raise ValueError('Set QUB_PHEO_DATASET_ROOT in .env, or provide --dataset-root. See .env.example.')
    for label, path in [('Video', videos), ('Landmark', landmarks), ('Quality report', quality)]:
        if path is not None and not path.is_dir():
            raise ValueError(f'{label} directory does not exist: {path}')
    raw_port, _ = setting('QUB_PHEO_VIEWER_PORT', a.port, '8767')
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError('Viewer port must be an integer between 1 and 65535') from exc
    if not 1 <= port <= 65535:
        raise ValueError('Viewer port must be between 1 and 65535')
    return ViewerConfig(videos, landmarks, quality, port, a.check)
