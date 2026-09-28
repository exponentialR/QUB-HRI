"""Retain an executable copy of the isolated pipeline for a recorded run."""

import hashlib
import json
from pathlib import Path


def snapshot_code(destination:Path,package:Path|None=None) -> dict:
    package=package or Path(__file__).resolve().parent
    files={f'preprocessing/ul_ur_landmarks/{path.name}':path.read_bytes()
           for path in sorted(package.iterdir()) if path.is_file() and path.suffix in {'.py','.html','.js','.txt','.md'}}
    files['preprocessing/__init__.py']=(package.parent/'__init__.py').read_bytes()
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}
    manifest=json.dumps({'schema':'pipeline_source_snapshot_v1','files':hashes},sort_keys=True,indent=2)+'\n'
    identity=hashlib.sha256(manifest.encode()).hexdigest()
    root=destination/identity
    files['manifest.json']=manifest.encode()
    for name,data in files.items():
        target=root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            with target.open('xb') as stream:
                stream.write(data)
        except FileExistsError:
            if target.read_bytes()!=data:
                raise ValueError(f'Existing source snapshot is corrupt: {target}')
    return {'sha256':identity,'path':str(root.resolve()),'files':len(hashes)}
