from pathlib import Path

import pytest

from preprocessing.ul_ur_landmarks.source_snapshot import snapshot_code


def test_snapshot_preserves_versions_and_rejects_corruption(tmp_path):
    package=tmp_path/'input/preprocessing/ul_ur_landmarks'
    package.mkdir(parents=True)
    (package.parent/'__init__.py').write_text('')
    module=package/'runner.py';module.write_text('x=1\n')
    output=tmp_path/'saved'
    first=snapshot_code(output,package)
    assert snapshot_code(output,package)==first
    module.write_text('x=2\n')
    second=snapshot_code(output,package)
    assert first['sha256']!=second['sha256']
    old=Path(first['path'])/'preprocessing/ul_ur_landmarks/runner.py'
    assert old.read_text()=='x=1\n'
    module.write_text('x=1\n');old.write_text('corrupt')
    with pytest.raises(ValueError,match='corrupt'):
        snapshot_code(output,package)
