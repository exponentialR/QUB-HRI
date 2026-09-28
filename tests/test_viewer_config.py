from pathlib import Path

import pytest

from preprocessing.ul_ur_landmarks.viewer_config import configuration, read_env


def dataset(root):
    (root/'videos').mkdir(parents=True)
    (root/'landmarks').mkdir()
    return root


def test_env_quotes_relative_paths_and_precedence(tmp_path, monkeypatch):
    repo=tmp_path/'checkout';repo.mkdir()
    local=dataset(repo/'data with spaces');other=dataset(tmp_path/'other');cli=dataset(tmp_path/'cli')
    env_file=repo/'.env';env_file.write_text('QUB_PHEO_DATASET_ROOT="data with spaces" # local\nQUB_PHEO_VIEWER_PORT=8768\n')
    monkeypatch.chdir(tmp_path)
    config=configuration([],environ={},repo_root=repo)
    assert config.video_root==local/'videos' and config.port==8768
    environment={'QUB_PHEO_DATASET_ROOT':str(other),'QUB_PHEO_VIEWER_PORT':'8769'}
    assert configuration([],environ=environment,repo_root=repo).video_root==other/'videos'
    config=configuration(['--dataset-root',str(cli),'--port','8770'],environ=environment,repo_root=repo)
    assert config.video_root==cli/'videos' and config.port==8770
    assert environment['QUB_PHEO_DATASET_ROOT']==str(other)  # never mutate process configuration


def test_explicit_env_file_and_separate_roots(tmp_path):
    data=dataset(tmp_path/'data')
    env=tmp_path/'settings.env'
    env.write_text(f'export QUB_PHEO_VIDEO_ROOT="{data}/videos"\nQUB_PHEO_LANDMARKS_ROOT="{data}/landmarks"\n')
    config=configuration(['--env-file',str(env),'--check'],environ={},repo_root=tmp_path)
    assert config.check and config.landmarks_root==data/'landmarks'


def test_values_are_literal_and_bad_settings_fail(tmp_path):
    path=tmp_path/'.env'
    path.write_text('VALUE="$(touch unsafe)"\nSECOND="${HOME}/data"\n')
    assert read_env(path)=={'VALUE':'$(touch unsafe)','SECOND':'${HOME}/data'}
    assert not (tmp_path/'unsafe').exists()
    for contents in ('bad setting\n','KEY="unclosed\n','KEY=a\nKEY=b\n','KEY=two words\n'):
        path.write_text(contents)
        with pytest.raises(ValueError):read_env(path)
    path.unlink()
    with pytest.raises(ValueError,match='Set QUB_PHEO_DATASET_ROOT'):
        configuration([],environ={},repo_root=tmp_path)
    data=dataset(tmp_path/'data')
    for port in ('abc','0','65536'):
        with pytest.raises(ValueError,match='port'):
            configuration([],environ={'QUB_PHEO_DATASET_ROOT':str(data),'QUB_PHEO_VIEWER_PORT':port},repo_root=tmp_path)
    with pytest.raises(ValueError,match='does not exist'):
        configuration(['--env-file',str(tmp_path/'absent')],environ={},repo_root=tmp_path)
