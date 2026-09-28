import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.refine_collection import CollectionRefiner, audit_preservation, producer_state
from preprocessing.ul_ur_landmarks.schema import Frame, sha256, write_hdf5


def fixture(tmp_path):
    video = tmp_path/'videos'; video.mkdir()
    parent = tmp_path/'parent'; parent.mkdir()
    model = {'hand_mode':'native_hand5', 'hand_models':{'wholebody':{'model':'rtmw_l_384x288'}}}
    key = 'rtmw_collection_v1_1'; pair = './P01-T'
    row = {'pair_id':pair, 'pid':'P01', 'views':{}}
    for view in ('CAM_UL', 'CAM_UR'):
        name = 'P01-'+view+'-T.mp4'
        (video/name).write_bytes(view.encode())
        meta = {'relpath':name, 'width':200, 'height':200, 'declared_frames':2}
        row['views'][view] = meta
        source = {**meta, 'pair_id':pair, 'pid':'P01', 'view':view, 'sha256':sha256(video/name)}
        pose = np.full((133,2), np.nan, dtype=np.float32); scores = np.zeros(133, dtype=np.float32)
        pose[91:112] = np.linspace([0,0],[20,20],21); pose[9] = [0,0]
        scores[91:112] = 5; scores[9] = 5
        output = parent/'results'/key/pair/(view+'.h5')
        write_hdf5(output, [Frame(i*20., pose.copy(), scores.copy()) for i in range(2)],
                   source=source, model=model, pose_count=133, pose_topology='coco_wholebody_133', schema_version='1.1')
    inventory = tmp_path/'inventory.json'; inventory.write_text(json.dumps([row]))
    (parent/'run_scope.json').write_text(json.dumps({'manifest_sha256':sha256(inventory), 'input_root':str(video)}))
    (parent/(key+'_configuration.json')).write_text(json.dumps(model))
    records = [{'pair_id':pair, 'view':view, 'model_key':key, 'frames':2, 'status':'written'} for view in row['views']]
    (parent/'run_ledger.jsonl').write_text(json.dumps(records[0])+'\n')
    return inventory, parent, tmp_path/'refined', records


def test_live_parent_partial_ledger_safe_restart_and_output_revalidation(tmp_path):
    inventory, parent, output, rows = fixture(tmp_path)
    runner = CollectionRefiner(inventory, parent, output); runner.prepare()
    try:
        assert runner.process_available()[0] == 1
        assert runner.status('test')['remaining_clips'] == 1
        with (parent/'run_ledger.jsonl').open('a') as stream: stream.write(json.dumps(rows[1])[:15])
        assert runner.process_available() == (0, True)
        with (parent/'run_ledger.jsonl').open('a') as stream: stream.write(json.dumps(rows[1])[15:]+'\n')
        assert runner.process_available()[0] == 1
        assert runner.status('test')['all_expected_outputs_logged_valid']
        duplicate = CollectionRefiner(inventory, parent, output)
        with pytest.raises(BlockingIOError): duplicate.prepare()
    finally: runner.close()
    files = sorted((output/'results').rglob('*.h5')); hashes = [sha256(path) for path in files]
    with (output/'run_ledger.jsonl').open('a') as stream: stream.write('{"interrupted":')
    resumed = CollectionRefiner(inventory, parent, output); resumed.prepare()
    try:
        assert resumed.process_available()[0] == 2
        assert resumed.status('test')['all_expected_outputs_logged_valid']
        assert [sha256(path) for path in files] == hashes
        assert len(list(output.glob('interrupted_ledger_tail_*'))) == 1
    finally: resumed.close()
    with h5py.File(files[0], 'r+') as h: h['participant/pose/xy_px'][0,9] = [5,5]
    resumed = CollectionRefiner(inventory, parent, output); resumed.prepare()
    try:
        resumed.process_available()
        status = resumed.status('test')
        assert status['logged_valid_clips'] == 1
        assert 'Preservation audit failed' in status['latest_failures'][0]['error']
    finally: resumed.close()


def test_refinement_rejects_wrong_inventory_model_output_and_source(tmp_path):
    inventory, parent, output, rows = fixture(tmp_path)
    with pytest.raises(ValueError, match='separate output'):
        CollectionRefiner(inventory, parent, parent/'nested')
    runner = CollectionRefiner(inventory, parent, output); runner.prepare(); runner.close()
    different = CollectionRefiner(inventory, parent, output, threshold=5.)
    with pytest.raises(ValueError, match='frozen inventory'): different.prepare()
    (tmp_path/'videos'/'P01-CAM_UL-T.mp4').write_bytes(b'changed source')
    runner = CollectionRefiner(inventory, parent, output); runner.prepare()
    try:
        runner.process_available()
        assert 'Source hash mismatch' in runner.status('test')['latest_failures'][0]['error']
        assert not list(output.rglob('*.h5'))
    finally: runner.close()
    inventory.write_text('[]')
    with pytest.raises(ValueError, match='inventory differ'): CollectionRefiner(inventory, parent, output)


def test_failed_producer_query_is_unknown_not_completion(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a,1,'inactive\n','failure'))
    assert producer_state('test.service') == 'unknown'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a,0,'active\n',''))
    assert producer_state('test.service') == 'active'
