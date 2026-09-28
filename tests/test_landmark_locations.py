import json
import sys
from pathlib import Path

import pytest

from preprocessing.ul_ur_landmarks.locations import Locations, contained, filename_path
from preprocessing.ul_ur_landmarks.organize import copy_verified, main
from preprocessing.ul_ur_landmarks.schema import Frame, sha256, write_hdf5


def test_exact_filename_and_path_guards(tmp_path):
    row = {'views': {'CAM_UL': {'relpath': 'BHO/p01-CAM_UL-T-0.000_1.00.mp4'}}}
    assert filename_path(tmp_path, row, 'CAM_UL') == tmp_path/'BHO/p01-CAM_UL-T-0.000_1.00.h5'
    for relative in ('../outside.mp4', '/outside.mp4'):
        with pytest.raises(ValueError):
            contained(tmp_path, relative)
    (tmp_path/'escape').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        contained(tmp_path, 'escape/outside.mp4')
    row['views']['CAM_UL']['relpath'] = 'BHO/p01-CAM_AV-T.mp4'
    with pytest.raises(ValueError):
        filename_path(tmp_path, row, 'CAM_UL')


def test_copy_is_independent_and_refuses_changes(tmp_path):
    source, destination = tmp_path/'original', tmp_path/'new'/'copied.h5'
    source.write_bytes(b'original bytes')
    digest = sha256(source)
    assert copy_verified(source, destination, digest) == 'copied'
    assert source.stat().st_ino != destination.stat().st_ino
    assert copy_verified(source, destination, digest) == 'already_identical'
    destination.write_bytes(b'changed')
    with pytest.raises(ValueError, match='refusing overwrite'):
        copy_verified(source, destination, digest)
    assert source.read_bytes() == b'original bytes'
    assert destination.read_bytes() == b'changed'


def test_organization_and_rerun_preserve_av_and_provenance(tmp_path, monkeypatch):
    collection, videos, landmarks = (tmp_path/name for name in ('collection', 'videos', 'landmarks'))
    collection.mkdir(); videos.mkdir(); landmarks.mkdir()
    av = landmarks/'T'/'p01-CAM_AV-T.h5'
    av.parent.mkdir(); av.write_bytes(b'historical aerial records')
    model = 'fixture'
    row = {'pair_id': 'T/P01-T-0_1', 'pid': 'P01', 'views': {}}
    records = []
    for view in ('CAM_UL', 'CAM_UR'):
        relative = f'T/p01-{view}-T-0_1.mp4'
        video = videos/relative
        video.parent.mkdir(exist_ok=True); video.write_bytes(view.encode())
        row['views'][view] = {'relpath': relative, 'width': 20, 'height': 20, 'declared_frames': 1}
        old = collection/'results'/model/row['pair_id']/(view+'.h5')
        old.parent.mkdir(parents=True, exist_ok=True)
        identity = {**row['views'][view], 'view': view, 'pid': 'P01',
                    'pair_id': row['pair_id'], 'sha256': sha256(video)}
        write_hdf5(old, [Frame(0.)], source=identity, model={},
                   pose_topology='coco_wholebody_133', pose_count=133, schema_version='1.1')
        records.append({'pair_id': row['pair_id'], 'view': view, 'output': str(old),
                        'status': 'validated', 'output_sha256': sha256(old), 'frames': 1})
    inventory = tmp_path/'inventory.json'; inventory.write_text(json.dumps([row]))
    scope = collection/'run_scope.json'
    scope.write_text(json.dumps({'manifest_sha256': sha256(inventory), 'input_root': str(tmp_path/'segmented')}))
    scope_bytes = scope.read_bytes()
    manifest = collection/'validated.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
    (collection/(model+'_configuration.json')).write_text('{}')
    (collection/'run_ledger.jsonl').write_text(''.join(json.dumps({**r, 'status': 'written',
        'model_key': model})+'\n' for r in records))
    args = ['organize', '--collection-root', str(collection), '--inventory', str(inventory),
            '--validated-manifest', str(manifest), '--validated-manifest-sha256', sha256(manifest),
            '--video-root', str(videos), '--landmarks-root', str(landmarks), '--model-key', model]
    monkeypatch.setattr(sys, 'argv', args + ['--audit-root', str(tmp_path/'dryrun')])
    main()
    assert not (tmp_path/'dryrun').exists()
    assert len(list(landmarks.rglob('*.h5'))) == 1
    for attempt in range(2):
        audit = tmp_path/f'audit{attempt}'
        monkeypatch.setattr(sys, 'argv', args + ['--audit-root', str(audit), '--apply'])
        main()
        summary = json.loads((audit/'summary.json').read_text())
        assert summary['copied' if attempt == 0 else 'already_identical'] == 2
        assert summary['existing_cam_av_preserved'] == 1
        current = Locations(collection, inventory)
        assert current.video_root == videos
        assert len(current.outputs_on_disk(model)) == 2  # AV is not an unexpected UL/UR output.
        for record in records:
            path = current.output(row, record['view'], model)
            assert sha256(path) == record['output_sha256']
            assert Path(record['output']).exists()
    assert av.read_bytes() == b'historical aerial records'
    assert scope.read_bytes() == scope_bytes
    # Both consumers must read canonical files and ignore the unrelated AV schema.
    from preprocessing.ul_ur_landmarks.collection_status import main as status_main
    monkeypatch.setattr(sys, 'argv', ['status', '--inventory', str(inventory),
        '--output-root', str(collection), '--model-key', model, '--validate-all'])
    status_main()
    status = json.loads((collection/'collection_status.json').read_text())
    assert status['collection_files_complete'] and not status['unexpected_outputs']
    from preprocessing.ul_ur_landmarks.collection_quality import main as quality_main
    report = tmp_path/'quality'
    monkeypatch.setattr(sys, 'argv', ['quality', '--inventory', str(inventory),
        '--output-root', str(collection), '--report-root', str(report),
        '--validated-manifest', str(tmp_path/'audit0'/'validated_outputs.jsonl')])
    quality_main()
    assert json.loads((report/'summary.json').read_text())['complete_collection_coverage']

    # Exercise the public normalization CLI with the organization manifest,
    # including dry run, independent archives, reruns and downstream audits.
    from preprocessing.ul_ur_landmarks.normalize import main as normalize_main
    organized = tmp_path/'audit0'/'validated_outputs.jsonl'
    normalized_audit = tmp_path/'normalization'
    norm_args = ['normalize', '--collection-root', str(collection), '--inventory', str(inventory),
        '--validated-manifest', str(organized), '--validated-manifest-sha256', sha256(organized),
        '--audit-root', str(normalized_audit), '--model-key', model, '--workers', '1']
    monkeypatch.setattr(sys, 'argv', norm_args)
    normalize_main()
    assert not normalized_audit.exists()
    for record in records:
        assert sha256(current.output(row, record['view'], model)) == record['output_sha256']
    for attempt in range(2):
        monkeypatch.setattr(sys, 'argv', norm_args + ['--apply'])
        normalize_main()
        normalized_status = json.loads((normalized_audit/'status.json').read_text())
        assert normalized_status['collection_complete']
        assert normalized_status['written' if attempt == 0 else 'already_valid'] == 2
        assert normalized_status['av_files_preserved'] == 1
        for record in records:
            assert sha256(Path(record['output'])) == record['output_sha256']
            assert sha256(current.output(row, record['view'], model)) != record['output_sha256']
    assert av.read_bytes() == b'historical aerial records'
    assert scope.read_bytes() == scope_bytes
    # The old copy manifest remains provenance; fresh validation describes the
    # current normalized files and can be consumed by the coverage report.
    monkeypatch.setattr(sys, 'argv', ['status', '--inventory', str(inventory),
        '--output-root', str(collection), '--model-key', model, '--validate-all'])
    status_main()
    current_status = json.loads((collection/'collection_status.json').read_text())
    assert current_status['collection_files_complete']
    monkeypatch.setattr(sys, 'argv', ['quality', '--inventory', str(inventory),
        '--output-root', str(collection), '--report-root', str(tmp_path/'normalized_quality'),
        '--validated-manifest', current_status['validated_output_manifest']])
    quality_main()
    assert json.loads((tmp_path/'normalized_quality/summary.json').read_text())['complete_collection_coverage']

    with pytest.raises(ValueError, match='Model key'):
        current.output(row, 'CAM_UL', 'wrong_model')
    # A moved source must still be the very same recording.
    # Restore only the synthetic canonical fixture to its original pixel layout
    # so the organization collision guard does not mask the source-hash check.
    for record in records:
        current.output(row, record['view'], model).write_bytes(Path(record['output']).read_bytes())
    (videos/row['views']['CAM_UL']['relpath']).write_bytes(b'different recording')
    monkeypatch.setattr(sys, 'argv', args + ['--audit-root', str(tmp_path/'bad_source'), '--apply'])
    with pytest.raises(ValueError, match='video content differs'):
        main()
