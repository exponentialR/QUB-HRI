"""Inventory checks independent of model inference and annotation tooling."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from preprocessing.ul_ur_landmarks.inventory import discover, inventory, main, parse_clip, probe, select_pilot


def test_exact_ids_and_pairing(tmp_path: Path) -> None:
    folder = tmp_path / "HCS"
    folder.mkdir()
    for filename in ("p01-CAM_UL-TASK-HCS-0_3.mp4", "p01-CAM_UR-TASK-HCS-0_3.mp4",
                     "p010-CAM_UL-TASK-HCS-0_3.mp4"):
        (folder / filename).touch()
    assert parse_clip(folder / "p010-CAM_UL-TASK-HCS-0_3.mp4", tmp_path, {"P01"}) is None
    pairs, issues = discover(tmp_path, {"P01"})
    assert len(pairs) == 1
    assert not issues


def test_selection_requires_temporal_context() -> None:
    rows = [{"pair_id": f"X/P{index:02d}-T", "pid": f"P{index:02d}", "subtask_dir": "X",
             "metadata_match": True,
             "views": {view: {"duration_s": duration} for view in ("CAM_UL", "CAM_UR")}}
            for index, duration in enumerate((0.1, 2.0, 3.1, 4.0))]
    selected = select_pilot(rows, 2)
    assert [row["pid"] for row in selected] == ["P02", "P03"]


def test_full_scope_keeps_unlisted_ids_and_reports_pair_issues(tmp_path: Path) -> None:
    for filename in ("P01-CAM_UL-T.mp4", "P01-CAM_UR-T.mp4", "P010-CAM_UL-T.mp4",
                     "P010-CAM_UR-T.mp4", "P69-CAM_UL-T.mp4", "p01-cam_ul-T.MP4",
                     "P70-CAM_LL-T.mp4", "notes.mp4"):
        (tmp_path / filename).touch()
    pairs, issues = discover(tmp_path, None)
    assert len(pairs) == 3
    assert {issue["kind"] for issue in issues} == {"duplicate_view", "missing_view"}
    assert len(discover(tmp_path, {"P01"})[0]) == 1
    assert not discover(tmp_path, set())[0]
    with pytest.raises(ValueError, match="Specify either"):
        inventory(tmp_path)
    ids = tmp_path / "ids.txt"
    ids.write_text("P01\n")
    with pytest.raises(ValueError, match="Specify either"):
        inventory(tmp_path, ids, all_participants=True)


def test_full_inventory_metadata_and_original_list_are_preserved(tmp_path: Path, monkeypatch) -> None:
    for pid in ("P01", "P010"):
        for view in ("CAM_UL", "CAM_UR"):
            (tmp_path / f"{pid}-{view}-T.mp4").touch()
    ids = tmp_path / "ids.txt"
    ids.write_text("P01\n")
    monkeypatch.setattr("preprocessing.ul_ur_landmarks.inventory.probe", lambda path: {
        "width": 64, "height": 48, "declared_frames": 2, "fps": 2.0, "duration_s": 1.0})
    subset, _ = inventory(tmp_path, ids)
    all_rows, issues = inventory(tmp_path, all_participants=True)
    assert len(subset) == 1 and len(all_rows) == 2 and issues == []
    assert ids.read_text() == "P01\n"


def test_duplicate_pairs_and_sources_outside_root_are_not_inventory_candidates(tmp_path, monkeypatch):
    root = tmp_path/'videos'; root.mkdir()
    for name in ('P01-CAM_UL-T.mp4', 'p01-cam_ul-T.MP4', 'P01-CAM_UR-T.mp4'):
        (root/name).touch()
    outside = tmp_path/'external.mp4'; outside.write_bytes(b'external')
    (root/'P02-CAM_UL-T.mp4').symlink_to(outside)
    (root/'P02-CAM_UR-T.mp4').touch()
    probed = []
    def metadata(path):
        probed.append(path)
        return {'width':64, 'height':48, 'fps':2., 'duration_s':4., 'declared_frames':8}
    monkeypatch.setattr('preprocessing.ul_ur_landmarks.inventory.probe', metadata)
    rows, issues = inventory(root, all_participants=True)
    assert rows == []
    assert {i['kind'] for i in issues} == {'duplicate_view','source_outside_root','missing_view'}
    assert root/'P02-CAM_UL-T.mp4' not in probed


def test_metadata_mismatches_and_unreadable_clips_are_reported(tmp_path, monkeypatch):
    for pid in ('P01','P02'):
        for view in ('CAM_UL','CAM_UR'):
            (tmp_path/f'{pid}-{view}-T.mp4').touch()
    def metadata(path):
        if path.name.startswith('P02-CAM_UR'):
            raise ValueError('Unreadable synthetic input')
        return {'width':64, 'height':48, 'fps':2., 'duration_s':4.,
                'declared_frames':9 if 'CAM_UR' in path.name else 8}
    monkeypatch.setattr('preprocessing.ul_ur_landmarks.inventory.probe', metadata)
    rows, issues = inventory(tmp_path, all_participants=True)
    assert len(rows) == 1 and rows[0]['pid'] == 'P01'
    assert not rows[0]['metadata_match']
    assert {i['kind'] for i in issues} == {'metadata_mismatch','unreadable'}
    with pytest.raises(ValueError, match='Only 0'):
        select_pilot(rows, 1)


def test_cli_probes_synthetic_videos_and_preserves_selection_and_scope(tmp_path, monkeypatch):
    videos = tmp_path/'videos'; videos.mkdir()
    seed = videos/'P01-CAM_UL-T.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=size=64x48:rate=2',
                    '-frames:v','8','-pix_fmt','yuv420p',str(seed)],check=True)
    for name in ('P01-CAM_UR-T.mp4','P010-CAM_UL-T.mp4','P010-CAM_UR-T.mp4'):
        (videos/name).write_bytes(seed.read_bytes())
    before = {p.name:p.read_bytes() for p in videos.iterdir()}
    assert probe(seed)['declared_frames'] == 8
    output = tmp_path/'inventory'
    args = ['inventory','--input-root',str(videos),'--all-participants',
            '--output-root',str(output),'--workers','1','--pilot-pairs','1']
    monkeypatch.setattr(sys,'argv',args)
    main()
    assert len(json.loads((output/'inventory.json').read_text())) == 2
    assert json.loads((output/'inventory_issues.json').read_text()) == []
    selection = output/'pilot_pairs.json'; saved = selection.read_bytes()
    monkeypatch.setattr(sys,'argv',args+['--inventory-only'])
    main()
    assert selection.read_bytes() == saved
    ids = tmp_path/'ids.txt'; ids.write_text('P01\n')
    monkeypatch.setattr(sys,'argv',['inventory','--input-root',str(videos),'--consented',str(ids),
                                  '--output-root',str(output),'--inventory-only'])
    with pytest.raises(SystemExit): main()
    assert json.loads((output/'inventory_scope.json').read_text())['scope'] == 'all_local_ul_ur'
    assert {p.name:p.read_bytes() for p in videos.iterdir()} == before


def test_cli_rejects_reports_inside_sources(tmp_path, monkeypatch):
    videos = tmp_path/'videos'; videos.mkdir()
    monkeypatch.setattr(sys,'argv',['inventory','--input-root',str(videos),'--all-participants',
                                  '--output-root',str(videos/'reports'),'--inventory-only'])
    with pytest.raises(SystemExit): main()
    assert not (videos/'reports').exists()
