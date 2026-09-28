"""Focused checks for the isolated UL/UR pilot contract."""

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from preprocessing.ul_ur_landmarks.inventory import discover, inventory, parse_clip, select_pilot
from preprocessing.ul_ur_landmarks.evaluate import evaluate, evaluate_hands, matching_hand, score_frame
from preprocessing.ul_ur_landmarks.mediapipe_backend import MediaPipeBackend, normalized_points
from preprocessing.ul_ur_landmarks.runner import frame_pts_ms, freeze_run_scope, selected_rows
from preprocessing.ul_ur_landmarks.sample import sample_positions
from preprocessing.ul_ur_landmarks.schema import Frame, Hand, validate_hdf5, write_hdf5
from preprocessing.ul_ur_landmarks.tracking import HandTracker, assign_actor


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


def test_full_runner_scope_retains_identity_checks_and_pilot_limit(tmp_path: Path) -> None:
    rows = []
    for i in range(21):
        pid = f"P{i:02d}"
        views = {}
        for view in ("CAM_UL", "CAM_UR"):
            name = f"{pid}-{view}-T.mp4"
            (tmp_path / name).touch()
            views[view] = {"relpath": name}
        rows.append({"pid": pid, "pair_id": f"./{pid}-T", "views": views})
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="at most 20"):
        selected_rows(manifest, {row["pid"] for row in rows}, tmp_path)
    assert len(selected_rows(manifest, None, tmp_path, max_pairs=None)) == 21
    rows[0]["pid"] = "P999"
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Participant identity mismatch"):
        selected_rows(manifest, None, tmp_path, max_pairs=None)


def test_full_run_scope_is_immutable_on_rerun(tmp_path: Path) -> None:
    path = tmp_path / "run_scope.json"
    identity = {"manifest_sha256": "original", "scope": "all_local_ul_ur"}
    freeze_run_scope(path, identity)
    previous = path.stat().st_mtime_ns
    freeze_run_scope(path, identity)
    assert path.stat().st_mtime_ns == previous
    with pytest.raises(ValueError, match="different frozen inventory"):
        freeze_run_scope(path, {**identity, "manifest_sha256": "changed"})
    assert json.loads(path.read_text()) == identity
    assert list(tmp_path.iterdir()) == [path]


def test_reference_includes_three_consecutive_frames() -> None:
    positions = sample_positions(69)
    assert len(positions) == 5
    assert positions[2] == positions[1] + 1
    assert positions[3] == positions[2] + 1


def test_missing_points_are_invalid_but_zero_can_be_valid(tmp_path: Path) -> None:
    source = {"sha256": "abc"}
    model = {"name": "fixture"}
    hand_xy = np.zeros((21, 2), dtype=np.float32)
    frame = Frame(timestamp_ms=0, pose_xy=np.array([[0, 0], [np.nan, np.nan]], dtype=np.float32),
                  pose_confidence=np.array([1, 0], dtype=np.float32),
                  hands=[Hand(xy=hand_xy, confidence=np.ones(21), actor="other_actor",
                              handedness="left", track_id=7)])
    path = tmp_path / "fixture.h5"
    write_hdf5(path, [frame, Frame(timestamp_ms=33)], source=source, model=model,
               pose_topology="fixture_2", pose_count=2)
    assert validate_hdf5(path, source_sha256="abc", model=model)["frames"] == 2
    import h5py
    with h5py.File(path) as data:
        assert bool(data["participant/pose/valid"][0, 0])
        assert not bool(data["participant/pose/valid"][0, 1])
        assert np.isnan(data["participant/pose/xy_px"][1]).all()
        assert data["hands/actor"][0] == 2
        assert data["hands/track_id"][0] == 7
    with pytest.raises(FileExistsError):
        write_hdf5(path, [frame], source=source, model=model, pose_topology="fixture_2", pose_count=2)
    with pytest.raises(ValueError, match="Source hash"):
        validate_hdf5(path, source_sha256="different", model=model)


def test_coordinates_and_actor_identity() -> None:
    class Point:
        def __init__(self, x, y, visibility):
            self.x, self.y, self.visibility = x, y, visibility
    xy, score = normalized_points([Point(0, 0, 1), Point(.5, .25, 0), Point(1.2, .5, 1)], 100, 80)
    assert np.array_equal(xy[0], [0, 0])
    assert np.array_equal(xy[1], [50, 20])
    assert score.tolist() == [1, 0, 0]
    assert np.isnan(xy[2]).all()
    participant = [np.array([20.0, 20.0])]
    assert assign_actor(np.tile([20, 20], (21, 1)), participant, [], 1000, 1000) == "participant"
    assert assign_actor(np.tile([900, 700], (21, 1)), participant, [], 1000, 1000) == "other_actor"
    assert assign_actor(np.tile([100, 100], (21, 1)), [], [], 1000, 1000) == "unknown"
    tracker = HandTracker(1000, 1000)
    first = Hand(np.tile([20, 20], (21, 1)), np.ones(21), actor="participant")
    second = Hand(np.tile([22, 21], (21, 1)), np.ones(21), actor="participant")
    tracker.update([first])
    tracker.update([second])
    assert first.track_id == second.track_id


def test_rerun_validation_rejects_corrupted_missingness_and_timestamps(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "damaged.h5"
    model = {"name": "fixture"}
    write_hdf5(path, [Frame(timestamp_ms=0)], source={"sha256": "fixture"}, model=model,
               pose_topology="mediapipe_pose_33", pose_count=33)
    with h5py.File(path, "r+") as data:
        data["participant/pose/xy_px"][0, 0] = [0, 0]
    with pytest.raises(ValueError, match="Inconsistent validity"):
        validate_hdf5(path, source_sha256="fixture", model=model)
    with h5py.File(path, "r+") as data:
        data["participant/pose/xy_px"][0, 0] = [np.nan, np.nan]
        data["frames/timestamp_ms"][0] = np.nan
    with pytest.raises(ValueError, match="finite and nonnegative"):
        validate_hdf5(path, source_sha256="fixture", model=model)


def test_pose_guided_face_crop_and_frame_coordinates() -> None:
    backend = object.__new__(MediaPipeBackend)
    backend.width, backend.height = 1728, 972
    backend.face_rect = (380, 0, 1347, 583)
    pose = np.full((33, 2), np.nan, dtype=np.float32)
    pose[0] = [1100, 150]
    pose[11] = [1000, 240]
    pose[12] = [1160, 240]
    rect = backend._pose_face_rect(pose)
    assert rect == (956, 0, 1244, 288)
    assert backend._pose_face_rect(None) == backend.face_rect
    class Point:
        x, y, visibility = 0.5, 0.5, None
    crop_xy, _ = normalized_points([Point()], rect[2] - rect[0], rect[3] - rect[1])
    assert np.array_equal(crop_xy[0] + rect[:2], [1100, 144])


def test_real_video_timestamps_from_synthetic_clip(tmp_path: Path) -> None:
    path = tmp_path / "two_frames.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=64x48:rate=2",
                    "-frames:v", "2", "-pix_fmt", "yuv420p", str(path)], check=True)
    assert frame_pts_ms(path) == [0, 500]


def test_evaluation_counts_visible_other_actor_hand(tmp_path: Path) -> None:
    xy = np.tile([100.0, 100.0], (21, 1))
    frame = Frame(timestamp_ms=0, hands=[Hand(xy=xy, confidence=np.ones(21),
                                               actor="other_actor", track_id=3)])
    output = tmp_path / "result.h5"
    write_hdf5(output, [frame], source={"sha256": "fixture"}, model={"name": "fixture"},
               pose_topology="mediapipe_pose_33", pose_count=33)
    label = {"frame_index": 0, "hands": [{"actor": "other_actor", "bbox": [90, 90, 30, 30],
                                            "points": {"wrist": [100, 100]}}],
             "pose": {}, "face": {"visible": False}}
    result = score_frame(label, output)
    assert result["gt_other_hands"] == result["detected_other_hands"] == 1
    assert result["correct_actor"] == result["actor_scored_hands"] == 1
    assert result["hand_errors"] == [0.0]


def test_hand_detection_recall_is_separate_from_actor_assignment(tmp_path: Path) -> None:
    xy = np.tile([100.0, 100.0], (21, 1))
    output = tmp_path / "result.h5"
    write_hdf5(output, [Frame(timestamp_ms=0, hands=[Hand(xy=xy, confidence=np.ones(21),
                                                           actor="participant", track_id=3)])],
               source={"sha256": "fixture"}, model={"name": "fixture"},
               pose_topology="mediapipe_pose_33", pose_count=33)
    label = {"frame_index": 0, "hands": [{"actor": "other_actor", "handedness": "left",
                                           "bbox": [90, 90, 30, 30], "points": {}}],
             "pose": {}, "face": {"visible": False}}
    result = score_frame(label, output)
    assert result["detected_other_hands"] == 1
    assert result["correct_actor"] == 0
    assert result["track_matches"] == {"other_actor:left": 3}


def test_face_labels_use_anatomical_eye_sides(tmp_path: Path) -> None:
    # Frontal subject: their left eye appears on the right side of the image.
    face = np.full((478, 2), np.nan, dtype=np.float32)
    face[263], face[362] = [86, 40], [74, 40]
    face[33], face[133] = [24, 40], [36, 40]
    path = tmp_path / "face.h5"
    write_hdf5(path, [Frame(timestamp_ms=0, face_xy=face)],
               source={"sha256": "fixture"}, model={"name": "fixture"},
               pose_topology="mediapipe_pose_33", pose_count=33)
    label = {"frame_index": 0, "hands": [], "pose": {},
             "face": {"visible": True, "anchors": {
                 "left_eye": [80, 40], "right_eye": [30, 40]}}}
    assert score_frame(label, path)["face_errors"] == [0.0, 0.0]


def test_partially_visible_hand_matches_without_wrist_in_box() -> None:
    xy = np.tile([100.0, 100.0], (21, 1))
    xy[0] = [60, 60]
    predicted = [{"xy": xy, "valid": np.ones(21, dtype=bool)}]
    assert matching_hand({"bbox": [90, 90, 20, 20]}, predicted, set()) == 0


def test_sequence_tracking_and_temporal_residual(tmp_path: Path) -> None:
    pair_id = "X/P01-T"
    path = tmp_path / "results/mediapipe" / pair_id / "CAM_UL.h5"
    frames = []
    labels = []
    for index in range(3):
        pose = np.full((33, 2), np.nan, dtype=np.float32)
        pose[11], pose[12], pose[15] = [40, 40], [90, 40], [100 + index, 80]
        hand_xy = np.tile([100.0 + index, 100.0], (21, 1))
        frames.append(Frame(timestamp_ms=index * 33, pose_xy=pose,
                            hands=[Hand(xy=hand_xy, confidence=np.ones(21),
                                        actor="other_actor", handedness="left",
                                        track_id=1 if index < 2 else 2)]))
        labels.append({"id": str(index), "pair_id": pair_id, "view": "CAM_UL",
                       "frame_index": index, "reviewed": True, "hands_checked": True,
                       "body_checked": True, "face_checked": True,
                       "hands": [{"actor": "other_actor", "handedness": "left",
                                  "bbox": [90 + index, 90, 30, 30],
                                  "points": {"wrist": [100 + index, 100]}}],
                       "pose": {"left_shoulder": [40, 40], "right_shoulder": [90, 40],
                                "left_wrist": [100 + index, 80]},
                       "face": {"visible": False, "anchors": {}}})
    write_hdf5(path, frames, source={"sha256": "fixture"}, model={"name": "fixture"},
               pose_topology="mediapipe_pose_33", pose_count=33)
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"schema_version": 1, "frames": labels}))
    view = evaluate(annotations, tmp_path)["views"]["CAM_UL"]
    assert view["other_actor_recall"] == 1
    assert view["track_links"] == 2
    assert view["track_switches"] == 1
    assert view["median_pose_temporal_residual_by_shoulder_width"] == 0
    assert view["median_hand_temporal_residual_by_box_diagonal"] == 0


def test_hand_drafts_exclude_ambiguity_and_cannot_complete_decision(tmp_path: Path) -> None:
    pair_id = "X/P01-T"
    labels = []
    for view in ("CAM_UL", "CAM_UR"):
        path = tmp_path / "results/mediapipe" / pair_id / f"{view}.h5"
        write_hdf5(path, [Frame(timestamp_ms=i * 33) for i in range(4)],
                   source={"sha256": "fixture"}, model={"name": "fixture"},
                   pose_topology="mediapipe_pose_33", pose_count=33)
        for index in range(4):
            labels.append({"id": f"{view}:{index}", "pair_id": pair_id, "view": view,
                           "width": 200, "height": 200, "frame_index": index,
                           "reviewed": False, "hands_checked": index < 3,
                           "body_checked": False, "face_checked": False,
                           "hands": [{"actor": "other_actor", "bbox": [90, 90, 30, 30]}]
                                    if index in (0, 2) else [],
                           "hand_annotation_provenance": {
                               "status": "provisional_assistant_visual", "needs_review": index == 2}})
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"schema_version": 1, "frames": labels}))
    summary = evaluate_hands(annotations, tmp_path)
    assert summary["scored_frames"] == summary["provisional_frames"] == 4
    assert len(summary["excluded_frames"]) == 4
    for view in summary["views"].values():
        assert view["gt_other_hands"] == 1
        assert view["other_actor_recall"] == 0
        assert view["all_hands_absent_frames"] == 1
        assert view["other_actor_accuracy_matched"] is None
    assert not summary["decision_ready"] and not summary["labels_complete"]
    assert evaluate(annotations, tmp_path)["reviewed_frames"] == 0
    sensitivity = evaluate_hands(annotations, tmp_path, include_uncertain=True)
    assert sensitivity["scored_frames"] == 6
    assert sensitivity["views"]["CAM_UL"]["gt_other_hands"] == 2
    assert not sensitivity["decision_ready"]
    labels[0]["hands"][0]["bbox"] = [190, 90, 30, 30]
    annotations.write_text(json.dumps({"schema_version": 1, "frames": labels}))
    with pytest.raises(ValueError, match="Invalid hand box"):
        evaluate_hands(annotations, tmp_path)
