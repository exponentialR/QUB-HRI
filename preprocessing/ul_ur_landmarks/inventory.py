"""Read-only UL/UR inventory with explicit participant scope and optional pilot selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess


NAME = re.compile(r"^(P\d+)-(CAM_UL|CAM_UR)-(.+)$", re.IGNORECASE)
VIEWS = ("CAM_UL", "CAM_UR")


def consented_ids(path: Path) -> set[str]:
    ids = {line.strip().upper() for line in path.read_text().splitlines() if line.strip()}
    if not ids or any(not re.fullmatch(r"P\d+", pid) for pid in ids):
        raise ValueError(f"Invalid or empty participant list: {path}")
    return ids


def parse_clip(path: Path, root: Path, allowed: set[str] | None) -> tuple[str, str] | None:
    """Parse exact IDs and views; None explicitly includes all participant IDs."""
    if path.suffix.lower() != ".mp4":
        return None
    match = NAME.fullmatch(path.stem)
    if match is None:
        return None
    pid, view, rest = match.groups()
    pid, view = pid.upper(), view.upper()
    if allowed is not None and pid not in allowed:
        return None
    rel_dir = path.relative_to(root).parent.as_posix()
    return f"{rel_dir}/{pid}-{rest}", view


def discover(root: Path, allowed: set[str] | None) -> tuple[dict[str, dict[str, Path]], list[dict]]:
    pairs: dict[str, dict[str, Path]] = {}
    issues: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parsed = parse_clip(path, root, allowed)
        if parsed is None:
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            issues.append({"kind": "source_outside_root", "path": str(path.relative_to(root))})
            continue
        pair_id, view = parsed
        slot = pairs.setdefault(pair_id, {})
        if view in slot:
            issues.append({"kind": "duplicate_view", "pair_id": pair_id, "view": view,
                           "paths": [str(slot[view].relative_to(root)), str(path.relative_to(root))]})
        else:
            slot[view] = path
    for pair_id, slot in pairs.items():
        for view in VIEWS:
            if view not in slot:
                issues.append({"kind": "missing_view", "pair_id": pair_id, "view": view})
    return pairs, issues


def probe(path: Path) -> dict:
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format",
               "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate", "0/1")
    numerator, denominator = (int(n) for n in rate.split("/"))
    fps = numerator / denominator if denominator else 0.0
    duration = float(stream.get("duration") or data["format"].get("duration") or 0)
    frames = stream.get("nb_frames")
    return {"bytes": path.stat().st_size, "codec": stream.get("codec_name"),
            "width": int(stream["width"]), "height": int(stream["height"]),
            "fps": fps, "duration_s": duration,
            "declared_frames": int(frames) if frames and frames.isdigit() else None,
            "start_time_s": float(stream.get("start_time") or 0)}


def inventory(root: Path, ids_file: Path | None = None, workers: int = 8, *,
              all_participants: bool = False) -> tuple[list[dict], list[dict]]:
    if (ids_file is None) != all_participants:
        raise ValueError("Specify either a participant list or all_participants=True")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Input root does not exist: {root}")
    pairs, issues = discover(root, None if all_participants else consented_ids(ids_file))
    ambiguous = {issue["pair_id"] for issue in issues if issue["kind"] == "duplicate_view"}
    paths = sorted({path for slot in pairs.values() for path in slot.values()})
    metadata: dict[Path, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe, path): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                metadata[path] = future.result()
            except Exception as exc:
                issues.append({"kind": "unreadable", "path": str(path.relative_to(root)),
                               "error": str(exc)})
    rows = []
    for pair_id, slot in sorted(pairs.items()):
        if pair_id in ambiguous or set(slot) != set(VIEWS) or any(path not in metadata for path in slot.values()):
            continue
        ul, ur = (metadata[slot[view]] for view in VIEWS)
        differences = []
        for field in ("width", "height", "declared_frames"):
            if ul[field] != ur[field]:
                differences.append(field)
        if abs(ul["fps"] - ur["fps"]) > 1e-3:
            differences.append("fps")
        if abs(ul["duration_s"] - ur["duration_s"]) > (1 / max(ul["fps"], ur["fps"], 1)):
            differences.append("duration_s")
        if differences:
            issues.append({"kind": "metadata_mismatch", "pair_id": pair_id,
                           "fields": differences})
        rows.append({"pair_id": pair_id, "pid": pair_id.split("/")[-1].split("-")[0],
                     "subtask_dir": pair_id.rsplit("/", 1)[0],
                     "views": {view: {"relpath": str(slot[view].relative_to(root)),
                                      **metadata[slot[view]]} for view in VIEWS},
                     "metadata_match": not differences})
    return rows, sorted(issues, key=lambda item: (item["kind"], item.get("pair_id", item.get("path", ""))))


def select_pilot(rows: list[dict], count: int = 20) -> list[dict]:
    """Prefer 3–8 second clips for temporal review and spread them across strata."""
    candidates = [row for row in rows if row["metadata_match"] and
                  3 <= max(row["views"][v]["duration_s"] for v in VIEWS) <= 8]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} matched 3–8 second pairs; need {count}")
    candidates.sort(key=lambda row: (max(row["views"][v]["duration_s"] for v in VIEWS),
                                     row["pair_id"]))
    selected: list[dict] = []
    used_subtasks: set[str] = set()
    used_ids: set[str] = set()
    for _ in range(count):
        remaining = [row for row in candidates if row not in selected]
        if not remaining:
            break
        best = min(remaining, key=lambda row: (row["subtask_dir"] in used_subtasks,
                                                row["pid"] in used_ids,
                                                max(row["views"][v]["duration_s"] for v in VIEWS),
                                                row["pair_id"]))
        selected.append(best)
        used_subtasks.add(best["subtask_dir"])
        used_ids.add(best["pid"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--consented", type=Path, help="Original pilot: exact participant list")
    scope.add_argument("--all-participants", action="store_true",
                       help="Inventory all local UL/UR participant IDs")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pilot-pairs", type=int, default=20)
    parser.add_argument("--inventory-only", action="store_true",
                        help="Do not create or change a pilot selection")
    args = parser.parse_args()
    if args.workers < 1 or args.pilot_pairs < 1:
        parser.error("--workers and --pilot-pairs must be positive")
    if not args.inventory_only and args.pilot_pairs > 20:
        parser.error("Pilot selections are limited to 20 pairs; use --inventory-only for a full inventory")
    output_root = args.output_root.resolve()
    if output_root.is_relative_to(args.input_root.resolve()) or output_root.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Output root must be outside the source-video root and repository")
    scope_record = {"scope": "all_local_ul_ur" if args.all_participants else "participant_list",
                    "input_root": str(args.input_root.resolve()),
                    "participant_list": str(args.consented.resolve()) if args.consented else None}
    scope_path = output_root / "inventory_scope.json"
    if scope_path.exists() and json.loads(scope_path.read_text()) != scope_record:
        parser.error("Output directory belongs to a different inventory scope; use a new directory")
    if args.all_participants and not scope_path.exists() and (output_root / "inventory.json").exists():
        parser.error("Preserve the existing inventory; choose a new directory for the expanded scope")
    rows, issues = inventory(args.input_root, args.consented, args.workers,
                             all_participants=args.all_participants)
    selected = [] if args.inventory_only else select_pilot(rows, args.pilot_pairs)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, data in (("inventory.json", rows), ("inventory_issues.json", issues),
                       ("inventory_scope.json", scope_record)):
        (args.output_root / name).write_text(json.dumps(data, indent=2) + "\n")
    if args.inventory_only:
        print(json.dumps({**scope_record, "complete_probed_pairs": len(rows),
                          "issues": len(issues), "output_root": str(output_root)}))
        return
    selection_path = args.output_root / "pilot_pairs.json"
    if not selection_path.exists():
        selection_path.write_text(json.dumps(selected, indent=2) + "\n")
    else:
        selected = json.loads(selection_path.read_text())
        eligible = {row["pair_id"] for row in rows}
        if not isinstance(selected, list) or len(selected) > 20 or any(
                row.get("pair_id") not in eligible for row in selected):
            raise ValueError(f"Existing pilot selection is invalid; preserved: {selection_path}")
    print(json.dumps({"eligible_complete_pairs": len(rows), "issues": len(issues),
                      "provisional_pilot_pairs": len(selected),
                      "output_root": str(args.output_root.resolve())}))


if __name__ == "__main__":
    main()
