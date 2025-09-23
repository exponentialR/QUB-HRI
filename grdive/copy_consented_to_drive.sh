#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via flags below)
SRC_DEFAULT="..."
DST_DEFAULT="gdrive:/QUB-PHEO-DATASET"
DRY_RUN=1
IGNORE_EXISTING=0

usage() {
  cat <<EOF
Usage: $0 [-s SRC] [-d DST] [--real] [--ignore-existing]

Options:
  -s, --src PATH          Local source root (default: ${SRC_DEFAULT})
  -d, --dst REMOTE_PATH   Remote destination (default: ${DST_DEFAULT})
      --real              Do the real copy (default is dry-run)
      --ignore-existing   Skip files already present on Drive
  -h, --help              Show this help

Example:
  $0 --real
  $0 -s "/data/videos" -d "gdrive:/QUB-PHEO-DATASET" --ignore-existing --real
EOF
}

# Parse args
SRC="$SRC_DEFAULT"
DST="$DST_DEFAULT"
while [[ $# -gt 0 ]]; then
  case "$1" in
    -s|--src) SRC="$2"; shift 2;;
    -d|--dst) DST="$2"; shift 2;;
    --real) DRY_RUN=0; shift;;
    --ignore-existing) IGNORE_EXISTING=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# Check deps
command -v rclone >/dev/null 2>&1 || { echo "rclone not found. Install it first."; exit 1; }

# Participants (case-insensitive patterns generated)
read -r -d '' PARTICIPANTS <<'EOS'
P01 P02 P03 P04 P05 P06 P07 P08 P09
P11 P12 P14 P15 P17 P18 P19
P21 P22 P23 P24 P25 P26 P29 P30 P31 P32 P33
P35 P36 P37 P40 P44 P45 P46 P47 P48 P50
P52 P53 P54 P55 P57 P58 P60
P64 P65 P67 P68 P69 P70
EOS

# Extensions to include (tweak if you have more than mp4)
EXTS=("mp4")

# Build a temporary include file for rclone
INC_FILE="$(mktemp -t consented_include_XXXX.txt)"
trap 'rm -f "$INC_FILE"' EXIT

for pid in $PARTICIPANTS; do
  lower="${pid,,}"   # p01
  upper="${pid^^}"   # P01
  for ext in "${EXTS[@]}"; do
    echo "**/${lower}-*.${ext}" >> "$INC_FILE"
    echo "**/${upper}-*.${ext}" >> "$INC_FILE"
  done
done

echo "Include patterns written to: $INC_FILE"
echo "Source:      $SRC"
echo "Destination: $DST"
[[ $DRY_RUN -eq 1 ]] && echo "Mode:        DRY RUN (no data will be uploaded)"
[[ $DRY_RUN -eq 0 ]] && echo "Mode:        REAL COPY"
[[ $IGNORE_EXISTING -eq 1 ]] && echo "Option:      --ignore-existing enabled"

# Assemble rclone flags
RFLAGS=(copy "$SRC" "$DST"
  --include-from "$INC_FILE"
  --recursive
  --progress -v --stats=30s
  --transfers=12 --checkers=16
  --drive-chunk-size=256M
  --log-file=rclone_copy.log --log-level=INFO
)

[[ $DRY_RUN -eq 1 ]] && RFLAGS+=(--dry-run)
[[ $IGNORE_EXISTING -eq 1 ]] && RFLAGS+=(--ignore-existing)

echo
echo "Running: rclone ${RFLAGS[*]}"
echo
rclone "${RFLAGS[@]}"

echo
echo "Done."
[[ $DRY_RUN -eq 1 ]] && echo "This was a dry-run. Re-run with --real to actually upload."
