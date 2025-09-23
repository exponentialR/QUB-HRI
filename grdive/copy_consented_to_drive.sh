#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Copy only consented & PRESENT participant videos to Google Drive,
# preserving folder structure, using rclone.
#
# The script first AUDITS the source tree to find which consented IDs
# are actually present (by filename prefix 'pXX-*.EXT'), writes an
# include file (case-insensitive patterns), then optionally runs rclone.
#
# Defaults (override via flags below)
SRC_DEFAULT="/home/alien_arise/Documents/qub-pheo-consented-videos/segmented"
DST_DEFAULT="gdrive:/QUB-PHEO-DATASET"
EXTS_DEFAULT=("mp4")            # add more via -e "mp4,mov,mkv"
INC_OUT_DEFAULT="participants_include_present.txt"
DRY_RUN=1                       # default: dry-run
IGNORE_EXISTING=0               # default: re-upload if differs
AUDIT_ONLY=0                    # default: run copy after audit
QUIET=0                         # default: verbose audit output
LOG_FILE="rclone_copy.log"

usage() {
  cat <<EOF
Usage: $0 [options]

Audit the source tree for consented participants that are PRESENT, generate an
rclone --include-from file, and (unless --audit-only) copy matching files to Drive.

Options:
  -s, --src PATH           Local source root (default: ${SRC_DEFAULT})
  -d, --dst REMOTE_PATH    Remote destination (default: ${DST_DEFAULT})
  -e, --exts LIST          Comma/space-separated extensions (default: ${EXTS_DEFAULT[*]})
                           e.g. -e mp4,mov or -e "mp4 mkv"
  -o, --inc-out FILE       Where to write the include file
                           (default: ${INC_OUT_DEFAULT})
      --real               Do the real copy (default is dry-run)
      --ignore-existing    Skip files already present on Drive
      --audit-only         Only audit + write include file; do not copy
  -q, --quiet              Minimal audit output
  -h, --help               Show this help

Examples:
  # Dry-run with defaults (audit + show planned copy)
  $0

  # Real copy with defaults
  $0 --real

  # Add extensions, skip already-uploaded on reruns
  $0 -e mp4,mov --real --ignore-existing

  # Only audit and write include file
  $0 --audit-only -o /tmp/include.txt

Notes:
- Filenames are assumed to start with participant ID like p01-*.EXT (case-insensitive).
- Folder structure is preserved on Google Drive.
- Safe to rerun; consider --ignore-existing for idempotent top-ups.
EOF
}

# ---- Parse args
SRC="$SRC_DEFAULT"
DST="$DST_DEFAULT"
EXTS=("${EXTS_DEFAULT[@]}")
INC_OUT="$INC_OUT_DEFAULT"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--src) SRC="$2"; shift 2;;
    -d|--dst) DST="$2"; shift 2;;
    -e|--exts)
      shift
      IFS=', ' read -r -a EXTS <<< "${1:-}"
      [[ ${#EXTS[@]} -eq 0 || -z "${EXTS[0]:-}" ]] && EXTS=("${EXTS_DEFAULT[@]}")
      shift || true
      ;;
    -o|--inc-out) INC_OUT="$2"; shift 2;;
    --real) DRY_RUN=0; shift;;
    --ignore-existing) IGNORE_EXISTING=1; shift;;
    --audit-only) AUDIT_ONLY=1; shift;;
    -q|--quiet) QUIET=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# ---- Check deps
command -v rclone >/dev/null 2>&1 || { echo "rclone not found. Install it first."; exit 1; }
command -v find   >/dev/null 2>&1 || { echo "find not found."; exit 1; }
command -v sed    >/dev/null 2>&1 || { echo "sed not found."; exit 1; }
command -v tr     >/dev/null 2>&1 || { echo "tr not found."; exit 1; }
command -v sort   >/dev/null 2>&1 || { echo "sort not found."; exit 1; }
command -v uniq   >/dev/null 2>&1 || { echo "uniq not found."; exit 1; }
command -v awk    >/dev/null 2>&1 || { echo "awk not found."; exit 1; }
command -v column >/dev/null 2>&1 || true   # nice-to-have only

# ---- Consented list (normalised to uppercase PXX)
read -r -d '' CONSENTED_STR <<'EOS' || true
P01 P02 P03 P04 P05 P06 P07 P08 P09
P11 P12 P14 P15 P17 P18 P19
P21 P22 P23 P24 P25 P26 P29 P30 P31 P32 P33
P35 P36 P37 P40 P44 P45 P46 P47 P48 P50
P52 P53 P54 P55 P57 P58 P60
P64 P65 P67 P68 P69 P70
EOS

# ---- Build find predicates for extensions (case-insensitive)
ext_globs=()
for ext in "${EXTS[@]}"; do
  ext_globs+=( -o -iname "p[0-9][0-9]-*.${ext}" )
done
ext_globs=( "${ext_globs[@]:1}" )  # drop leading -o

# ---- Scan filenames under SRC that match PXX-*.EXT
if [[ $QUIET -eq 0 ]]; then
  echo "Auditing source for consented participants:"
  echo "  Source:      $SRC"
  echo "  Extensions:  ${EXTS[*]}"
  echo "  Include out: $INC_OUT"
  echo
fi

mapfile -t FILENAMES < <(find "$SRC" -type f \( "${ext_globs[@]}" \) -printf '%f\n' || true)

if [[ ${#FILENAMES[@]} -eq 0 ]]; then
  echo "No matching files found under: $SRC"
  echo "Nothing to include. Exiting."
  exit 0
fi

# ---- Extract PIDs and count
declare -A COUNTS=()
declare -A FOUND=()

for f in "${FILENAMES[@]}"; do
  pid=$(sed -E 's/^([pP][0-9]{2})-.*/\1/' <<<"$f" | tr '[:lower:]' '[:upper:]')
  if [[ "$pid" =~ ^P[0-9]{2}$ ]]; then
    FOUND["$pid"]=1
    ((COUNTS["$pid"]++)) || true
  fi
done

read -ra CONSENTED <<< "$CONSENTED_STR"
FOUND_LIST=$(printf "%s\n" "${!FOUND[@]}" | sort -V)
CONSENTED_LIST=$(printf "%s\n" "${CONSENTED[@]}" | sort -V)

tmp_found=$(mktemp); tmp_consent=$(mktemp)
trap 'rm -f "$tmp_found" "$tmp_consent"' EXIT
printf "%s\n" $FOUND_LIST     | sort -V > "$tmp_found"
printf "%s\n" $CONSENTED_LIST | sort -V > "$tmp_consent"

# ---- Audit reports
if [[ $QUIET -eq 0 ]]; then
  echo "============ Counts per participant (FOUND) ============"
  for pid in $(printf "%s\n" "${!COUNTS[@]}" | sort -V); do
    printf "%s\t%s\n" "$pid" "${COUNTS[$pid]}"
  done | (command -v column >/dev/null 2>&1 && column -t || cat)
  echo

  echo "============ Consented & PRESENT ============"
  comm -12 "$tmp_consent" "$tmp_found" | sed 's/^/  /' || true
  echo

  echo "============ Consented but MISSING ============"
  comm -23 "$tmp_consent" "$tmp_found" | sed 's/^/  /' || true
  echo

  echo "============ PRESENT but UNCONSENTED ============"
  comm -13 "$tmp_consent" "$tmp_found" | sed 's/^/  /' || true
  echo
fi

# ---- Build include file ONLY for consented & present (case-insensitive patterns)
: > "$INC_OUT"
while read -r pid; do
  [[ -z "${pid:-}" ]] && continue
  lower="${pid,,}"
  upper="${pid^^}"
  for ext in "${EXTS[@]}"; do
    echo "**/${lower}-*.${ext}" >> "$INC_OUT"
    echo "**/${upper}-*.${ext}" >> "$INC_OUT"
  done
done < <(comm -12 "$tmp_consent" "$tmp_found")

if [[ ! -s "$INC_OUT" ]]; then
  echo "No consented participants found in source. Nothing to copy."
  exit 0
fi

[[ $QUIET -eq 0 ]] && echo "Include patterns written to: $INC_OUT"

# ---- Stop here if audit-only
if [[ $AUDIT_ONLY -eq 1 ]]; then
  [[ $QUIET -eq 0 ]] && echo "Audit-only requested; skipping copy."
  exit 0
fi

# ---- Assemble rclone flags
[[ $QUIET -eq 0 ]] && {
  echo "Source:      $SRC"
  echo "Destination: $DST"
  [[ $DRY_RUN -eq 1 ]] && echo "Mode:        DRY RUN (no data will be uploaded)"
  [[ $DRY_RUN -eq 0 ]] && echo "Mode:        REAL COPY"
  [[ $IGNORE_EXISTING -eq 1 ]] && echo "Option:      --ignore-existing enabled"
  echo
}

RFLAGS=(copy "$SRC" "$DST"
  --include-from "$INC_OUT"
  --recursive
  --progress -v --stats=30s
  --transfers=12 --checkers=16
  --drive-chunk-size=256M
  --log-file="$LOG_FILE" --log-level=INFO
)

[[ $DRY_RUN -eq 1 ]] && RFLAGS+=(--dry-run)
[[ $IGNORE_EXISTING -eq 1 ]] && RFLAGS+=(--ignore-existing)

echo "Running: rclone ${RFLAGS[*]}"
rclone "${RFLAGS[@]}"

echo
echo "Done."
[[ $DRY_RUN -eq 1 ]] && echo "This was a dry-run. Re-run with --real to actually upload."
# =========================================
