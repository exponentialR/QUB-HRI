#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Copy only consented & PRESENT participant videos to Google Drive,
# preserving folder structure, using rclone.
#
# Flow:
# 1) Audit the source tree for consented IDs present (filenames 'pXX-*.EXT').
# 2) Build an include file containing ONLY (PID, EXT) pairs actually observed.
# 3) Optionally rclone copy (dry-run by default).
#
# Defaults (override via flags below)
SRC_DEFAULT="/qub-pheo-consented-videos/segmented"
DST_DEFAULT="gdrive:/QUB-PHEO-DATASET"
EXTS_DEFAULT=(mp4 mov avi)       # -iname is case-insensitive
INC_OUT_DEFAULT="participants_include_present.txt"
DRY_RUN=1
IGNORE_EXISTING=0
AUDIT_ONLY=0
QUIET=0
LOG_FILE="rclone_copy.log"

usage() {
  cat <<EOF
Usage: $0 [options]

Audit the source tree for consented participants that are PRESENT, generate an
rclone --include-from file (only observed extensions per PID), and (unless
--audit-only) copy matching files to Drive.

Options:
  -s, --src PATH           Local source root (default: ${SRC_DEFAULT})
  -d, --dst REMOTE_PATH    Remote destination (default: ${DST_DEFAULT})
  -e, --exts LIST          Comma/space-separated extensions (default: ${EXTS_DEFAULT[*]})
                           e.g. -e mp4,mov or -e "mp4 mkv"
  -o, --inc-out FILE       Where to write the include file (default: ${INC_OUT_DEFAULT})
      --real               Do the real copy (default is dry-run)
      --ignore-existing    Skip files already present on Drive
      --audit-only         Only audit + write include file; do not copy
  -q, --quiet              Minimal audit output
  -h, --help               Show this help

Examples:
  $0                         # Dry-run (audit + planned copy)
  $0 --real                  # Real copy
  $0 -e mp4,mov --real --ignore-existing
  $0 --audit-only -o /tmp/include.txt
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

# ---- Consented list (CRLF-safe, normalised)
read -r -d '' CONSENTED_STR_RAW <<'EOS' || true
P01 P02 P03 P04 P05 P06 P07 P08 P09
P11 P12 P14 P15 P17 P18 P19
P21 P22 P23 P24 P25 P26 P29 P30 P31 P32 P33
P35 P36 P37 P40 P44 P45 P46 P47 P48 P50
P52 P53 P54 P55 P57 P58 P60
P64 P65 P67 P68 P69 P70
EOS

CONSENTED_LIST_NORM=$(
  printf '%s\n' "$CONSENTED_STR_RAW" \
  | tr -d '\r' \
  | tr '[:lower:]' '[:upper:]' \
  | tr ' \t' '\n' \
  | grep -E '^P[0-9]{2}$' \
  | sort -V | uniq
)
readarray -t CONSENTED_ARR <<< "$CONSENTED_LIST_NORM"

# ---- Build find predicates for extensions (case-insensitive via -iname)
ext_globs=()
for ext in "${EXTS[@]}"; do
  ext_globs+=( -o -iname "p[0-9][0-9]-*.${ext}" )
done
ext_globs=( "${ext_globs[@]:1}" )

# ---- Audit header
if [[ $QUIET -eq 0 ]]; then
  echo "Auditing source for consented participants:"
  echo "  Source:      $SRC"
  echo "  Extensions:  ${EXTS[*]}"
  echo "  Include out: $INC_OUT"
  echo "  Consented IDs parsed (${#CONSENTED_ARR[@]}):"
  printf '    %s\n' "${CONSENTED_ARR[@]}"
  echo
fi

# ---- Scan matching filenames (names only)
mapfile -t FILENAMES < <(find "$SRC" -type f \( "${ext_globs[@]}" \) -printf '%f\n' || true)
if [[ ${#FILENAMES[@]} -eq 0 ]]; then
  echo "No matching files found under: $SRC"
  echo "Nothing to include. Exiting."
  exit 0
fi

# ---- Extract PIDs, counts, and per-PID observed extensions
declare -A COUNTS=()
declare -A FOUND=()
declare -A SEEN_PID_EXT=()   # key "P11.mp4" -> 1

for f in "${FILENAMES[@]}"; do
  pid=$(sed -E 's/^([pP][0-9]{2})-.*/\1/' <<<"$f" | tr '[:lower:]' '[:upper:]')
  [[ "$pid" =~ ^P[0-9]{2}$ ]] || continue
  ext="${f##*.}"; ext="${ext,,}"

  FOUND["$pid"]=1
  ((COUNTS["$pid"]++)) || true
  SEEN_PID_EXT["$pid.$ext"]=1
done

FOUND_LIST=$(printf "%s\n" "${!FOUND[@]}" | sort -V | uniq)
CONSENTED_LIST=$(printf "%s\n" "${CONSENTED_ARR[@]}" | sort -V | uniq)

# ---- Set math with comm (quote to preserve newlines)
tmp_found=$(mktemp); tmp_consent=$(mktemp)
trap 'rm -f "$tmp_found" "$tmp_consent"' EXIT
printf "%s\n" "$FOUND_LIST"     | sort -V > "$tmp_found"
printf "%s\n" "$CONSENTED_LIST" | sort -V > "$tmp_consent"

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

# ---- Build include file ONLY for consented & present, and ONLY seen extensions
: > "$INC_OUT"
while read -r pid; do
  [[ -z "${pid:-}" ]] && continue
  lower="${pid,,}"
  upper="${pid^^}"

  # Emit patterns only for (PID, ext) pairs actually observed
  for ext in "${EXTS[@]}"; do
    ext_lc="${ext,,}"
    if [[ -n "${SEEN_PID_EXT["$pid.$ext_lc"]+x}" ]]; then
      echo "**/${lower}-*.${ext_lc}" >> "$INC_OUT"
      echo "**/${upper}-*.${ext_lc}" >> "$INC_OUT"
    fi
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

# ---- rclone copy (recursive by default)
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
  -P --stats=30s
  --fast-list
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
