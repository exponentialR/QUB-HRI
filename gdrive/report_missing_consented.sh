#!/usr/bin/env bash
set -euo pipefail

# Report consented participants/files NOT yet on Google Drive.

SRC_DEFAULT="/qub-pheo-consented-videos/segmented"
DST_DEFAULT="gdrive:/QUB-PHEO-DATASET"
EXTS_DEFAULT=(mp4 mov avi)          # case-insensitive

usage() {
  cat <<EOF
Usage: $0 [options]
  -s, --src PATH     Local source root (default: ${SRC_DEFAULT})
  -d, --dst REMOTE   Remote destination (default: ${DST_DEFAULT})
  -e, --exts LIST    Comma/space-separated extensions (default: ${EXTS_DEFAULT[*]})
  -o, --out FILE     Write list of missing files to FILE (relative paths)
  -q, --quiet        Suppress intermediate messages
  -h, --help         Show this help

Exit codes: 0 = nothing missing; 2 = some files missing
EOF
}

# ---- Parse args
SRC="$SRC_DEFAULT"; DST="$DST_DEFAULT"
EXTS=("${EXTS_DEFAULT[@]}"); QUIET=0; OUT_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--src) SRC="$2"; shift 2;;
    -d|--dst) DST="$2"; shift 2;;
    -e|--exts) shift; IFS=', ' read -r -a EXTS <<< "${1:-}"; shift || true;;
    -o|--out) OUT_FILE="$2"; shift 2;;
    -q|--quiet) QUIET=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# ---- Deps
for bin in rclone find sed tr sort comm awk; do
  command -v "$bin" >/dev/null 2>&1 || { echo "Missing dependency: $bin"; exit 1; }
done

# ---- Consented list (normalised)
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
  | tr -d '\r' | tr '[:lower:]' '[:upper:]' \
  | tr ' \t' '\n' | grep -E '^P[0-9]{2}$' | sort -V | uniq
)
readarray -t CONSENTED_ARR <<< "$CONSENTED_LIST_NORM"

# ---- Build find predicates for extensions
ext_globs=()
for ext in "${EXTS[@]}"; do
  ext_globs+=( -o -iname "p[0-9][0-9]-*.${ext}" )
done
ext_globs=( "${ext_globs[@]:1}" )

# ---- Scan local to know which PIDs/exts exist
[[ $QUIET -eq 1 ]] || {
  echo "Scanning local:"
  echo "  SRC:   $SRC"
  echo "  DST:   $DST"
  echo "  EXTS:  ${EXTS[*]}"
  echo
}

mapfile -t FILENAMES < <(find "$SRC" -type f \( "${ext_globs[@]}" \) -printf '%f\n' || true)
if [[ ${#FILENAMES[@]} -eq 0 ]]; then
  echo "No matching local files in: $SRC"
  exit 0
fi

declare -A SEEN_PID_EXT=()  # key "P11.mp4" -> 1
declare -A FOUND_PIDS=()
for f in "${FILENAMES[@]}"; do
  pid=$(sed -E 's/^([pP][0-9]{2})-.*/\1/' <<<"$f" | tr '[:lower:]' '[:upper:]')
  [[ "$pid" =~ ^P[0-9]{2}$ ]] || continue
  ext="${f##*.}"; ext="${ext,,}"
  FOUND_PIDS["$pid"]=1
  SEEN_PID_EXT["$pid.$ext"]=1
done

# Consent ∩ Found
tmp_found=$(mktemp); tmp_consent=$(mktemp)
trap 'rm -f "$tmp_found" "$tmp_consent" "$inc" /tmp/local.lsf /tmp/remote.lsf /tmp/missing.list /tmp/*.csv 2>/dev/null || true' EXIT
printf "%s\n" "${!FOUND_PIDS[@]}" | sort -V > "$tmp_found"
printf "%s\n" "${CONSENTED_ARR[@]}" | sort -V > "$tmp_consent"

# ---- Build a temp include file only for observed (PID,ext)
inc="$(mktemp)"
while read -r pid; do
  [[ -z "${pid:-}" ]] && continue
  lower="${pid,,}"; upper="${pid^^}"
  for ext in "${EXTS[@]}"; do
    ext_lc="${ext,,}"
    [[ -n "${SEEN_PID_EXT["$pid.$ext_lc"]+x}" ]] || continue
    echo "**/${lower}-*.${ext_lc}" >> "$inc"
    echo "**/${upper}-*.${ext_lc}" >> "$inc"
  done
done < <(comm -12 "$tmp_consent" "$tmp_found")

if [[ ! -s "$inc" ]]; then
  echo "No consented participants detected locally. Nothing to compare."
  exit 0
fi

# ---- Enumerate local and remote sets (relative paths)
rclone lsf -R "$SRC" --include-from "$inc" | sort > /tmp/local.lsf
rclone lsf -R "$DST" --include-from "$inc" | sort > /tmp/remote.lsf || true

# ---- Compute missing files (present locally but not on Drive)
comm -23 /tmp/local.lsf /tmp/remote.lsf > /tmp/missing.list || true
[[ -n "$OUT_FILE" ]] && cp /tmp/missing.list "$OUT_FILE"

# ---- Helpers to extract PID from path safely (no awk function needed)
extract_pid() {
  # get filename, pull pXX-, uppercase; drop lines without match
  sed -E 's#.*/##; t; d' \
  | sed -E 's/^([pP][0-9]{2})-.*/\1/; t; d' \
  | tr '[:lower:]' '[:upper:]'
}

# ---- Build per-PID counts (local, remote, missing)
extract_pid < /tmp/local.lsf   | sort | uniq -c | awk '{print $2","$1}' > /tmp/local_counts.csv
extract_pid < /tmp/remote.lsf  | sort | uniq -c | awk '{print $2","$1}' > /tmp/remote_counts.csv
extract_pid < /tmp/missing.list | sort | uniq -c | awk '{print $2","$1}' > /tmp/missing_counts.csv || true

# Join three CSVs by PID
join -t, -a1 -e0 -o 1.1,1.2,2.2 <(sort -t, -k1,1 /tmp/local_counts.csv) <(sort -t, -k1,1 /tmp/remote_counts.csv) \
| join -t, -a1 -e0 -o 1.1,1.2,1.3,2.2 - <(sort -t, -k1,1 /tmp/missing_counts.csv) \
> /tmp/joined_counts.csv || true

# ---- Print report
total_local=$(wc -l < /tmp/local.lsf | tr -d ' ')
total_remote=$(wc -l < /tmp/remote.lsf | tr -d ' ')
total_missing=$(wc -l < /tmp/missing.list | tr -d ' ')

echo "===== QUB-PHEO consented sync status ====="
echo "Source:   $SRC"
echo "Dest:     $DST"
echo "Exts:     ${EXTS[*]}"
echo
printf "Files (consented subset): local=%d, remote=%d, missing=%d\n" "$total_local" "$total_remote" "$total_missing"
echo

printf "%-5s %10s %10s %10s\n" "PID" "LOCAL" "REMOTE" "MISSING"
printf "%-5s %10s %10s %10s\n" "-----" "--------" "--------" "--------"
sort -t, -k1,1 /tmp/joined_counts.csv \
| awk -F, '{printf "%-5s %10d %10d %10d\n", $1, $2, $3, $4}'

echo
if [[ -s /tmp/missing.list ]]; then
  echo "Participants with ANY missing files:"
  awk -F, '$4>0 {print "  " $1}' /tmp/joined_counts.csv | sort -V
  echo
  [[ -n "$OUT_FILE" ]] && echo "Missing file list written to: $OUT_FILE"
  exit 2
else
  echo "✅ Drive is complete for the consented subset (no missing files)."
  exit 0
fi
