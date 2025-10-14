#!/usr/bin/env bash
set -euo pipefail

# =========================================
# QUB-PHEO consented subset -> Google Drive (rclone)
# - Audits local tree and writes include file for consented & present PIDs
# - Emits patterns only for actually observed extensions per PID
# - Copies with Drive-friendly pacing (stable, resumable)
# - Optional reconcile (copy only missing) and verify (size-only)
# - This script idempotently copy files from local to drive
#
# Defaults (override via flags)
SRC_DEFAULT="qub-pheo-consented-videos/segmented"
DST_DEFAULT="gdrive:/QUB-PHEO-DATASET"
EXTS_DEFAULT=(mp4 mov avi)                # matched case-insensitively via -iname
INC_OUT_DEFAULT="participants_include_present.txt"
LOG_FILE_DEFAULT="rclone_copy.log"

# Behaviour toggles
DRY_RUN=1
IGNORE_EXISTING=0
AUDIT_ONLY=0
RECONCILE_AFTER=1
VERIFY_AFTER=0
QUIET=0

# Pacing (safe defaults for Drive API)
TRANSFERS=4
CHECKERS=8
TPSLIMIT=8
TPSBURST=8
PACER_SLEEP="1s"
CHUNK_SIZE="256M"
RETRIES=10
RETRY_SLEEP="30s"

usage() {
  cat <<EOF
Usage: $0 [options]

Audit consented participants present in SRC, generate an rclone include file
(only observed extensions per PID), then copy to Google Drive with safe pacing.
Optionally reconcile missing files and verify sizes.

Options:
  -s, --src PATH              Local source root (default: ${SRC_DEFAULT})
  -d, --dst REMOTE            Remote destination (default: ${DST_DEFAULT})
  -e, --exts LIST             Comma/space-separated extensions (default: ${EXTS_DEFAULT[*]})
  -o, --inc-out FILE          Include file path (default: ${INC_OUT_DEFAULT})
  -l, --log FILE              rclone log file (default: ${LOG_FILE_DEFAULT})

      --real                  Do the real copy (default is dry-run)
      --ignore-existing       Skip files already present on Drive
      --audit-only            Only audit + write include file; do not copy
      --no-reconcile          Skip the "copy only missing files" pass
      --verify                After copy, run size-only verification

  # Pacing (advanced)
      --transfers N           Concurrent transfers   (default: ${TRANSFERS})
      --checkers N            Concurrent checkers    (default: ${CHECKERS})
      --tpslimit N            API calls/sec cap      (default: ${TPSLIMIT})
      --tpsburst N            Burst size             (default: ${TPSBURST})
      --pacer-sleep DUR       Min pacer sleep        (default: ${PACER_SLEEP})
      --chunk SIZE            Drive chunk size       (default: ${CHUNK_SIZE})
      --retries N             Retries on failure     (default: ${RETRIES})
      --retry-sleep DUR       Sleep between retries  (default: ${RETRY_SLEEP})

  -q, --quiet                 Minimal audit output
  -h, --help                  Show this help

Examples:
  $0                               # Dry-run: audit + plan
  $0 --real --ignore-existing      # Real copy, idempotent top-up
  $0 -e mp4,mov --real --verify    # Restrict to mp4/mov and verify afterwards
EOF
}

# ---------- Parse args ----------
SRC="$SRC_DEFAULT"; DST="$DST_DEFAULT"
EXTS=("${EXTS_DEFAULT[@]}")
INC_OUT="$INC_OUT_DEFAULT"
LOG_FILE="$LOG_FILE_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--src) SRC="$2"; shift 2;;
    -d|--dst) DST="$2"; shift 2;;
    -e|--exts) shift; IFS=', ' read -r -a EXTS <<< "${1:-}"; shift || true;;
    -o|--inc-out) INC_OUT="$2"; shift 2;;
    -l|--log) LOG_FILE="$2"; shift 2;;

    --real) DRY_RUN=0; shift;;
    --ignore-existing) IGNORE_EXISTING=1; shift;;
    --audit-only) AUDIT_ONLY=1; shift;;
    --no-reconcile) RECONCILE_AFTER=0; shift;;
    --verify) VERIFY_AFTER=1; shift;;

    --transfers) TRANSFERS="$2"; shift 2;;
    --checkers) CHECKERS="$2"; shift 2;;
    --tpslimit) TPSLIMIT="$2"; shift 2;;
    --tpsburst) TPSBURST="$2"; shift 2;;
    --pacer-sleep) PACER_SLEEP="$2"; shift 2;;
    --chunk) CHUNK_SIZE="$2"; shift 2;;
    --retries) RETRIES="$2"; shift 2;;
    --retry-sleep) RETRY_SLEEP="$2"; shift 2;;

    -q|--quiet) QUIET=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# ---------- Deps ----------
command -v rclone >/dev/null 2>&1 || { echo "rclone not found"; exit 1; }
command -v find   >/dev/null 2>&1 || { echo "find not found"; exit 1; }
command -v sed    >/dev/null 2>&1 || { echo "sed not found"; exit 1; }
command -v tr     >/dev/null 2>&1 || { echo "tr not found"; exit 1; }
command -v sort   >/dev/null 2>&1 || { echo "sort not found"; exit 1; }
command -v comm   >/dev/null 2>&1 || { echo "comm not found"; exit 1; }

# ---------- Consented (CRLF-safe, normalised) ----------
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

# ---------- Build find predicates ----------
ext_globs=()
for ext in "${EXTS[@]}"; do
  ext_globs+=( -o -iname "p[0-9][0-9]-*.${ext}" )
done
ext_globs=( "${ext_globs[@]:1}" )

# ---------- Audit ----------
if [[ $QUIET -eq 0 ]]; then
  echo "Auditing source for consented participants:"
  echo "  Source:      $SRC"
  echo "  Destination: $DST"
  echo "  Extensions:  ${EXTS[*]}"
  echo "  Include out: $INC_OUT"
  echo "  Consented IDs parsed (${#CONSENTED_ARR[@]}):"
  printf '    %s\n' "${CONSENTED_ARR[@]}"
  echo
fi

mapfile -t FILENAMES < <(find "$SRC" -type f \( "${ext_globs[@]}" \) -printf '%f\n' || true)
if [[ ${#FILENAMES[@]} -eq 0 ]]; then
  echo "No matching files under: $SRC"
  exit 0
fi

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

# ---------- Include file: consented & present, only seen extensions ----------
: > "$INC_OUT"
while read -r pid; do
  [[ -z "${pid:-}" ]] && continue
  lower="${pid,,}"; upper="${pid^^}"
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

# ---------- Stop here if audit-only ----------
if [[ $AUDIT_ONLY -eq 1 ]]; then
  [[ $QUIET -eq 0 ]] && echo "Audit-only requested; skipping copy."
  exit 0
fi

# ---------- Main copy (paced) ----------
RFLAGS=( copy "$SRC" "$DST"
  --include-from "$INC_OUT"
  -P --stats=30s --fast-list
  --transfers="$TRANSFERS" --checkers="$CHECKERS"
  --tpslimit="$TPSLIMIT" --tpslimit-burst="$TPSBURST" --drive-pacer-min-sleep="$PACER_SLEEP"
  --drive-chunk-size="$CHUNK_SIZE"
  --retries="$RETRIES" --retries-sleep="$RETRY_SLEEP"
  --log-file="$LOG_FILE" --log-level=INFO
)
[[ $DRY_RUN -eq 1 ]] && RFLAGS+=(--dry-run)
[[ $IGNORE_EXISTING -eq 1 ]] && RFLAGS+=(--ignore-existing)

echo "Running: rclone ${RFLAGS[*]}"
rclone "${RFLAGS[@]}"

# ---------- Reconcile: copy only files still missing ----------
if [[ $RECONCILE_AFTER -eq 1 ]]; then
  echo "Reconciling (compute local vs remote set difference)…"
  rclone lsf -R "$SRC" --include-from "$INC_OUT" | sort > /tmp/local.lsf
  rclone lsf -R "$DST" --include-from "$INC_OUT" | sort > /tmp/remote.lsf || true
  comm -23 /tmp/local.lsf /tmp/remote.lsf > /tmp/missing.list || true

  if [[ -s /tmp/missing.list ]]; then
    echo "Copying $(wc -l < /tmp/missing.list) missing files…"
    rclone copy "$SRC" "$DST" \
      --files-from-raw /tmp/missing.list \
      -P --stats=30s --fast-list \
      --transfers="$TRANSFERS" --checkers="$CHECKERS" \
      --tpslimit="$TPSLIMIT" --tpslimit-burst="$TPSBURST" --drive-pacer-min-sleep="$PACER_SLEEP" \
      --drive-chunk-size="$CHUNK_SIZE" \
      --retries="$RETRIES" --retries-sleep="$RETRY_SLEEP" \
      --log-file="$LOG_FILE" --log-level=INFO
  else
    echo "No missing files detected on Drive."
  fi
fi

# ---------- Verify (optional, size-only fast pass) ----------
if [[ $VERIFY_AFTER -eq 1 ]]; then
  echo "Verifying (one-way, size-only)…"
  rclone check "$SRC" "$DST" \
    --include-from "$INC_OUT" \
    --one-way --size-only -P \
    --log-file="$LOG_FILE" --log-level=INFO || true
fi

echo "Done."
[[ $DRY_RUN -eq 1 ]] && echo "This was a dry-run. Re-run with --real to upload."
# =========================================
