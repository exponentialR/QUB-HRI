#!/usr/bin/env bash
set -euo pipefail
SRC="/qub-pheo-consented-videos/segmented"
DST="gdrive:/QUB-PHEO-DATASET"
LIST="./missing.list"

[[ -s "$LIST" ]] || { echo "No missing.list found or it is empty."; exit 0; }

rclone copy "$SRC" "$DST" \
  --files-from-raw "$LIST" \
  -P --stats=30s --fast-list \
  --transfers=4 --checkers=8 \
  --tpslimit=8 --tpslimit-burst=8 --drive-pacer-min-sleep=1s \
  --drive-chunk-size=256M \
  --retries=10 --retries-sleep=30s \
  --log-file="rclone_missing_copy.log" --log-level=INFO
