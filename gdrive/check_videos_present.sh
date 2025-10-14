SRC="qub-pheo-consented-videos/segmented"

# 1) Distinct participant IDs detected (normalised to uppercase: P01..P70)
find "$SRC" -type f -iname 'p[0-9][0-9]-*.mp4' -printf '%f\n' \
| sed -E 's/^([pP][0-9]{2})-.*/\1/' | tr '[:lower:]' '[:upper:]' \
| sort -u

# 2) Counts per participant (how many files for each PXX)
find "$SRC" -type f -iname 'p[0-9][0-9]-*.mp4' -printf '%f\n' \
| sed -E 's/^([pP][0-9]{2})-.*/\1/' | tr '[:lower:]' '[:upper:]' \
| sort | uniq -c \
| awk '{printf "%s\t%s\n", $2, $1}' | sort -V | column -t
