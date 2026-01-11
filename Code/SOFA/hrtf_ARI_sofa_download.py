#!/usr/bin/env bash
set -euo pipefail

DEST="/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/SOFA"
BASE="https://sofacoustics.org/data/database/ari"
LIST_URL="${BASE}/list.txt"

mkdir -p "$DEST"
cd "$DEST"

echo "Downloading list: $LIST_URL"
curl -fsSL "$LIST_URL" -o list.txt

# Extract all ".sofa" entries even if list.txt is a single long line.
# If entries are relative (e.g., dtf/b_nh10.sofa), prefix with $BASE/.
# If they are already absolute URLs (https://...), keep as-is.
echo "Preparing URLs..."
tr -s '[:space:]' '\n' < list.txt \
  | sed '/^$/d' \
  | grep -E '\.sofa$' \
  | awk -v base="$BASE" '
      /^https?:\/\// { print; next }
      { print base "/" $0 }
    ' > sofa_urls.txt

echo "Total SOFA files to download: $(wc -l < sofa_urls.txt)"

# Download in parallel (tune -P to your network/FS). Use -c to resume.
# Files are saved with their original filename (-O).
cat sofa_urls.txt | xargs -n 1 -P 8 -I {} bash -c '
  url="{}"
  echo "GET $url"
  curl -fL -C - -O "$url"
'

echo "Done. Files saved under: $DEST"
