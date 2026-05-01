#!/bin/bash
# Lazy-download SAM2.1 large checkpoint (~898 MB) on first boot.
# Skipped if the file already exists (volume-mounted across pod restarts).

set -e

CHECKPOINT_PATH="${SAM2_CHECKPOINT:-/app/models/sam2/sam2.1_hiera_large.pt}"
CHECKPOINT_URL="${SAM2_CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt}"

CHECKPOINT_DIR="$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$CHECKPOINT_DIR"

if [ -f "$CHECKPOINT_PATH" ]; then
    SIZE_MB=$(du -m "$CHECKPOINT_PATH" 2>/dev/null | cut -f1)
    echo "[sam2] Checkpoint already present at $CHECKPOINT_PATH (${SIZE_MB} MB), skipping download."
    exit 0
fi

echo "[sam2] Downloading $CHECKPOINT_URL"
echo "[sam2] -> $CHECKPOINT_PATH"

# --fail: error on HTTP >=400; --location: follow redirects;
# --retry 3: resilience to flaky CDNs; --continue-at -: resume on partial.
curl --fail --location --retry 3 --continue-at - \
     --output "$CHECKPOINT_PATH" \
     "$CHECKPOINT_URL"

SIZE_MB=$(du -m "$CHECKPOINT_PATH" | cut -f1)
echo "[sam2] Downloaded ${SIZE_MB} MB to $CHECKPOINT_PATH"

# Sanity: SAM2.1 large is ~898 MB; flag suspicious results
if [ "$SIZE_MB" -lt 500 ]; then
    echo "[sam2] WARNING: checkpoint is suspiciously small (${SIZE_MB} MB)." >&2
    exit 2
fi
