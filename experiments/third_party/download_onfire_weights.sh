#!/usr/bin/env bash
# Download the pre-trained OnFire models (ICMLA 2020, Durham University)
# into models/onfire (shufflenet_ff.pt, shufflenet_sp.pt, nasnet_ff.pt, nasnet_sp.pt).
# The ShuffleNetV2 weights used in the paper are already in models/onfire (Git LFS).
# Source: https://github.com/NeelBhowmik/efficient-compact-fire-detection-cnn (download-models.sh)
#
# If the Durham server does not answer, the Wayback Machine copy of the same
# link (2021-01-20, "version1" of the release) is used. Both MD5 sums are the
# ones published in the original download-models.sh.
# If onfire_weights_version1.zip is next to this script, it is used directly.

set -e

URL=https://collections.durham.ac.uk/downloads/r1tb09j570z
ARCHIVE_URL=https://web.archive.org/web/20210120074112id_/https://collections.durham.ac.uk/downloads/r1tb09j570z
LOCAL_ZIP=onfire_weights_version1.zip
MD5_SUMS="5c7be76d7f8149be4057849c998ab214 6511b75be1c36147a94c98587d6bccb6"
ZIP=onfire_weights.zip
TARGET="$(cd "$(dirname "$0")/../.." && pwd)/models/onfire"

mkdir -p "$TARGET"

if [ -f "$LOCAL_ZIP" ]; then
    echo "Using local $LOCAL_ZIP"
    cp "$LOCAL_ZIP" "$ZIP"
else
    echo "Downloading OnFire weights from Durham..."
    if ! curl -fL -k --max-time 300 "$URL" -o "$ZIP"; then
        echo "Durham server not available, using the Wayback Machine copy..."
        curl -fL --max-time 600 "$ARCHIVE_URL" -o "$ZIP"
    fi
fi

echo "Checking MD5..."
SUM=$(md5sum "$ZIP" | cut -d' ' -f1)
case " $MD5_SUMS " in
    *" $SUM "*) echo "$ZIP: OK ($SUM)" ;;
    *) echo "MD5 mismatch: $SUM"; exit 1 ;;
esac

# The zip contains the .pt files directly or inside a nested weights.zip
TMP=$(mktemp -d)
unzip -q "$ZIP" -d "$TMP"
find "$TMP" -name "*.zip" -exec unzip -q -o {} -d "$TMP" \;
find "$TMP" -name "*.pt" -exec cp {} "$TARGET"/ \;
rm -rf "$TMP" "$ZIP"

echo "Weights in $TARGET:"
ls -l "$TARGET"
