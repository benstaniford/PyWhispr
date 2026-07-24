#!/bin/bash
# Build packaging/PyWhispr.icns from the 512px app icon.
set -euo pipefail
cd "$(dirname "$0")"

SRC=../src/pywhispr/assets/icon.png
ICONSET=PyWhispr.iconset
rm -rf "$ICONSET" && mkdir "$ICONSET"

for size in 16 32 64 128 256 512; do
    sips -z $size $size "$SRC" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    if [ $size -lt 512 ]; then
        double=$((size * 2))
        sips -z $double $double "$SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
    fi
done

iconutil -c icns "$ICONSET" -o PyWhispr.icns
rm -rf "$ICONSET"
echo "wrote packaging/PyWhispr.icns"
