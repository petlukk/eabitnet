#!/bin/bash
set -euo pipefail

KERNEL_DIR=kernels
OUT_DIR=build/lib
EA="${EA:-$HOME/projects/eacompute/target/release/ea}"

mkdir -p "$OUT_DIR"

echo "Building eabitnet kernels..."

ARCH=$(uname -m)

for f in $KERNEL_DIR/*.ea; do
    stem=$(basename "$f" .ea)
    # Skip ARM kernels on x86_64
    if [ "$ARCH" = "x86_64" ] && echo "$stem" | grep -q '_arm$'; then
        echo "  $stem.ea — skipped (ARM-only on $ARCH)"
        continue
    fi
    echo "  $stem.ea → lib${stem}.so"
    "$EA" "$f" --lib -o "$OUT_DIR/lib${stem}.so"
done

echo ""
echo "Done."
ls -la "$OUT_DIR/"
