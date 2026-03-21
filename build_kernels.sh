#!/bin/bash
set -euo pipefail

KERNEL_DIR=kernels
OUT_DIR=build/lib
EA="${EA:-$HOME/projects/eacompute/target/release/ea}"

mkdir -p "$OUT_DIR"

echo "Building eabitnet kernels..."

for f in $KERNEL_DIR/*.ea; do
    stem=$(basename "$f" .ea)
    echo "  $stem.ea → lib${stem}.so"
    "$EA" "$f" --lib -o "$OUT_DIR/lib${stem}.so"
done

echo ""
echo "Done."
ls -la "$OUT_DIR/"
