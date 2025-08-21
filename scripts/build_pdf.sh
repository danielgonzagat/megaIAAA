#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist"
mkdir -p "$OUT_DIR"
pandoc "$ROOT_DIR/docs/whitepaper.md" -o "$OUT_DIR/Lemniscata_de_Penin.pdf"
echo "Generated $OUT_DIR/Lemniscata_de_Penin.pdf"
