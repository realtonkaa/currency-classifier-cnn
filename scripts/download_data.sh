#!/usr/bin/env bash
# download_data.sh — placeholder for dataset download script
#
# TODO: replace with actual download commands once dataset is hosted.
# Expected structure after download:
#   data/
#     USD_1/   *.jpg
#     USD_5/   *.jpg
#     USD_10/  *.jpg
#     USD_20/  *.jpg
#     USD_50/  *.jpg
#     USD_100/ *.jpg
#     EUR_5/   *.jpg
#     ...
#
# Usage:
#   bash scripts/download_data.sh

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"

echo "Data directory: $DATA_DIR"
echo "This script is a placeholder — add download commands here."

# Example (commented out):
# wget -q -O /tmp/banknotes.zip "https://example.com/banknotes.zip"
# unzip -q /tmp/banknotes.zip -d "$DATA_DIR"
# rm /tmp/banknotes.zip

echo "Done."
