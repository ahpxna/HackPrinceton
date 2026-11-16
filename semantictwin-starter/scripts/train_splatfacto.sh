#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "Usage: bash scripts/train_splatfacto.sh /path/to/processed-nerfstudio-dataset" >&2; exit 2; fi
if ! command -v ns-train >/dev/null; then echo 'Nerfstudio ns-train is not installed. Follow official Nerfstudio CUDA instructions on the RTX 3090.' >&2; exit 1; fi
if [ ! -d "$1" ]; then echo 'Missing dataset directory' >&2; exit 1; fi
echo 'Training Splatfacto. Scene must have valid camera poses; scale accuracy is separate.'
ns-train splatfacto --data "$1"
