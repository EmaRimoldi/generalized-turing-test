#!/usr/bin/env bash
# Compatibility entrypoint for the HF image.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build-container.sh" --hf
