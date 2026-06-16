#!/bin/bash
# Wrapper to run LatentFlow scripts with proper Python path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Activate venv if available
if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

# Run the actual script
python3 "${SCRIPT_DIR}/scripts/$1" "${@:2}"
