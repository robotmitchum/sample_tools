#!/bin/bash

# Build executable on Linux/macOS
cd "$(dirname "$0")"
source .venv/bin/activate
python3 -m build_exe sample_tools_UI.py build_config.json