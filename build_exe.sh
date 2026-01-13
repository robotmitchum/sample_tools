#!/bin/bash

# Build executable on Linux/macOS
cd "$(dirname "$0")"
source .venv/bin/activate
python3 -m build_exe ir_tool_UI.py build_config.json bundle_config.json