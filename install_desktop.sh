#!/bin/bash

# Install app with icon on Linux

cd "$(dirname "$0")"
source .venv/bin/activate
python3 -m install_desktop ~/opt/sample_tools/sample_tools_UI tools/UI/icons/sample_tools_64.png "SampleTools" "Tools for creating and editing virtual sampled instruments"