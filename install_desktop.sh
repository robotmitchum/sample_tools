#!/bin/bash

# Install app with icon on Linux

cd "$(dirname "$0")"
source .venv/bin/activate
python3 -m install_desktop ~/opt/splitstream/SplitStream UI/icons/splitstream_64.png "SplitStream"