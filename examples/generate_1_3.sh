#!/usr/bin/env bash
# Generate Vulkan 1.3 core bindings.
#
# Prerequisites:
#   git clone https://github.com/KhronosGroup/Vulkan-Docs.git
#   git clone https://github.com/KhronosGroup/Vulkan-Headers.git
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/../gen.py" \
    --version 1.3 \
    --vk-xml ./Vulkan-Docs/xml/vk.xml \
    --vulkan-headers ./Vulkan-Headers/include \
    --output-dir ./vulkan-bindings
