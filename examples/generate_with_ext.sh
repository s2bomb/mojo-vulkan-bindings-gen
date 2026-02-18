#!/usr/bin/env bash
# Generate Vulkan 1.3 bindings with common extensions.
#
# Prerequisites: same as generate_1_3.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/../gen.py" \
    --version 1.3 \
    --ext VK_KHR_swapchain VK_KHR_surface VK_EXT_debug_utils \
    --vk-xml ./Vulkan-Docs/xml/vk.xml \
    --vulkan-headers ./Vulkan-Headers/include \
    --output-dir ./vulkan-bindings
