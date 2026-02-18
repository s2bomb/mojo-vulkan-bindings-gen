# vulkan-bindings-gen

Generate typed Vulkan FFI bindings for Mojo from the Khronos vk.xml spec.

---

## Prerequisites

- Python 3.11+
- `gcc` in PATH (for union sizeof measurement)
- A copy of [vk.xml](https://github.com/KhronosGroup/Vulkan-Docs) and [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers)

---

## Quick Start

```bash
git clone https://github.com/KhronosGroup/Vulkan-Docs.git
git clone https://github.com/KhronosGroup/Vulkan-Headers.git

python gen.py \
    --version 1.3 \
    --vk-xml Vulkan-Docs/xml/vk.xml \
    --vulkan-headers Vulkan-Headers/include \
    --output-dir ./vulkan-bindings
```

This generates a complete Mojo package under `./vulkan-bindings/` containing 8 files.

---

## Flag Reference

### Generate Mode

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--version VERSION` | **yes** | — | Target Vulkan version (`1.0`–`1.4`). |
| `--vk-xml PATH` | **yes** | — | Path to `vk.xml` from Vulkan-Docs. Must exist. |
| `--vulkan-headers DIR` | **yes** | — | Path to Vulkan-Headers `include/` directory. Must exist. |
| `--output-dir DIR` | no | `./vulkan-bindings` | Output directory. Created if absent. |
| `--ext NAME...` | no | — | Space-separated extension names to include. |
| `--all-extensions` | no | false | Include all extensions. Mutually exclusive with `--ext`. |

### Discovery Mode

| Flag | Required | Description |
|------|----------|-------------|
| `--list-versions` | no | Print available Vulkan versions from vk.xml. Requires `--vk-xml`. |
| `--list-extensions` | no | Print all extensions. Requires `--vk-xml`. |
| `--filter TEXT` | no | Filter extension list. Modifier for `--list-extensions` only. |
| `--info EXTENSION` | no | Print detail for a named extension. Requires `--vk-xml`. |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success — generation complete or discovery output printed. |
| `1` | Semantic error — `ConfigError` with code and hint to stderr. |
| `2` | CLI usage error — argparse usage message to stderr. |

---

## Generate Mode Examples

```bash
# Vulkan 1.3 core only
python gen.py \
    --version 1.3 \
    --vk-xml Vulkan-Docs/xml/vk.xml \
    --vulkan-headers Vulkan-Headers/include \
    --output-dir ./vulkan-bindings

# Vulkan 1.4 core only
python gen.py \
    --version 1.4 \
    --vk-xml Vulkan-Docs/xml/vk.xml \
    --vulkan-headers Vulkan-Headers/include \
    --output-dir ./vulkan-bindings

# With named extensions
python gen.py \
    --version 1.3 \
    --ext VK_KHR_swapchain VK_EXT_debug_utils \
    --vk-xml Vulkan-Docs/xml/vk.xml \
    --vulkan-headers Vulkan-Headers/include \
    --output-dir ./vulkan-bindings

# All extensions
python gen.py \
    --version 1.3 \
    --all-extensions \
    --vk-xml Vulkan-Docs/xml/vk.xml \
    --vulkan-headers Vulkan-Headers/include \
    --output-dir ./vulkan-bindings

# Custom output directory
python gen.py \
    --version 1.3 \
    --vk-xml Vulkan-Docs/xml/vk.xml \
    --vulkan-headers Vulkan-Headers/include \
    --output-dir /path/to/my/project/vulkan
```

---

## Discovery Mode Examples

```bash
# List available Vulkan versions
python gen.py --list-versions --vk-xml Vulkan-Docs/xml/vk.xml
# Output:
#   Vulkan versions in vk.xml:
#     1.0, 1.1, 1.2, 1.3, 1.4

# List extensions, filtered by keyword
python gen.py --list-extensions --filter swapchain --vk-xml Vulkan-Docs/xml/vk.xml
# Output:
#   Vulkan extensions (filter: swapchain):
#     VK_KHR_swapchain
#     VK_KHR_swapchain_mutable_format

# Show extension detail
python gen.py --info VK_KHR_swapchain --vk-xml Vulkan-Docs/xml/vk.xml
# Output:
#   VK_KHR_swapchain
#   Promoted: VK_VERSION_1_3 (partial)
#   Depends: VK_KHR_surface
```

---

## Output Structure

Each successful generation writes exactly 8 files to `--output-dir`:

| File | Contents |
|------|---------|
| `__init__.mojo` | Re-exports from all modules |
| `vk_base_types.mojo` | Base types, bitmask aliases, funcpointer types |
| `vk_enums.mojo` | All enums for target version + extensions |
| `vk_handles.mojo` | All handles (dispatchable and non-dispatchable) |
| `vk_structs.mojo` | All structs (topologically sorted) |
| `vk_unions.mojo` | All unions (as aligned byte arrays) |
| `vk_loader.mojo` | LoadProc, FuncPtr, init functions |
| `vk_commands.mojo` | Command wrappers + aliases |

All files carry a provenance header identifying the generator and target version.

---

## Running Tests

The full test suite ships with the tool. No real vk.xml is required — all tests use
inline fixture XML:

```bash
pytest tests/
```

To run only the external contract tests (subprocess-based):

```bash
pytest tests/external/
```

---

## CI Gates

The repository ships two GitHub Actions workflows:

- `CI` (PR + push):
  - unit and external tests
  - determinism check (generate twice, byte-for-byte equal output)
  - real registry gate (`vk.xml` from Vulkan-Docs + headers from Vulkan-Headers)
  - real-registry version matrix for Vulkan targets `1.0`, `1.1`, `1.2`, `1.3`, `1.4`
- `Nightly Deep Checks`:
  - full-generation stress path with `--all-extensions`
  - real-registry determinism re-check

---

## Compatibility

- CLI supports target versions `1.0` through `1.4`.
- CI validates real-registry generation for all supported target versions (`1.0`–`1.4`).
- Future `vk.xml` schema changes may require generator updates.

---

## License

MIT. See [LICENSE](LICENSE).
