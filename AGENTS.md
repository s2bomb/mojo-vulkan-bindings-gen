# vulkan-bindings-gen

Python code generator that produces typed Mojo FFI bindings from the Khronos vk.xml spec. Outputs a multi-file Mojo package under `--output-dir`.

**Category**: tool
**Language**: Python
**Input**: `--vk-xml` (required), `--vulkan-headers` (required)
**Output**: Multi-file Mojo package under `--output-dir` (default: `./vulkan-bindings`):
`__init__.mojo`, `vk_base_types.mojo`, `vk_enums.mojo`, `vk_handles.mojo`,
`vk_structs.mojo`, `vk_unions.mojo`, `vk_loader.mojo`, `vk_commands.mojo`

---

## What This Does

Parses vk.xml and generates a split Mojo package (~8,200 lines across 8 files):
- 388 enum types with named constants
- 58 handle types (dispatchable + non-dispatchable)
- 1,379 C-ABI-compatible structs with `@fieldwise_init`
- 16 unions as aligned byte arrays (Mojo has no native union)
- 704 command wrappers (snake_case) + 105 alias wrappers
- Loader infrastructure (`init_vulkan_global`, `init_vulkan_instance`, `init_vulkan_device`)

Every struct size is verified against gcc ground truth at generation time.

---

## Usage

```bash
# Regenerate bindings
python gen.py \
  --version 1.4 \
  --vk-xml /path/to/Vulkan-Docs/xml/vk.xml \
  --vulkan-headers /path/to/Vulkan-Headers/include \
  --output-dir ./vulkan-bindings

# Run tests (no real vk.xml required)
pytest tests/

# Verify output compiles
mojo package ./vulkan-bindings -o /tmp/vulkan_test.mojopkg
```

---

## Generated Output Files

| File | Contents |
|------|---------|
| `__init__.mojo` | Re-exports from all 7 modules |
| `vk_base_types.mojo` | Base types, bitmask aliases, funcpointer types |
| `vk_enums.mojo` | All enums for target version + extensions |
| `vk_handles.mojo` | All handles (dispatchable and non-dispatchable) |
| `vk_structs.mojo` | All structs (topologically sorted) |
| `vk_unions.mojo` | All unions (as aligned byte arrays) |
| `vk_loader.mojo` | LoadProc, FuncPtr, init functions |
| `vk_commands.mojo` | Command wrappers + aliases |

---

## Discovery Mode

```bash
# List versions available in vk.xml
python gen.py \
  --list-versions \
  --vk-xml /path/to/Vulkan-Docs/xml/vk.xml

# List extensions, filter by keyword
python gen.py \
  --list-extensions --filter swapchain \
  --vk-xml /path/to/Vulkan-Docs/xml/vk.xml

# Show extension detail
python gen.py \
  --info VK_KHR_swapchain \
  --vk-xml /path/to/Vulkan-Docs/xml/vk.xml
```

---

## Notes

The tool intentionally has no required dependency on any particular repository layout.
Always pass explicit `--vk-xml` and `--vulkan-headers` paths for portable usage.
