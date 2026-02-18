"""Vulkan FFI bindings generator for Mojo.

Generates typed Vulkan bindings from the Khronos vk.xml spec.
Produces a staged `vk_*` module package under libs/vulkan/src.

Usage:
    pixi run python tools/vulkan-bindings-gen/gen.py --version 1.4
"""

import os
import argparse
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import NamedTuple, TypeVar

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_VK_XML = PROJECT_ROOT / "thoughts" / "repos" / "Vulkan-Docs" / "xml" / "vk.xml"
DEFAULT_VULKAN_HEADERS = (
    PROJECT_ROOT / "thoughts" / "repos" / "Vulkan-Headers" / "include"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "libs" / "vulkan" / "src"
VK_XML = DEFAULT_VK_XML
VULKAN_HEADERS = DEFAULT_VULKAN_HEADERS
OUT_DIR = DEFAULT_OUTPUT_DIR


# ===--- CLI config contracts ---=== #


class VulkanVersion(NamedTuple):
    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class GenerateConfig:
    version: VulkanVersion
    extensions: frozenset[str]
    all_extensions: bool
    vk_xml: Path
    vulkan_headers: Path
    output_dir: Path


@dataclass(frozen=True)
class DiscoveryConfig:
    command: str
    filter_text: str | None
    info_extension: str | None
    vk_xml: Path


VALID_ERROR_CODES = {
    "INVALID_VERSION",
    "MISSING_VERSION",
    "INVALID_EXTENSION_NAME",
    "CONFLICT_EXT_FLAGS",
    "CONFLICT_GENERATE_DISCOVERY",
    "FILTER_WITHOUT_LIST",
    "PATH_NOT_FOUND",
}
VALID_VERSIONS = {"1.0", "1.1", "1.2", "1.3", "1.4"}
_EXT_NAME_RE = re.compile(r"^VK_[A-Z0-9]+_[A-Za-z0-9_]+$")


class ConfigError(Exception):
    def __init__(self, code: str, message: str, suggestion: str | None = None):
        if code not in VALID_ERROR_CODES:
            raise ValueError(f"Unknown config error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion


def parse_version(raw: str) -> VulkanVersion:
    if raw not in VALID_VERSIONS:
        raise ConfigError(
            "INVALID_VERSION",
            f"Unsupported Vulkan version: {raw}",
            "Use one of: 1.0, 1.1, 1.2, 1.3, 1.4.",
        )
    major_text, minor_text = raw.split(".", maxsplit=1)
    return VulkanVersion(int(major_text), int(minor_text))


def validate_extension_name(name: str) -> str:
    if _EXT_NAME_RE.match(name):
        return name
    raise ConfigError(
        "INVALID_EXTENSION_NAME",
        f"Invalid extension name: {name}",
        "Extension names must match VK_<VENDOR>_<name> (for example VK_KHR_swapchain).",
    )


def validate_path_exists(
    path: Path | None, flag: str, suggestion: str | None = None
) -> Path:
    if path is None:
        raise ConfigError(
            "PATH_NOT_FOUND",
            f"{flag} is required: no path provided.",
            suggestion or f"Pass the path explicitly: {flag} /path/to/resource",
        )
    if path.exists():
        return path
    raise ConfigError(
        "PATH_NOT_FOUND",
        f"Path for {flag} does not exist: {path}",
        suggestion or "Provide an existing path for this flag.",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Vulkan bindings for Mojo")

    parser.add_argument("--version", type=str, default=None)

    ext_group = parser.add_mutually_exclusive_group()
    ext_group.add_argument("--ext", action="append", nargs="+", default=None)
    ext_group.add_argument("--all-extensions", action="store_true", default=False)

    parser.add_argument("--vk-xml", type=Path, default=DEFAULT_VK_XML)
    parser.add_argument("--vulkan-headers", type=Path, default=DEFAULT_VULKAN_HEADERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    discovery_group = parser.add_mutually_exclusive_group()
    discovery_group.add_argument("--list-versions", action="store_true", default=False)
    discovery_group.add_argument(
        "--list-extensions", action="store_true", default=False
    )
    discovery_group.add_argument("--info", type=str, default=None)

    parser.add_argument("--filter", type=str, default=None)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_argument_parser()
    return parser.parse_args(argv)


def normalize_extensions(raw_extensions: object) -> tuple[str, ...]:
    if raw_extensions is None:
        return tuple()
    if not isinstance(raw_extensions, list):
        raise ConfigError(
            "INVALID_EXTENSION_NAME",
            f"Invalid --ext value type: {type(raw_extensions).__name__}",
            "Pass extension names as --ext VK_VENDOR_name.",
        )

    normalized: list[str] = []
    for entry in raw_extensions:
        if isinstance(entry, str):
            normalized.append(entry)
            continue
        if isinstance(entry, list):
            for name in entry:
                if not isinstance(name, str):
                    raise ConfigError(
                        "INVALID_EXTENSION_NAME",
                        f"Invalid extension name type: {type(name).__name__}",
                        "Pass extension names as --ext VK_VENDOR_name.",
                    )
                normalized.append(name)
            continue
        raise ConfigError(
            "INVALID_EXTENSION_NAME",
            f"Invalid --ext entry type: {type(entry).__name__}",
            "Pass extension names as --ext VK_VENDOR_name.",
        )

    return tuple(normalized)


def validate_config(args: argparse.Namespace) -> GenerateConfig | DiscoveryConfig:
    raw_extensions = normalize_extensions(args.ext)
    has_generate_input = bool(args.version or raw_extensions or args.all_extensions)
    has_discovery_command = bool(
        args.list_versions or args.list_extensions or args.info
    )

    if args.filter and not args.list_extensions:
        raise ConfigError(
            "FILTER_WITHOUT_LIST",
            "--filter requires --list-extensions.",
            "Add --list-extensions or remove --filter.",
        )

    if has_generate_input and has_discovery_command:
        raise ConfigError(
            "CONFLICT_GENERATE_DISCOVERY",
            "Generate flags cannot be combined with discovery flags.",
            "Choose either generate mode or one discovery command.",
        )

    if raw_extensions and args.all_extensions:
        raise ConfigError(
            "CONFLICT_EXT_FLAGS",
            "Cannot combine --ext with --all-extensions.",
            "Use --ext with one or more names, or --all-extensions.",
        )

    if has_discovery_command:
        vk_xml = validate_path_exists(
            args.vk_xml,
            "--vk-xml",
            "Clone Vulkan-Docs:\n"
            "  git clone https://github.com/KhronosGroup/Vulkan-Docs.git thoughts/repos/Vulkan-Docs\n"
            "Or pass a custom path: --vk-xml /your/path/to/vk.xml",
        )
        if args.list_versions:
            command = "list-versions"
        elif args.list_extensions:
            command = "list-extensions"
        else:
            command = "info"

        info_extension = (
            validate_extension_name(args.info) if args.info is not None else None
        )
        return DiscoveryConfig(
            command=command,
            filter_text=args.filter,
            info_extension=info_extension,
            vk_xml=vk_xml,
        )

    if args.version is None:
        raise ConfigError(
            "MISSING_VERSION",
            "Generate mode requires --version.",
            "Pass --version with one of: 1.0, 1.1, 1.2, 1.3, 1.4.",
        )

    version = parse_version(args.version)
    vk_xml = validate_path_exists(
        args.vk_xml,
        "--vk-xml",
        "Clone Vulkan-Docs:\n"
        "  git clone https://github.com/KhronosGroup/Vulkan-Docs.git thoughts/repos/Vulkan-Docs\n"
        "Or pass a custom path: --vk-xml /your/path/to/vk.xml",
    )
    vulkan_headers = validate_path_exists(
        args.vulkan_headers,
        "--vulkan-headers",
        "Clone Vulkan-Headers:\n"
        "  git clone https://github.com/KhronosGroup/Vulkan-Headers.git thoughts/repos/Vulkan-Headers\n"
        "Or pass a custom path: --vulkan-headers /your/path/to/include",
    )
    extensions = (
        frozenset(validate_extension_name(name) for name in raw_extensions)
        if raw_extensions
        else frozenset()
    )

    return GenerateConfig(
        version=version,
        extensions=extensions,
        all_extensions=bool(args.all_extensions),
        vk_xml=vk_xml,
        vulkan_headers=vulkan_headers,
        output_dir=args.output_dir,
    )


def build_config(argv: list[str] | None = None) -> GenerateConfig | DiscoveryConfig:
    return validate_config(parse_args(argv))


# ===--- Constants ---=== #

C_TO_MOJO = {
    "void": "NoneType",
    "char": "c_char",
    "float": "c_float",
    "double": "c_double",
    "int": "c_int",
    "int32_t": "Int32",
    "int64_t": "Int64",
    "uint8_t": "UInt8",
    "uint16_t": "UInt16",
    "uint32_t": "UInt32",
    "uint64_t": "UInt64",
    "size_t": "c_size_t",
}

VK_BASE_TYPES = {
    "VkBool32",
    "VkDeviceAddress",
    "VkDeviceSize",
    "VkFlags",
    "VkFlags64",
    "VkSampleMask",
    "VkRemoteAddressNV",
}

# Platform-specific types mapped to size-compatible Mojo types.
PLATFORM_TYPES = {
    "DWORD": "UInt32",
    "HANDLE": "UInt64",
    "HINSTANCE": "UInt64",
    "HWND": "UInt64",
    "HMONITOR": "UInt64",
    "LPCWSTR": "UnsafePointer[UInt16, MutAnyOrigin]",
    "SECURITY_ATTRIBUTES": "UInt8",
    "MTLDevice_id": "UInt64",
    "MTLBuffer_id": "UInt64",
    "MTLCommandQueue_id": "UInt64",
    "MTLSharedEvent_id": "UInt64",
    "MTLTexture_id": "UInt64",
    "IOSurfaceRef": "UInt64",
    "CAMetalLayer": "UInt8",
    "Display": "UInt8",
    "Window": "UInt64",
    "xcb_connection_t": "UInt8",
    "xcb_window_t": "UInt32",
    "wl_display": "UInt8",
    "wl_surface": "UInt8",
    "ANativeWindow": "UInt8",
    "AHardwareBuffer": "UInt8",
    "zx_handle_t": "UInt32",
    "GgpFrameToken": "UInt32",
    "GgpStreamDescriptor": "UInt32",
    "NvSciBufAttrList": "UInt64",
    "NvSciBufObj": "UInt64",
    "NvSciSyncAttrList": "UInt64",
    "NvSciSyncFence": "UInt64",
    "NvSciSyncObj": "UInt64",
    "_screen_buffer": "UInt8",
    "_screen_context": "UInt8",
    "_screen_window": "UInt8",
    "IDirectFB": "UInt8",
    "IDirectFBSurface": "UInt8",
    "ubm_device": "UInt8",
    "ubm_surface": "UInt8",
    "OHBufferHandle": "UInt64",
    "OHNativeWindow": "UInt8",
    "OH_NativeBuffer": "UInt8",
}

MOJO_RESERVED = {"ref", "in", "out", "var", "fn", "type"}

ENUM_BASE_VALUE = 1000000000
ENUM_RANGE_SIZE = 1000

INSTANCE_LEVEL_TYPES = {"VkInstance", "VkPhysicalDevice"}
DEVICE_LEVEL_TYPES = {
    "VkDevice",
    "VkQueue",
    "VkCommandBuffer",
    "VkExternalComputeQueueNV",
}
GLOBAL_COMMANDS = {
    "vkCreateInstance",
    "vkEnumerateInstanceLayerProperties",
    "vkEnumerateInstanceExtensionProperties",
    "vkEnumerateInstanceVersion",
    "vkGetInstanceProcAddr",
}


# ===--- Data classes ---=== #


class StructMember:
    def __init__(
        self,
        name: str,
        type_name: str,
        is_pointer: bool,
        is_double_pointer: bool,
        array_size: str | None,
        stype_value: str | None,
        bitwidth: int | None = None,
    ):
        self.name = name
        self.type_name = type_name
        self.is_pointer = is_pointer
        self.is_double_pointer = is_double_pointer
        self.array_size = array_size
        self.stype_value = stype_value
        self.bitwidth = bitwidth


class StructDef:
    def __init__(self, name: str, members: list[StructMember]):
        self.name = name
        self.members = members


class CommandParam:
    def __init__(self, name: str, type_name: str, ptr_count: int, is_const: bool):
        self.name = name
        self.type_name = type_name
        self.ptr_count = ptr_count
        self.is_const = is_const

    @property
    def mojo_name(self) -> str:
        if self.name in MOJO_RESERVED:
            return self.name + "_"
        return self.name


class CommandDef:
    def __init__(
        self,
        name: str,
        return_type: str,
        return_is_void: bool,
        params: list[CommandParam],
    ):
        self.name = name
        self.return_type = return_type
        self.return_is_void = return_is_void
        self.params = params


# ===--- XML parsing ---=== #


def load_api_constants(root: ET.Element) -> dict[str, int]:
    """Load VK_MAX_* etc constants from the API Constants enum block."""
    constants = {}
    for block in root.findall("enums"):
        if block.get("name") == "API Constants":
            for val in block.findall("enum"):
                name = val.get("name")
                value = val.get("value")
                alias = val.get("alias")
                if name and value:
                    try:
                        constants[name] = int(value)
                    except ValueError:
                        pass
                elif name and alias and alias in constants:
                    constants[name] = constants[alias]
    return constants


def load_all_type_names(root: ET.Element) -> dict[str, str]:
    """Build a map of all type names -> category for resolution."""
    types = {}
    for t in root.findall("types/type"):
        if t.get("api", "") == "vulkansc":
            continue
        cat = t.get("category", "")
        if not cat:
            continue
        name = t.get("name")
        if not name:
            name_el = t.find("name")
            if name_el is not None:
                name = name_el.text
        if not name and cat == "funcpointer":
            proto_name = t.find("proto/name")
            if proto_name is not None:
                name = proto_name.text
        if name:
            types[name] = cat
    return types


def build_known_types(root: ET.Element) -> set[str]:
    """Build set of known Mojo types from vk.xml."""
    known = set(VK_BASE_TYPES)
    for t in root.findall("types/type"):
        if t.get("api", "") == "vulkansc":
            continue
        cat = t.get("category", "")
        if cat in ("enum", "struct", "union", "handle", "bitmask", "funcpointer"):
            name = t.get("name")
            if not name:
                name_el = t.find("name")
                if name_el is not None:
                    name = name_el.text
            if name:
                known.add(name)
        elif cat == "basetype":
            type_el = t.find("type")
            if type_el is not None:
                name_el = t.find("name")
                if name_el is not None and name_el.text:
                    known.add(name_el.text)
    for t in root.findall("types/type[@category='struct']"):
        for m in t.findall("member"):
            type_el = m.find("type")
            if (
                type_el is not None
                and type_el.text
                and type_el.text.startswith("StdVideo")
            ):
                known.add(type_el.text)
    return known


# ===--- Base types ---=== #


def extract_basetypes(root: ET.Element) -> list[tuple[str, str, str]]:
    results = []
    for t in root.findall("types/type[@category='basetype']"):
        if t.get("api", "") == "vulkansc":
            continue
        name_el = t.find("name")
        type_el = t.find("type")
        if name_el is None:
            continue
        name = name_el.text
        raw_text = ET.tostring(t, encoding="unicode")
        if type_el is not None and type_el.text:
            if "*" in raw_text and type_el.text == "void":
                results.append((name, "void", "void_ptr"))
            else:
                results.append((name, type_el.text, "typedef"))
        else:
            results.append((name, "", "opaque"))
    return results


def generate_base_types(basetypes: list[tuple[str, str, str]]) -> list[str]:
    lines = []
    lines.append("comptime Ptr = UnsafePointer")
    lines.append("")
    lines.append("# ========= BASE TYPES =========")
    lines.append("")
    for name, underlying, pattern in sorted(basetypes, key=lambda x: x[0]):
        if pattern == "typedef":
            mojo_type = C_TO_MOJO.get(underlying)
            if mojo_type is None:
                continue
            lines.append(f"comptime {name} = {mojo_type}")
        elif pattern == "void_ptr":
            lines.append(f"comptime {name} = Ptr[NoneType, MutAnyOrigin]")
    lines.append("")
    return lines


# ===--- Enums ---=== #


def _parse_c_int(s: str) -> int:
    s = s.strip()
    if s.startswith("0x") or s.startswith("0X"):
        return int(s, 16)
    if s.startswith("-"):
        return -int(s[1:], 16 if s[1:].startswith("0x") else 10)
    return int(s)


def collect_enum_values(root: ET.Element) -> dict[str, list[tuple[str, int]]]:
    enums = defaultdict(list)
    seen = defaultdict(set)

    for block in root.findall("enums"):
        block_name = block.get("name", "")
        if block_name == "API Constants":
            continue
        for val in block.findall("enum"):
            name = val.get("name")
            if not name or val.get("alias"):
                continue
            value_str = val.get("value")
            bitpos = val.get("bitpos")
            if value_str is not None:
                int_val = _parse_c_int(value_str)
            elif bitpos is not None:
                int_val = 1 << int(bitpos)
            else:
                continue
            if name not in seen[block_name]:
                enums[block_name].append((name, int_val))
                seen[block_name].add(name)

    for feat in root.findall("feature"):
        if "vulkan" not in feat.get("api", ""):
            continue
        for req in feat.findall("require"):
            for val in req.findall("enum"):
                _process_extension_enum(val, None, enums, seen)

    for ext in root.findall("extensions/extension"):
        if "vulkan" not in ext.get("supported", ""):
            continue
        extnumber = ext.get("number")
        for req in ext.findall("require"):
            for val in req.findall("enum"):
                _process_extension_enum(val, extnumber, enums, seen)

    return dict(enums)


def _process_extension_enum(val, default_extnumber, enums, seen):
    name = val.get("name")
    extends = val.get("extends")
    if not name or not extends or val.get("alias") or name in seen[extends]:
        return
    value_str = val.get("value")
    bitpos = val.get("bitpos")
    offset = val.get("offset")
    if value_str is not None:
        int_val = _parse_c_int(value_str)
    elif bitpos is not None:
        int_val = 1 << int(bitpos)
    elif offset is not None:
        extnumber = val.get("extnumber", default_extnumber)
        if extnumber is None:
            return
        int_val = ENUM_BASE_VALUE + (int(extnumber) - 1) * ENUM_RANGE_SIZE + int(offset)
        if val.get("dir") == "-":
            int_val = -int_val
    else:
        return
    enums[extends].append((name, int_val))
    seen[extends].add(name)


def get_enum_block_types(root: ET.Element) -> dict[str, str]:
    types = {}
    for block in root.findall("enums"):
        name = block.get("name", "")
        block_type = block.get("type", "")
        if name and block_type:
            types[name] = block_type
    return types


def get_enum_type_names(root: ET.Element) -> set[str]:
    names = set()
    for t in root.findall("types/type[@category='enum']"):
        if t.get("api", "") == "vulkansc":
            continue
        name = t.get("name")
        if name:
            names.add(name)
    return names


def generate_enums(
    enum_values: dict[str, list[tuple[str, int]]],
    block_types: dict[str, str],
    enum_type_names: set[str],
) -> list[str]:
    lines = []
    lines.append("# ========= ENUMS =========")
    lines.append("")

    for enum_name in sorted(enum_type_names):
        values = enum_values.get(enum_name, [])
        is_bitmask = block_types.get(enum_name) == "bitmask"

        lines.append("@fieldwise_init")
        lines.append(f"struct {enum_name}(TrivialRegisterPassable, Intable):")
        lines.append("    var value: Int32")
        lines.append("    ")

        if values:
            for vname, vval in values:
                lines.append(f"    comptime {vname} = {enum_name}({vval})")
            lines.append("    ")

        lines.append("    @always_inline")
        lines.append("    fn __int__(self) -> Int:")
        lines.append("        return Int(self.value)")

        if is_bitmask:
            lines.append("    ")
            lines.append("    @always_inline")
            lines.append("    fn __or__(lhs, rhs: Self) -> Self:")
            lines.append("        return Self(lhs.value | rhs.value)")

        lines.append("")

    return lines


# ===--- Handles ---=== #


def extract_handles(root: ET.Element):
    handles = []
    aliases = []
    for t in root.findall("types/type[@category='handle']"):
        if t.get("api", "") == "vulkansc":
            continue
        alias = t.get("alias")
        if alias:
            aliases.append((t.get("name", ""), alias))
            continue
        name_el = t.find("name")
        type_el = t.find("type")
        if name_el is None or type_el is None:
            continue
        name = name_el.text
        parent = t.get("parent", "")
        if type_el.text == "VK_DEFINE_HANDLE":
            handles.append((name, parent, "dispatchable"))
        elif type_el.text == "VK_DEFINE_NON_DISPATCHABLE_HANDLE":
            handles.append((name, parent, "non_dispatchable"))
    return handles, aliases


def generate_handles(handles, aliases) -> list[str]:
    lines = []
    lines.append("comptime Ptr = UnsafePointer")
    lines.append("")
    lines.append("# ========= HANDLES =========")
    lines.append(f"# {len(handles)} handle types + {len(aliases)} aliases")
    lines.append("")

    for name, parent, kind in sorted(handles, key=lambda x: x[0]):
        lines.append("@fieldwise_init")
        lines.append(f"struct {name}_T(Copyable, Movable):")
        lines.append(
            f'    """Opaque {kind} handle internal type. parent={parent or "none"}"""'
        )
        lines.append("    pass")
        lines.append("")
        lines.append(f"comptime {name} = Ptr[{name}_T, MutAnyOrigin]")
        lines.append("")

    for alias_name, target_name in sorted(aliases, key=lambda x: x[0]):
        lines.append(f"comptime {alias_name} = {target_name}")
    lines.append("")

    return lines


# ===--- Bitmask aliases ---=== #


def generate_bitmask_aliases(root: ET.Element) -> list[str]:
    lines = []
    lines.append("# ========= BITMASK TYPE ALIASES =========")
    lines.append("# VkFooFlags = VkFlags (UInt32) or VkFlags64 (UInt64)")
    lines.append("")

    for t in root.findall("types/type[@category='bitmask']"):
        if t.get("api", "") == "vulkansc":
            continue
        alias = t.get("alias")
        if alias:
            name = t.get("name", "")
            if name:
                lines.append(f"comptime {name} = {alias}")
            continue
        name_el = t.find("name")
        type_el = t.find("type")
        if name_el is None or type_el is None:
            continue
        name = name_el.text
        underlying = type_el.text
        if not name:
            continue
        if underlying == "VkFlags":
            lines.append(f"comptime {name} = VkFlags")
        elif underlying == "VkFlags64":
            lines.append(f"comptime {name} = VkFlags64")
        else:
            lines.append(
                f"comptime {name} = UInt32  # unknown underlying: {underlying}"
            )
    lines.append("")

    return lines


# ===--- Function pointer aliases ---=== #


def generate_funcpointer_aliases(root: ET.Element) -> list[str]:
    lines = []
    lines.append("# ========= FUNCTION POINTER TYPES =========")
    lines.append("# Opaque pointers — actual signatures in command generation")
    lines.append("")

    for t in root.findall("types/type[@category='funcpointer']"):
        name_el = t.find("name")
        if name_el is None:
            name_el = t.find("proto/name")
        if name_el is None:
            continue
        name = name_el.text
        if name:
            lines.append(f"comptime {name} = UnsafePointer[NoneType, MutAnyOrigin]")
    lines.append("")

    return lines


# ===--- StdVideo placeholder types ---=== #


def generate_stdvideo_types(root: ET.Element) -> list[str]:
    lines = []
    lines.append("# ========= STD VIDEO TYPES =========")
    lines.append("# From vulkan_video.h — placeholder definitions for compilation.")
    lines.append("")

    value_types = set()
    ptr_only_types = set()

    for t in root.findall("types/type[@category='struct']"):
        if t.get("api") == "vulkansc" or t.get("alias"):
            continue
        for m in t.findall("member"):
            type_el = m.find("type")
            if type_el is None:
                continue
            type_name = type_el.text or ""
            type_tail = type_el.tail or ""
            if type_name.startswith("StdVideo"):
                if "*" in type_tail:
                    ptr_only_types.add(type_name)
                else:
                    value_types.add(type_name)

    ptr_only_types -= value_types

    for name in sorted(value_types):
        lines.append("@fieldwise_init")
        lines.append(f"struct {name}(TrivialRegisterPassable, Intable):")
        lines.append('    """Video codec enum from vulkan_video.h."""')
        lines.append("    var value: Int32")
        lines.append("    ")
        lines.append("    @always_inline")
        lines.append("    fn __int__(self) -> Int:")
        lines.append("        return Int(self.value)")
        lines.append("")

    for name in sorted(ptr_only_types):
        lines.append("@fieldwise_init")
        lines.append(f"struct {name}(Copyable, Movable):")
        lines.append('    """Opaque video codec struct from vulkan_video.h."""')
        lines.append("    pass")
        lines.append("")

    lines.append("")
    return lines


# ===--- Struct/union member parsing ---=== #


def parse_member(m: ET.Element) -> StructMember | None:
    if m.get("api", "") == "vulkansc":
        return None
    type_el = m.find("type")
    name_el = m.find("name")
    if type_el is None or name_el is None:
        return None
    type_name = type_el.text or ""
    member_name = name_el.text or ""
    type_tail = type_el.tail or ""
    name_tail = name_el.tail or ""

    star_count = type_tail.count("*")
    is_pointer = star_count >= 1
    is_double_pointer = star_count >= 2

    # Array size
    array_size = None
    if "[" in name_tail:
        enum_el = m.find("enum")
        if enum_el is not None and enum_el.text:
            array_size = enum_el.text.strip()
        else:
            dims = re.findall(r"\[(\d+)\]", name_tail)
            if dims:
                total = 1
                for d in dims:
                    total *= int(d)
                array_size = str(total)

    # Bitfield width
    bitwidth = None
    bit_match = re.search(r":(\d+)", name_tail)
    if bit_match and "[" not in name_tail:
        bitwidth = int(bit_match.group(1))

    stype_value = m.get("values") if member_name == "sType" else None

    return StructMember(
        name=member_name,
        type_name=type_name,
        is_pointer=is_pointer,
        is_double_pointer=is_double_pointer,
        array_size=array_size,
        stype_value=stype_value,
        bitwidth=bitwidth,
    )


def extract_structs(root: ET.Element):
    structs = []
    aliases = []
    for t in root.findall("types/type[@category='struct']"):
        if t.get("api", "") == "vulkansc":
            continue
        alias = t.get("alias")
        if alias:
            aliases.append((t.get("name", ""), alias))
            continue
        name = t.get("name", "")
        members = []
        seen_api_members = set()
        for m in t.findall("member"):
            m_api = m.get("api", "")
            name_el = m.find("name")
            m_name = name_el.text if name_el is not None else ""
            if m_api == "vulkansc":
                continue
            if m_api and "vulkan" in m_api and m_name in seen_api_members:
                continue
            parsed = parse_member(m)
            if parsed:
                if m_api:
                    seen_api_members.add(parsed.name)
                members.append(parsed)
        structs.append(StructDef(name, members))
    return structs, aliases


def extract_unions(root: ET.Element):
    unions = []
    aliases = []
    for t in root.findall("types/type[@category='union']"):
        if t.get("api", "") == "vulkansc":
            continue
        alias = t.get("alias")
        if alias:
            aliases.append((t.get("name", ""), alias))
            continue
        name = t.get("name", "")
        members = []
        for m in t.findall("member"):
            parsed = parse_member(m)
            if parsed:
                members.append(parsed)
        unions.append(StructDef(name, members))
    return unions, aliases


# ===--- Dependency ordering ---=== #


def topo_sort_structs(structs: list[StructDef], all_type_names: dict[str, str]):
    struct_map = {s.name: s for s in structs}
    struct_names = set(struct_map.keys())
    deps = {}
    for s in structs:
        value_deps = set()
        for m in s.members:
            if (
                not m.is_pointer
                and m.type_name in struct_names
                and m.type_name != s.name
            ):
                value_deps.add(m.type_name)
            if m.array_size and not m.is_pointer and m.type_name in struct_names:
                value_deps.add(m.type_name)
        deps[s.name] = value_deps

    in_degree = {s: 0 for s in deps}
    adj = defaultdict(list)
    for s, dd in deps.items():
        for d in dd:
            if d in deps:
                adj[d].append(s)
                in_degree[s] += 1

    queue = [s for s in deps if in_degree[s] == 0]
    result = []
    while queue:
        queue.sort()
        node = queue.pop(0)
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(deps):
        remaining = set(deps.keys()) - set(result)
        raise RuntimeError(f"Dependency cycle in structs: {remaining}")

    return [struct_map[name] for name in result]


# ===--- Mojo type mapping ---=== #


def mojo_type_for_member(member, all_type_names, api_constants) -> str:
    base_type = member.type_name
    if base_type in C_TO_MOJO:
        mojo_base = C_TO_MOJO[base_type]
    elif base_type in VK_BASE_TYPES:
        mojo_base = base_type
    elif base_type in PLATFORM_TYPES:
        mojo_base = PLATFORM_TYPES[base_type]
    elif base_type in all_type_names:
        mojo_base = base_type
    else:
        mojo_base = "UInt8"

    if member.is_double_pointer:
        inner = f"UnsafePointer[{mojo_base}, MutAnyOrigin]"
        return f"UnsafePointer[{inner}, MutAnyOrigin]"
    elif member.is_pointer:
        return f"UnsafePointer[{mojo_base}, MutAnyOrigin]"

    if member.array_size:
        size = member.array_size
        if not size.isdigit():
            if size in api_constants:
                size = str(api_constants[size])
            else:
                raise ValueError(f"Unknown array size constant: {size}")
        return f"InlineArray[{mojo_base}, {size}]"

    return mojo_base


def _bitfield_width(type_name: str, all_type_names: dict[str, str]) -> int:
    width_map = {
        "uint8_t": 8,
        "uint16_t": 16,
        "uint32_t": 32,
        "uint64_t": 64,
        "int8_t": 8,
        "int16_t": 16,
        "int32_t": 32,
        "int64_t": 64,
    }
    if type_name in width_map:
        return width_map[type_name]
    cat = all_type_names.get(type_name, "")
    if cat == "bitmask":
        return 32
    if type_name in ("VkBool32", "VkFlags", "VkSampleMask"):
        return 32
    if type_name in ("VkFlags64", "VkDeviceSize", "VkDeviceAddress"):
        return 64
    return 32


# ===--- Struct/union generation ---=== #


def generate_struct_mojo(struct, all_type_names, api_constants) -> list[str]:
    lines = []
    lines.append("@fieldwise_init")
    lines.append(f"struct {struct.name}(Copyable, Movable):")
    lines.append(f'    """Vulkan struct {struct.name}."""')

    bitfield_group_idx = 0
    bitfield_accum = 0
    bitfield_width = 0
    in_bitfield = False

    for m in struct.members:
        if m.bitwidth is not None:
            member_width = _bitfield_width(m.type_name, all_type_names)
            if not in_bitfield or member_width != bitfield_width:
                if in_bitfield:
                    bitfield_group_idx += 1
                in_bitfield = True
                bitfield_width = member_width
                bitfield_accum = m.bitwidth
                width_to_mojo = {8: "UInt8", 16: "UInt16", 32: "UInt32", 64: "UInt64"}
                mojo_type = width_to_mojo.get(member_width, "UInt32")
                lines.append(f"    var _bitfield_{bitfield_group_idx}: {mojo_type}")
            else:
                bitfield_accum += m.bitwidth
                if bitfield_accum >= bitfield_width:
                    bitfield_group_idx += 1
                    in_bitfield = False
                    bitfield_accum = 0
                    bitfield_width = 0
        else:
            if in_bitfield:
                bitfield_group_idx += 1
                in_bitfield = False
                bitfield_accum = 0
                bitfield_width = 0
            mojo_type = mojo_type_for_member(m, all_type_names, api_constants)
            lines.append(f"    var {m.name}: {mojo_type}")

    lines.append("")
    return lines


def generate_union_mojo(union, union_sizes) -> list[str]:
    lines = []
    size = union_sizes.get(union.name, 64)

    if size % 8 == 0:
        elem_type = "Int64"
        elem_count = size // 8
    elif size % 4 == 0:
        elem_type = "Int32"
        elem_count = size // 4
    elif size % 2 == 0:
        elem_type = "Int16"
        elem_count = size // 2
    else:
        elem_type = "UInt8"
        elem_count = size

    lines.append("@fieldwise_init")
    lines.append(f"struct {union.name}(Copyable, Movable):")
    lines.append(
        f'    """Vulkan union {union.name}. Opaque byte array (sizeof={size})."""'
    )
    lines.append(f"    var _data: InlineArray[{elem_type}, {elem_count}]")
    lines.append("")
    return lines


def measure_union_sizes(unions: list[StructDef]) -> dict[str, int]:
    """Compile and run a C sizeof program for each union individually."""
    sizes: dict[str, int] = {}
    failed: list[str] = []

    for u in unions:
        c_src = (
            "#include <stdio.h>\n"
            '#include "vulkan/vulkan_core.h"\n'
            f'int main() {{ printf("%zu\\n", sizeof({u.name})); return 0; }}\n'
        )
        c_file = ""
        out_file = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
                f.write(c_src)
                c_file = f.name
            out_file = c_file.replace(".c", "")
            subprocess.run(
                ["gcc", "-o", out_file, c_file, f"-I{VULKAN_HEADERS}"],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [out_file],
                check=True,
                capture_output=True,
                text=True,
            )
            sizes[u.name] = int(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            failed.append(u.name)
        finally:
            for path in (c_file, out_file):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    for name in failed:
        sizes[name] = 8
        print(f"    {name}: estimated 8 (not in headers)")

    return sizes


# ===--- Commands ---=== #


def parse_command_param(p: ET.Element) -> CommandParam | None:
    type_el = p.find("type")
    name_el = p.find("name")
    if type_el is None or name_el is None:
        return None
    type_name = type_el.text or ""
    param_name = name_el.text or ""
    text_before = p.text or ""
    type_tail = type_el.tail or ""
    is_const = "const" in text_before
    ptr_count = type_tail.count("*")
    return CommandParam(param_name, type_name, ptr_count, is_const)


def extract_commands(root: ET.Element):
    commands = []
    aliases = []
    for cmd in root.findall("commands/command"):
        if cmd.get("api", "") == "vulkansc":
            continue
        alias = cmd.get("alias")
        if alias:
            aliases.append((cmd.get("name", ""), alias))
            continue
        proto = cmd.find("proto")
        if proto is None:
            continue
        name_el = proto.find("name")
        type_el = proto.find("type")
        if name_el is None:
            continue
        cmd_name = name_el.text or ""
        ret_type = type_el.text if type_el is not None else "void"
        ret_type = ret_type or "void"
        return_is_void = ret_type == "void"
        params = []
        for p in cmd.findall("param"):
            if p.get("api", "") == "vulkansc":
                continue
            parsed = parse_command_param(p)
            if parsed:
                params.append(parsed)
        commands.append(CommandDef(cmd_name, ret_type, return_is_void, params))
    return commands, aliases


def classify_command(cmd: CommandDef) -> str:
    if cmd.name in GLOBAL_COMMANDS:
        return "global"
    if not cmd.params:
        return "global"
    first_type = cmd.params[0].type_name
    if first_type in INSTANCE_LEVEL_TYPES:
        return "instance"
    elif first_type in DEVICE_LEVEL_TYPES:
        return "device"
    return "global"


def mojo_param_type(param: CommandParam, known_types: set[str]) -> str:
    base = param.type_name
    if base in C_TO_MOJO:
        mojo_base = C_TO_MOJO[base]
    elif base in known_types:
        mojo_base = base
    else:
        mojo_base = "NoneType" if param.ptr_count > 0 else "UInt64"

    if param.ptr_count >= 2:
        inner = f"UnsafePointer[{mojo_base}, MutAnyOrigin]"
        return f"UnsafePointer[{inner}, MutAnyOrigin]"
    elif param.ptr_count == 1:
        return f"UnsafePointer[{mojo_base}, MutAnyOrigin]"
    return mojo_base


def mojo_return_type(ret_type: str, known_types: set[str]) -> str:
    if ret_type == "void":
        return ""
    if ret_type in C_TO_MOJO:
        return C_TO_MOJO[ret_type]
    if ret_type in known_types:
        return ret_type
    return "UInt64"


def generate_command_type(cmd: CommandDef, known_types: set[str]) -> str:
    param_strs = []
    for p in cmd.params:
        mojo_type = mojo_param_type(p, known_types)
        param_strs.append(f"{p.mojo_name}: {mojo_type}")
    params_joined = ", ".join(param_strs)
    ret = mojo_return_type(cmd.return_type, known_types)
    if ret:
        return f"comptime {cmd.name} = fn({params_joined}) -> {ret}"
    else:
        return f"comptime {cmd.name} = fn({params_joined}) -> None"


def generate_commands(commands, aliases, known_types) -> list[str]:
    lines = []
    lines.append("# ========= COMMANDS =========")
    lines.append(
        f"# {len(commands)} command type declarations + {len(aliases)} aliases"
    )
    lines.append("#")
    lines.append(
        "# Each command is a `comptime` type alias for its C function signature."
    )
    lines.append("# The loader will cast loaded function pointers to these types.")
    lines.append("")

    for cmd in sorted(commands, key=lambda c: c.name):
        lines.append(generate_command_type(cmd, known_types))

    lines.append("")
    lines.append("# ========= COMMAND ALIASES =========")
    lines.append(f"# {len(aliases)} command aliases")
    lines.append("")
    for alias_name, target_name in sorted(aliases, key=lambda x: x[0]):
        lines.append(f"comptime {alias_name} = {target_name}")
    lines.append("")

    return lines


# ===--- Loader infrastructure ---=== #


def generate_loader() -> list[str]:
    lines = []
    lines.append("# ========= LOADER INFRASTRUCTURE =========")
    lines.append("")
    lines.append("comptime LoadProc = fn(var proc: String) raises -> fn() -> None")
    lines.append("comptime FuncPtr = ImmutOpaquePointer[ImmutExternalOrigin]")
    lines.append("")
    lines.append("fn _init_empty_table() -> Dict[String, FuncPtr]:")
    lines.append("    return Dict[String, FuncPtr]()")
    lines.append('comptime func_table = _Global["vk_table", _init_empty_table]()')
    lines.append("")
    lines.append("")
    lines.append("@always_inline")
    lines.append("fn load_fn_ptr(name: String, load: LoadProc) raises -> FuncPtr:")
    lines.append("    var func = load(name)")
    lines.append("    var addr = UnsafePointer(to=func).bitcast[FuncPtr]()[]")
    lines.append("    if not addr:")
    lines.append('        raise Error("Failed to load Vulkan function " + name)')
    lines.append("    return addr")
    lines.append("")
    lines.append("@always_inline")
    lines.append("fn try_load_fn_ptr(name: String, load: LoadProc) raises -> FuncPtr:")
    lines.append("    var func = load(name)")
    lines.append("    return UnsafePointer(to=func).bitcast[FuncPtr]()[]")
    lines.append("")
    lines.append("")
    lines.append("@always_inline")
    lines.append(
        "fn get_fn[fn_type: TrivialRegisterPassable, name: StaticString]() raises -> fn_type:"
    )
    lines.append("    var ptr = func_table.get_or_create_ptr()[][name]")
    lines.append("    if not ptr:")
    lines.append(
        '        raise Error("Vulkan function " + String(name) + " is not available on this platform")'
    )
    lines.append("    return UnsafePointer(to=ptr).bitcast[fn_type]()[]")
    lines.append("")
    return lines


def generate_init_function(
    name: str, commands: list[CommandDef], skip_missing: bool = True
) -> list[str]:
    lines = []
    lines.append(f"fn {name}(load: LoadProc) raises:")
    lines.append("    table = func_table.get_or_create_ptr()")
    load_fn = "try_load_fn_ptr" if skip_missing else "load_fn_ptr"
    for cmd in sorted(commands, key=lambda c: c.name):
        lines.append(f'    table[]["{cmd.name}"] = {load_fn}("{cmd.name}", load)')
    lines.append("")
    return lines


def to_snake_case(name: str) -> str:
    name = re.sub(r"(\d)D\b", r"_\1d", name)
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def vk_to_snake(name: str) -> str:
    return to_snake_case(name.removeprefix("vk"))


def generate_wrapper_fn(cmd: CommandDef, known_types: set[str]) -> list[str]:
    snake_name = vk_to_snake(cmd.name)
    param_strs = []
    call_args = []
    for p in cmd.params:
        mojo_type = mojo_param_type(p, known_types)
        param_strs.append(f"{p.mojo_name}: {mojo_type}")
        call_args.append(p.mojo_name)
    params_joined = ", ".join(param_strs)
    args_joined = ", ".join(call_args)
    ret = mojo_return_type(cmd.return_type, known_types)
    if ret:
        sig = f"fn {snake_name}({params_joined}) raises -> {ret}"
    else:
        sig = f"fn {snake_name}({params_joined}) raises -> None"
    body = f'    return get_fn[{cmd.name}, "{cmd.name}"]()({args_joined})'
    return [f"{sig}:", body, ""]


def generate_wrapper_alias(
    alias_name: str,
    target_name: str,
    target_cmd: "CommandDef | None" = None,
    known_types: "set[str] | None" = None,
) -> list[str]:
    if target_cmd is None:
        return []
    if known_types is None:
        known_types = set()
    alias_snake = vk_to_snake(alias_name)
    target_snake = vk_to_snake(target_name)
    param_strs = []
    call_args = []
    for p in target_cmd.params:
        mojo_type = mojo_param_type(p, known_types)
        param_strs.append(f"{p.mojo_name}: {mojo_type}")
        call_args.append(p.mojo_name)
    params_joined = ", ".join(param_strs)
    args_joined = ", ".join(call_args)
    ret = mojo_return_type(target_cmd.return_type, known_types)
    if ret:
        sig = f"fn {alias_snake}({params_joined}) raises -> {ret}"
    else:
        sig = f"fn {alias_snake}({params_joined}) raises -> None"
    body = f"    return {target_snake}({args_joined})"
    return [f"{sig}:", body, ""]


def generate_loader_section(commands, aliases, known_types) -> list[str]:
    """Generate loader infrastructure + init functions + wrappers."""
    global_cmds = []
    instance_cmds = []
    device_cmds = []

    for cmd in commands:
        phase = classify_command(cmd)
        if phase == "global":
            global_cmds.append(cmd)
        elif phase == "instance":
            instance_cmds.append(cmd)
        else:
            device_cmds.append(cmd)

    print("  Command classification:")
    print(f"    Global:   {len(global_cmds)}")
    print(f"    Instance: {len(instance_cmds)}")
    print(f"    Device:   {len(device_cmds)}")

    lines = []

    # Loader infrastructure
    lines.extend(generate_loader())

    # Init functions
    lines.append("# ========= INIT FUNCTIONS =========")
    lines.append("")
    lines.extend(
        generate_init_function("init_vulkan_global", global_cmds, skip_missing=False)
    )
    lines.extend(
        generate_init_function("init_vulkan_instance", instance_cmds, skip_missing=True)
    )
    lines.extend(
        generate_init_function("init_vulkan_device", device_cmds, skip_missing=True)
    )

    # Top-level init
    lines.append("# ========= INIT =========")
    lines.append(
        "fn init_vulkan(global_load: LoadProc, instance_load: LoadProc, device_load: LoadProc) raises:"
    )
    lines.append("    init_vulkan_global(global_load)")
    lines.append("    init_vulkan_instance(instance_load)")
    lines.append("    init_vulkan_device(device_load)")
    lines.append("")

    # Wrapper functions
    lines.append("# ========= WRAPPER FUNCTIONS =========")
    lines.append(f"# {len(commands)} wrapper functions")
    lines.append("")
    for cmd in sorted(commands, key=lambda c: c.name):
        lines.extend(generate_wrapper_fn(cmd, known_types))

    # Wrapper aliases
    lines.append("# ========= WRAPPER ALIASES =========")
    lines.append(f"# {len(aliases)} command alias wrappers")
    lines.append("")
    cmd_by_name = {cmd.name: cmd for cmd in commands}
    for alias_name, target_name in sorted(aliases, key=lambda x: x[0]):
        target_cmd = cmd_by_name.get(target_name)
        if target_cmd is None:
            continue
        lines.extend(
            generate_wrapper_alias(alias_name, target_name, target_cmd, known_types)
        )

    return lines


# ===--- S2 Target resolution ---=== #

_VK_VERSION_RE = re.compile(r"^VK_VERSION_(\d+)_(\d+)$")


def _split_depends_tokens(depends_str: str) -> tuple[str, ...]:
    """Split a vk.xml depends expression into normalized tokens.

    Supports AND (`+`) and OR (`,`) separators, trims whitespace, and strips
    wrapper parentheses from each token fragment.

    Args:
        depends_str: Raw depends attribute value.

    Returns:
        Tuple of normalized token strings in input order.
    """
    stripped = depends_str.strip()
    if not stripped:
        return ()

    tokens: list[str] = []
    for raw_token in re.split(r"[+,]", stripped):
        token = raw_token.strip().strip("() ")
        if token:
            tokens.append(token)
    return tuple(tokens)


def _supports_vulkan_api(api_value: str) -> bool:
    """Return True when a comma-separated api/supported value includes vulkan.

    Args:
        api_value: Raw vk.xml API selector value.

    Returns:
        True when `vulkan` is present as one of the comma-separated values.
    """
    return any(token.strip() == "vulkan" for token in api_value.split(","))


def parse_extension_depends(depends_str: str) -> frozenset[str]:
    """Parse a vk.xml extension depends string into a flat set of extension names.

    Flattens both AND (+) and OR (,) operators into a single set, collecting all
    referenced extension and version names. This is conservative: every token is
    treated as a dependency regardless of the AND/OR grouping structure.

    Args:
        depends_str: Raw depends attribute value, e.g. "VK_A+VK_B,VK_C".

    Returns:
        Frozenset of all unique token strings, or empty frozenset for blank input.
    """
    deps = {
        token
        for token in _split_depends_tokens(depends_str)
        if not _VK_VERSION_RE.match(token)
    }
    return frozenset(deps)


def all_vulkan_extension_names(root: ET.Element) -> frozenset[str]:
    """Return all extension names that support the vulkan API.

    Includes extensions with supported="vulkan" or supported="vulkan,vulkansc".
    Excludes vulkansc-only and disabled extensions.

    Args:
        root: Registry XML root element.

    Returns:
        Frozenset of extension name strings.
    """
    names: set[str] = set()
    for ext in root.findall("extensions/extension"):
        supported = ext.get("supported", "")
        if _supports_vulkan_api(supported):
            name = ext.get("name")
            if name:
                names.add(name)
    return frozenset(names)


def resolve_extension_deps(
    root: ET.Element,
    requested: frozenset[str],
) -> frozenset[str]:
    """Resolve extension dependencies transitively to a closed set.

    Walks the depends attribute on each extension element, expanding the
    requested set until no new names can be added. Unknown extension names
    pass through unchanged (no error raised). Cycle-safe via visited tracking.

    Args:
        root: Registry XML root element.
        requested: Initial set of requested extension names.

    Returns:
        Frozenset containing requested names plus all transitive dependencies.
    """
    if not requested:
        return frozenset()

    dep_map: dict[str, frozenset[str]] = {}
    for ext in root.findall("extensions/extension"):
        name = ext.get("name")
        if name:
            dep_map[name] = parse_extension_depends(ext.get("depends", ""))

    resolved: set[str] = set(requested)
    frontier: set[str] = set(requested)

    while frontier:
        new_frontier: set[str] = set()
        for ext_name in frontier:
            for dep in dep_map.get(ext_name, frozenset()):
                if dep not in resolved:
                    resolved.add(dep)
                    new_frontier.add(dep)
        frontier = new_frontier

    return frozenset(resolved)


def _require_block_satisfied(depends: str, version: VulkanVersion) -> bool:
    """Return True if a require block's depends condition is met by version.

    Checks VK_VERSION_X_Y tokens: all such tokens must be satisfied by the
    current version (version >= required). Non-version tokens are treated as
    satisfied (extension presence is managed by resolution, not here).

    Args:
        depends: Raw depends attribute value from a require element.
        version: Current target Vulkan version.

    Returns:
        True if the condition is satisfied or absent, False otherwise.
    """
    stripped = depends.strip()
    if not stripped:
        return True
    for token in _split_depends_tokens(stripped):
        m = _VK_VERSION_RE.match(token)
        if m:
            required = VulkanVersion(int(m.group(1)), int(m.group(2)))
            if version < required:
                return False
    return True


def _extension_promoted_to_version(ext: ET.Element, version: VulkanVersion) -> bool:
    """Return True if the extension has been promoted into the given version or earlier.

    Checks the promotedto="VK_VERSION_X_Y" attribute. Extensions promoted to
    a version <= the target version are already part of core and contribute no
    new types or commands beyond what version feature blocks already provide.

    Args:
        ext: Extension XML element.
        version: Current target Vulkan version.

    Returns:
        True if the extension is promoted to a version at or below version.
    """
    promotedto = ext.get("promotedto", "")
    if not promotedto:
        return False
    m = _VK_VERSION_RE.match(promotedto)
    if not m:
        return False
    promoted_version = VulkanVersion(int(m.group(1)), int(m.group(2)))
    return promoted_version <= version


def collect_version_types(root: ET.Element, version: VulkanVersion) -> frozenset[str]:
    """Collect all type names from Vulkan feature elements up to and including version.

    Iterates feature elements with api="vulkan", skipping entries for higher versions
    and silently dropping entries missing a name attribute.

    Args:
        root: Registry XML root element.
        version: Maximum version to include (inclusive).

    Returns:
        Frozenset of type name strings.
    """
    types: set[str] = set()
    for feat in root.findall("feature"):
        if not _supports_vulkan_api(feat.get("api", "")):
            continue
        number_str = feat.get("number", "0.0")
        major_s, _, minor_s = number_str.partition(".")
        feat_version = VulkanVersion(int(major_s), int(minor_s or "0"))
        if feat_version > version:
            continue
        for req in feat.findall("require"):
            for t in req.findall("type"):
                name = t.get("name")
                if name:
                    types.add(name)
    return frozenset(types)


def collect_version_commands(
    root: ET.Element,
    version: VulkanVersion,
) -> frozenset[str]:
    """Collect all command names from Vulkan feature elements up to and including version.

    Mirrors collect_version_types but for command elements. Silently drops
    entries missing a name attribute.

    Args:
        root: Registry XML root element.
        version: Maximum version to include (inclusive).

    Returns:
        Frozenset of command name strings.
    """
    commands: set[str] = set()
    for feat in root.findall("feature"):
        if not _supports_vulkan_api(feat.get("api", "")):
            continue
        number_str = feat.get("number", "0.0")
        major_s, _, minor_s = number_str.partition(".")
        feat_version = VulkanVersion(int(major_s), int(minor_s or "0"))
        if feat_version > version:
            continue
        for req in feat.findall("require"):
            for cmd in req.findall("command"):
                name = cmd.get("name")
                if name:
                    commands.add(name)
    return frozenset(commands)


def collect_extension_types(
    root: ET.Element,
    extensions: frozenset[str],
    version: VulkanVersion,
) -> frozenset[str]:
    """Collect type names from extension require blocks for the given extension set.

    Only processes extensions present in the requested set that support the vulkan
    API. Conditional require blocks (depends="VK_VERSION_X_Y") are evaluated
    against version and skipped when unsatisfied.

    Args:
        root: Registry XML root element.
        extensions: Resolved set of extension names to collect from.
        version: Current target Vulkan version for conditional require evaluation.

    Returns:
        Frozenset of type name strings.
    """
    if not extensions:
        return frozenset()

    types: set[str] = set()
    for ext in root.findall("extensions/extension"):
        name = ext.get("name", "")
        if name not in extensions:
            continue
        if not _supports_vulkan_api(ext.get("supported", "")):
            continue
        if _extension_promoted_to_version(ext, version):
            continue
        for req in ext.findall("require"):
            if not _require_block_satisfied(req.get("depends", ""), version):
                continue
            for t in req.findall("type"):
                type_name = t.get("name")
                if type_name:
                    types.add(type_name)
    return frozenset(types)


def collect_extension_commands(
    root: ET.Element,
    extensions: frozenset[str],
    version: VulkanVersion,
) -> frozenset[str]:
    """Collect command names from extension require blocks for the given extension set.

    Mirrors collect_extension_types but for command elements. Conditional require
    blocks are evaluated against version.

    Args:
        root: Registry XML root element.
        extensions: Resolved set of extension names to collect from.
        version: Current target Vulkan version for conditional require evaluation.

    Returns:
        Frozenset of command name strings.
    """
    if not extensions:
        return frozenset()

    commands: set[str] = set()
    for ext in root.findall("extensions/extension"):
        name = ext.get("name", "")
        if name not in extensions:
            continue
        if not _supports_vulkan_api(ext.get("supported", "")):
            continue
        if _extension_promoted_to_version(ext, version):
            continue
        for req in ext.findall("require"):
            if not _require_block_satisfied(req.get("depends", ""), version):
                continue
            for cmd in req.findall("command"):
                cmd_name = cmd.get("name")
                if cmd_name:
                    commands.add(cmd_name)
    return frozenset(commands)


# ===--- S2 Type closure and target assembly ---=== #

_TYPE_CLOSURE_SAFETY_LIMIT = 1000


@dataclass(frozen=True)
class TargetSet:
    """Compile-safe set of types and commands for a given version and extension selection.

    Both fields contain only registry-known names that are safe to emit together:
    the type set is transitively closed over non-pointer struct dependencies so
    every value-embedded struct is included alongside its container.
    """

    types: frozenset[str]
    commands: frozenset[str]


@dataclass(frozen=True)
class TypeClosureStats:
    """Diagnostics from a single close_type_deps run.

    Attributes:
        initial_count: Number of types in the seed set before closure.
        final_count: Number of types in the closed set after closure.
        added_types: Frozenset of type names added during closure (not in seed).
        iteration_count: Number of iterations required to reach fixed point.
    """

    initial_count: int
    final_count: int
    added_types: frozenset[str]
    iteration_count: int


def close_type_deps(
    initial_types: frozenset[str],
    structs: list[StructDef],
    all_type_names: dict[str, str],
) -> tuple[frozenset[str], TypeClosureStats]:
    """Expand an initial type set to a compile-safe closure over non-pointer deps.

    Mirrors the non-pointer dependency semantics used by topo_sort_structs: a
    member contributes a dependency only when it is not a pointer (value-embedded).
    Iterates until fixed point. Raises RuntimeError if iteration count exceeds
    _TYPE_CLOSURE_SAFETY_LIMIT (safety backstop against infinite expansion).

    Only types present in all_type_names are added during closure; primitive C
    types and unknown names are silently skipped.

    Args:
        initial_types: Seed set of type names to close.
        structs: Struct definitions to traverse for member dependencies.
        all_type_names: Registry type map used to filter which deps are known.

    Returns:
        Tuple of (closed frozenset, TypeClosureStats).

    Raises:
        RuntimeError: If iteration count exceeds _TYPE_CLOSURE_SAFETY_LIMIT.
    """
    struct_map = {s.name: s for s in structs}
    current: set[str] = set(initial_types)
    added: set[str] = set()
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > _TYPE_CLOSURE_SAFETY_LIMIT:
            raise RuntimeError(
                f"Type closure exceeded safety limit of {_TYPE_CLOSURE_SAFETY_LIMIT} iterations"
            )

        new_types: set[str] = set()
        for type_name in list(current):
            struct = struct_map.get(type_name)
            if struct is None:
                continue
            for member in struct.members:
                if member.is_pointer:
                    continue
                dep = member.type_name
                if dep in all_type_names and dep not in current:
                    new_types.add(dep)

        if not new_types:
            break

        current.update(new_types)
        added.update(new_types)

    closed = frozenset(current)
    stats = TypeClosureStats(
        initial_count=len(initial_types),
        final_count=len(closed),
        added_types=frozenset(added),
        iteration_count=iteration_count,
    )
    return closed, stats


def build_target_set(
    root: ET.Element,
    version: VulkanVersion,
    resolved_extensions: frozenset[str],
    structs: list[StructDef],
    all_type_names: dict[str, str],
) -> tuple[TargetSet, TypeClosureStats]:
    """Assemble a compile-safe TargetSet from version and extension selections.

    Collects core types and commands up to version, extension types and commands
    for the resolved extension set, then expands the type set via close_type_deps
    to ensure every non-pointer struct dependency is included.

    Args:
        root: Registry XML root element.
        version: Target Vulkan version (inclusive upper bound for feature blocks).
        resolved_extensions: Closed set of extension names (already transitively resolved).
        structs: Struct definitions for dependency closure traversal.
        all_type_names: Registry type map for filtering closure candidates.

    Returns:
        Tuple of (TargetSet, TypeClosureStats).

    Raises:
        RuntimeError: Propagated from close_type_deps if safety limit is breached.
    """
    version_types = collect_version_types(root, version)
    ext_types = collect_extension_types(root, resolved_extensions, version)
    version_commands = collect_version_commands(root, version)
    ext_commands = collect_extension_commands(root, resolved_extensions, version)

    initial_types = version_types | ext_types
    closed_types, stats = close_type_deps(initial_types, structs, all_type_names)

    target = TargetSet(
        types=closed_types,
        commands=version_commands | ext_commands,
    )
    return target, stats


TypeT = TypeVar("TypeT")


def filter_by_target(
    items: list[TypeT],
    target_set: frozenset[str],
    key: Callable[[TypeT], str],
) -> list[TypeT]:
    """Return a new list of items whose key is a member of target_set.

    Preserves the original input order. Non-mutating.

    Args:
        items: Sequence of items to filter.
        target_set: Set of names to include.
        key: Function mapping each item to its name string.

    Returns:
        New list containing only items whose key is in target_set.
    """
    return [item for item in items if key(item) in target_set]


def filter_aliases_by_target(
    aliases: list[tuple[str, str]],
    target_set: frozenset[str],
) -> list[tuple[str, str]]:
    """Return alias pairs where either the alias name or the target name is in target_set.

    Preserves input order. Non-mutating.

    Args:
        aliases: Sequence of (alias_name, target_name) pairs.
        target_set: Set of names to include.

    Returns:
        New list of alias pairs where at least one side is in target_set.
    """
    return [
        (alias, target)
        for alias, target in aliases
        if alias in target_set or target in target_set
    ]


# ===--- S3 Discovery commands ---=== #


ALL_VULKAN_VERSIONS: tuple[VulkanVersion, ...] = (
    VulkanVersion(1, 0),
    VulkanVersion(1, 1),
    VulkanVersion(1, 2),
    VulkanVersion(1, 3),
    VulkanVersion(1, 4),
)
"""Ordered sequence of all known Vulkan API versions, lowest to highest.

Used by gather_version_summaries to compute deltas in correct order.
Mirrors VALID_VERSIONS but as an ordered tuple of VulkanVersion values.
"""


@dataclass(frozen=True)
class VersionSummary:
    """One row of the --list-versions table.

    delta_type_count and delta_command_count are the types/commands introduced
    at exactly this version (not cumulative). For version 1.0 (the base),
    cumulative == delta since there is no prior version.

    Attributes:
        version: The Vulkan version this row represents.
        delta_type_count: Types added at this specific version.
        delta_command_count: Commands added at this specific version.
        cumulative_type_count: Total types up to and including this version.
        cumulative_command_count: Total commands up to and including this version.
        is_base: True only for the first (lowest) Vulkan version (1.0).
    """

    version: VulkanVersion
    delta_type_count: int
    delta_command_count: int
    cumulative_type_count: int
    cumulative_command_count: int
    is_base: bool


@dataclass(frozen=True)
class ExtensionSummary:
    """One row of the --list-extensions table.

    depends_raw is the raw vk.xml depends= attribute value (may be empty string).
    It is stored uninterpreted so both the one-line display (truncated) and
    full detail display can derive their own formatting from it.

    promoted_to is the raw vk.xml promotedto= attribute value
    (e.g. "VK_VERSION_1_3") or None if the extension is not promoted.

    Attributes:
        name: Extension name, e.g. "VK_KHR_swapchain".
        ext_type: "device", "instance", or "" if unspecified in vk.xml.
        type_count: Number of distinct type names in the extension's require blocks.
        command_count: Number of distinct command names in the extension's require blocks.
        depends_raw: Raw depends= attribute value from vk.xml, or "" if absent.
        promoted_to: Raw promotedto= value, or None if not promoted.
    """

    name: str
    ext_type: str
    type_count: int
    command_count: int
    depends_raw: str
    promoted_to: str | None


@dataclass(frozen=True)
class TypeEntry:
    """A single type name and its category, used in --info output.

    Attributes:
        name: Type name, e.g. "VkSwapchainKHR".
        category: vk.xml category string: "struct", "handle", "enum",
                  "bitmask", "funcpointer", or "".
    """

    name: str
    category: str


@dataclass(frozen=True)
class ExtensionDetail:
    """Full --info output for one extension.

    types is ordered as they appear in the extension's require blocks.
    commands is ordered as they appear in the extension's require blocks.
    Duplicates are excluded; first occurrence wins.

    Attributes:
        summary: The ExtensionSummary for this extension.
        types: Ordered type entries from the extension's require blocks.
        commands: Ordered command names from the extension's require blocks.
    """

    summary: ExtensionSummary
    types: tuple[TypeEntry, ...]
    commands: tuple[str, ...]


# ===--- Internal extension helpers ---=== #


def _get_extension_type(ext: ET.Element) -> str:
    """Return the type= attribute value or "" if absent.

    Args:
        ext: Extension XML element.

    Returns:
        Extension type string, or "" if not present.
    """
    return ext.get("type", "")


def _get_extension_promoted_to(ext: ET.Element) -> str | None:
    """Return the promotedto= attribute value or None if absent.

    Args:
        ext: Extension XML element.

    Returns:
        promotedto string, or None if not present.
    """
    value = ext.get("promotedto")
    return value if value else None


def _count_extension_types_raw(ext: ET.Element) -> int:
    """Count distinct type names in an extension's own require blocks (not version-gated).

    Args:
        ext: Extension XML element.

    Returns:
        Count of distinct type names across all require blocks.
    """
    seen: set[str] = set()
    for req in ext.findall("require"):
        for t in req.findall("type"):
            name = t.get("name")
            if name:
                seen.add(name)
    return len(seen)


def _count_extension_commands_raw(ext: ET.Element) -> int:
    """Count distinct command names in an extension's own require blocks (not version-gated).

    Args:
        ext: Extension XML element.

    Returns:
        Count of distinct command names across all require blocks.
    """
    seen: set[str] = set()
    for req in ext.findall("require"):
        for cmd in req.findall("command"):
            name = cmd.get("name")
            if name:
                seen.add(name)
    return len(seen)


# ===--- Registry metadata ---=== #


def extract_registry_version(root: ET.Element) -> str:
    """Return the vk.xml version string, e.g. "1.4.343".

    The version string is derived from two sources in the registry:
      - Major.minor: highest <feature api="vulkan" number="X.Y"> in the registry.
      - Patch: VK_HEADER_VERSION constant from the API Constants enum block.

    Returns "<major>.<minor>.<patch>" when both sources are present.
    Returns "<major>.<minor>" when VK_HEADER_VERSION is absent.
    Returns "unknown" when no vulkan feature elements exist.

    Args:
        root: Registry XML root element.

    Returns:
        Version display string, e.g. "1.4.343".
    """
    best: VulkanVersion | None = None
    for feat in root.findall("feature"):
        if not _supports_vulkan_api(feat.get("api", "")):
            continue
        number_str = feat.get("number", "")
        if not number_str:
            continue
        major_s, _, minor_s = number_str.partition(".")
        try:
            version = VulkanVersion(int(major_s), int(minor_s or "0"))
        except ValueError:
            continue
        if best is None or version > best:
            best = version

    if best is None:
        return "unknown"

    constants = load_api_constants(root)
    patch = constants.get("VK_HEADER_VERSION")
    if patch is None:
        return f"{best.major}.{best.minor}"
    return f"{best.major}.{best.minor}.{patch}"


# ===--- Data extractors ---=== #


def gather_version_summaries(root: ET.Element) -> list[VersionSummary]:
    """Return one VersionSummary per entry in ALL_VULKAN_VERSIONS, in order.

    Each summary includes the cumulative type/command count at that version and
    the delta (types/commands added at exactly that version vs. the prior one).
    Counts are derived from collect_version_types and collect_version_commands.

    The returned list always has exactly len(ALL_VULKAN_VERSIONS) entries,
    one per version, in ascending version order.

    Args:
        root: Registry XML root element.

    Returns:
        List of VersionSummary, one per known Vulkan version, in ascending order.
    """
    summaries: list[VersionSummary] = []
    prev_type_count = 0
    prev_command_count = 0

    for index, version in enumerate(ALL_VULKAN_VERSIONS):
        types = collect_version_types(root, version)
        commands = collect_version_commands(root, version)
        cumulative_types = len(types)
        cumulative_commands = len(commands)
        delta_types = cumulative_types - prev_type_count
        delta_commands = cumulative_commands - prev_command_count

        summaries.append(
            VersionSummary(
                version=version,
                delta_type_count=delta_types,
                delta_command_count=delta_commands,
                cumulative_type_count=cumulative_types,
                cumulative_command_count=cumulative_commands,
                is_base=(index == 0),
            )
        )

        prev_type_count = cumulative_types
        prev_command_count = cumulative_commands

    return summaries


def gather_extension_summaries(root: ET.Element) -> list[ExtensionSummary]:
    """Return one ExtensionSummary per Vulkan-supporting extension, sorted by name.

    Only includes extensions where supported= includes "vulkan" (uses existing
    _supports_vulkan_api). Counts reflect the extension's own require blocks
    (not transitive dependencies and not version-gated — raw spec content).

    The returned list is sorted alphabetically by extension name.

    Args:
        root: Registry XML root element.

    Returns:
        Alphabetically sorted list of ExtensionSummary, one per Vulkan extension.
    """
    summaries: list[ExtensionSummary] = []
    for ext in root.findall("extensions/extension"):
        supported = ext.get("supported", "")
        if not _supports_vulkan_api(supported):
            continue
        name = ext.get("name", "")
        if not name:
            continue
        summaries.append(
            ExtensionSummary(
                name=name,
                ext_type=_get_extension_type(ext),
                type_count=_count_extension_types_raw(ext),
                command_count=_count_extension_commands_raw(ext),
                depends_raw=ext.get("depends", ""),
                promoted_to=_get_extension_promoted_to(ext),
            )
        )
    summaries.sort(key=lambda s: s.name)
    return summaries


def filter_extensions_by_text(
    summaries: list[ExtensionSummary],
    filter_text: str,
) -> list[ExtensionSummary]:
    """Return summaries whose name contains filter_text as a case-insensitive substring.

    Preserves input order. Non-mutating. Empty filter_text returns all summaries
    unchanged (equivalent to no filter).

    Args:
        summaries: Sequence of extension summaries to filter.
        filter_text: Substring to match against extension names (case-insensitive).

    Returns:
        New list containing only summaries whose name contains filter_text.
    """
    if not filter_text:
        return list(summaries)
    needle = filter_text.lower()
    return [s for s in summaries if needle in s.name.lower()]


def gather_extension_detail(
    root: ET.Element,
    extension_name: str,
) -> ExtensionDetail | None:
    """Return full detail for a named extension, or None if not found in the registry.

    The extension_name must be a valid extension name format (already enforced by
    S1 validate_extension_name before reaching this function). This function performs
    the runtime lookup: returns None when the name is absent from the registry.

    Type categories are resolved from load_all_type_names(root). Types and commands
    are ordered by first appearance in require blocks; duplicates excluded.

    Args:
        root: Registry XML root element.
        extension_name: Extension name to look up, e.g. "VK_KHR_swapchain".

    Returns:
        ExtensionDetail for the named extension, or None if not in the registry.
    """
    target: ET.Element | None = None
    for ext in root.findall("extensions/extension"):
        if ext.get("name") == extension_name:
            target = ext
            break

    if target is None:
        return None

    all_type_names = load_all_type_names(root)

    seen_types: dict[str, None] = {}  # ordered set via insertion-order dict
    seen_commands: dict[str, None] = {}

    for req in target.findall("require"):
        for t in req.findall("type"):
            name = t.get("name")
            if name and name not in seen_types:
                seen_types[name] = None
        for cmd in req.findall("command"):
            name = cmd.get("name")
            if name and name not in seen_commands:
                seen_commands[name] = None

    type_entries = tuple(
        TypeEntry(name=name, category=all_type_names.get(name, ""))
        for name in seen_types
    )
    command_names = tuple(seen_commands)

    summary = ExtensionSummary(
        name=extension_name,
        ext_type=_get_extension_type(target),
        type_count=len(type_entries),
        command_count=len(command_names),
        depends_raw=target.get("depends", ""),
        promoted_to=_get_extension_promoted_to(target),
    )

    return ExtensionDetail(summary=summary, types=type_entries, commands=command_names)


# ===--- S3 Formatters ---=== #


def format_versions_table(
    summaries: list[VersionSummary],
    registry_version: str,
) -> str:
    """Return the complete --list-versions output as a string.

    Output format (matches spec.md:81):

        Vulkan versions in vk.xml {registry_version}:

          1.0    538 types    215 commands    (base)
          1.1    +104 types   +28 commands    (642 total)
          ...

    The base version (is_base=True) displays without "+" prefix. All other
    versions display delta counts with "+" prefix and cumulative totals.
    Column widths are fixed-format (not dynamic from data).

    Args:
        summaries: Version summaries in ascending version order.
        registry_version: Registry version string, e.g. "1.4.343".

    Returns:
        Formatted multi-line string including trailing newline.
    """
    lines = [f"Vulkan versions in vk.xml {registry_version}:", ""]
    for row in summaries:
        ver = row.version
        if row.is_base:
            type_col = f"{row.delta_type_count} types"
            cmd_col = f"{row.delta_command_count} commands"
            suffix = "(base)"
        else:
            type_col = f"+{row.delta_type_count} types"
            cmd_col = f"+{row.delta_command_count} commands"
            suffix = f"({row.cumulative_type_count} total)"
        lines.append(f"  {ver}    {type_col:<14} {cmd_col:<17} {suffix}")
    lines.append("")
    return "\n".join(lines)


def format_extensions_table(
    summaries: list[ExtensionSummary],
    registry_version: str,
) -> str:
    """Return the complete --list-extensions output as a single string.

    Output format (matches spec.md:95):

        {N} Vulkan extensions in vk.xml {registry_version}:

          VK_KHR_swapchain            device    13 types   9 cmds   depends: VK_KHR_surface
          ...

    Column widths for name and type are derived from the widest value in summaries.
    The depends/promoted trailing field is omitted when depends_raw is empty and
    promoted_to is None. Long depends_raw values are truncated with "...".

    Filtering is NOT applied here — callers must pre-filter with
    filter_extensions_by_text before calling this function.

    Args:
        summaries: Extension summaries to format (already filtered if applicable).
        registry_version: Registry version string, e.g. "1.4.343".

    Returns:
        Formatted multi-line string including trailing newline.
    """
    n = len(summaries)
    lines = [f"{n} Vulkan extensions in vk.xml {registry_version}:", ""]

    if not summaries:
        lines.append("")
        return "\n".join(lines)

    name_width = max((len(s.name) for s in summaries), default=0)
    type_width = max((len(s.ext_type) for s in summaries), default=0)

    for s in summaries:
        type_col = s.ext_type.ljust(type_width) if type_width else ""
        type_count_col = f"{s.type_count} types"
        cmd_count_col = f"{s.command_count} cmds"

        if s.promoted_to is not None:
            annotation = f"promoted: {s.promoted_to}"
        elif s.depends_raw:
            raw = s.depends_raw
            if len(raw) > 40:
                raw = raw[:37] + "..."
            annotation = f"depends: {raw}"
        else:
            annotation = ""

        name_col = s.name.ljust(name_width)
        row = f"  {name_col}  {type_col}  {type_count_col:<10} {cmd_count_col:<8}"
        if annotation:
            row = row.rstrip() + f"  {annotation}"
        lines.append(row.rstrip())

    lines.append("")
    return "\n".join(lines)


def format_extension_detail(detail: ExtensionDetail) -> str:
    """Return the complete --info output for one extension as a string.

    Output format (matches spec.md:109):

        VK_KHR_swapchain (device extension)
          Depends:  VK_KHR_surface
          Promoted: no

          Types (13):
            VkSwapchainKHR                     handle
            ...

          Commands (9):
            vkCreateSwapchainKHR
            ...

    Args:
        detail: ExtensionDetail for the extension to display.

    Returns:
        Formatted multi-line string including trailing newline.
    """
    s = detail.summary
    ext_type_label = f"{s.ext_type} extension" if s.ext_type else "extension"
    lines = [f"{s.name} ({ext_type_label})"]

    deps = parse_extension_depends(s.depends_raw)
    ext_deps = sorted(d for d in deps if not _VK_VERSION_RE.match(d))
    if ext_deps:
        lines.append(f"  Depends:  {', '.join(ext_deps)}")

    promoted_label = s.promoted_to if s.promoted_to is not None else "no"
    lines.append(f"  Promoted: {promoted_label}")

    lines.append("")
    lines.append(f"  Types ({len(detail.types)}):")
    name_width = max((len(e.name) for e in detail.types), default=0)
    for entry in detail.types:
        if entry.category:
            lines.append(f"    {entry.name.ljust(name_width)}  {entry.category}")
        else:
            lines.append(f"    {entry.name}")

    lines.append("")
    lines.append(f"  Commands ({len(detail.commands)}):")
    for cmd in detail.commands:
        lines.append(f"    {cmd}")

    lines.append("")
    return "\n".join(lines)


# ===--- S3 Dispatch ---=== #


def run_discovery(config: DiscoveryConfig) -> None:
    """Execute the discovery command specified in config.

    Loads vk.xml, branches on config.command, extracts data, formats output,
    and prints to stdout. On error conditions that are user-visible (e.g. unknown
    extension name for --info), prints an error message to stderr and exits with
    code 1.

    dispatch table:
      "list-versions"   → gather_version_summaries → format_versions_table → print
      "list-extensions" → gather_extension_summaries → [filter] → format_extensions_table → print
      "info"            → gather_extension_detail → [None check] → format_extension_detail → print

    Args:
        config: Validated DiscoveryConfig from build_config().

    Raises:
        SystemExit(1): When config.command == "info" and config.info_extension is
                       not found in the registry.

    Note:
        vk.xml parse errors (malformed XML) propagate as xml.etree.ElementTree.ParseError.
        These are truly exceptional and are not caught here.
    """
    import sys

    root = ET.parse(config.vk_xml).getroot()
    registry_version = extract_registry_version(root)

    if config.command == "list-versions":
        summaries = gather_version_summaries(root)
        output = format_versions_table(summaries, registry_version)
        print(output, end="")

    elif config.command == "list-extensions":
        summaries = gather_extension_summaries(root)
        if config.filter_text is not None:
            summaries = filter_extensions_by_text(summaries, config.filter_text)
        output = format_extensions_table(summaries, registry_version)
        print(output, end="")

    elif config.command == "info":
        assert (
            config.info_extension is not None
        )  # S1 guarantees this when command == "info"
        detail = gather_extension_detail(root, config.info_extension)
        if detail is None:
            print(
                f"Error: extension '{config.info_extension}' not found in vk.xml {registry_version}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        output = format_extension_detail(detail)
        print(output, end="")


# ===--- S4 Package writer ---=== #

# ===--- Module stem constants ---=== #
# Single source of truth for all module filenames.
# Use these everywhere — never hardcode "vk_structs" as a string literal.

MODULE_BASE_TYPES: str = "vk_base_types"
MODULE_ENUMS: str = "vk_enums"
MODULE_HANDLES: str = "vk_handles"
MODULE_STRUCTS: str = "vk_structs"
MODULE_UNIONS: str = "vk_unions"
MODULE_LOADER: str = "vk_loader"
MODULE_COMMANDS: str = "vk_commands"

MODULE_ORDER: tuple[str, ...] = (
    MODULE_BASE_TYPES,
    MODULE_ENUMS,
    MODULE_HANDLES,
    MODULE_STRUCTS,
    MODULE_UNIONS,
    MODULE_LOADER,
    MODULE_COMMANDS,
)
"""Declaration order for modules in __init__.mojo re-exports.

Mirrors the spec-defined file list (spec.md:172). Note that vk_structs
appears before vk_unions despite importing from vk_unions — Mojo resolves
package-internal deps at compile time regardless of re-export order."""

# ===--- Loader export constants ---=== #

LOADER_SELECTIVE_EXPORTS: tuple[str, ...] = (
    "LoadProc",
    "FuncPtr",
    "init_vulkan",
    "init_vulkan_global",
    "init_vulkan_instance",
    "init_vulkan_device",
)
"""Names selectively re-exported from vk_loader in __init__.mojo.

These are the only loader symbols consumers need. Internal types
(func_table, etc.) are deliberately excluded from the package surface.
Order matches spec.md:224-231 exactly."""


# ===--- Shared run metadata ---=== #


@dataclass(frozen=True)
class WriteConfig:
    """Shared generation metadata embedded in every file preamble.

    Constructed once per generation run from GenerateConfig + parsed XML
    registry version. Passed to every write_* and assemble_* call.

    Attributes:
        vk_xml_version: Registry version string, e.g. "1.4.343". Extracted
            from vk.xml via the existing extract_registry_version() function.
        target_version: Vulkan version selected via --version, e.g.
            VulkanVersion(1, 3).
        extensions: Extension names included in this run. Empty frozenset for
            core-only generation.
        all_extensions: True when --all-extensions was specified. Affects the
            Extensions header line and __init__.mojo docstring.
    """

    vk_xml_version: str
    target_version: VulkanVersion
    extensions: frozenset[str]
    all_extensions: bool = False


# ===--- Import spec types ---=== #


@dataclass(frozen=True)
class ExternalImport:
    """Import from a non-sibling module (ffi, stdlib, etc.).

    Renders as a single line:
        from <module> import <name1>, <name2>, ...

    Names are emitted in declaration order (caller is responsible for
    alphabetical sort if desired; S4 does not re-sort).

    Attributes:
        module: Fully qualified module path, e.g. "ffi".
        names: Tuple of names to import. Must be non-empty.
    """

    module: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class SiblingImport:
    """Named import from a sibling module in the same package.

    Renders as a single line:
        from .<module_stem> import <name1>, <name2>, ...

    Names are emitted in declaration order. Callers sort alphabetically
    before constructing (ensures deterministic output for tests).

    Attributes:
        module_stem: Module filename stem without .mojo extension,
            e.g. "vk_base_types". Use MODULE_* constants.
        names: Tuple of symbol names to import. Must be non-empty.
    """

    module_stem: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class ModuleSpec:
    """Complete input for one generated .mojo module file (not __init__.mojo).

    Passed verbatim to assemble_module_source and write_module. S5 is
    responsible for assembling correct imports and content_lines.

    Attributes:
        filename: Output filename including .mojo extension,
            e.g. "vk_structs.mojo".
        external_imports: Imports from non-package modules, in declaration
            order. Empty tuple for modules with no external imports.
        sibling_imports: Named imports from sibling package modules, in
            declaration order. Empty tuple for modules with no sibling deps.
        content_lines: Generated Mojo source lines (the body, without header
            or imports). Each string is one line without a trailing newline.
    """

    filename: str
    external_imports: tuple[ExternalImport, ...]
    sibling_imports: tuple[SiblingImport, ...]
    content_lines: tuple[str, ...]


# ===--- __init__.mojo spec types ---=== #


@dataclass(frozen=True)
class InitReExport:
    """One module's re-export entry in __init__.mojo.

    Wildcard (wildcard=True):
        from .<module_stem> import *

    Selective with multiple names (wildcard=False):
        from .<module_stem> import (
            Name1,
            Name2,
            ...
        )

    Selective with exactly one name (wildcard=False):
        from .<module_stem> import Name1

    Attributes:
        module_stem: Module filename stem, e.g. "vk_loader".
        wildcard: True to emit "import *". False to emit named import block.
        names: Symbol names for selective import. Must be non-empty when
            wildcard is False. Ignored (but may be empty) when wildcard is True.
    """

    module_stem: str
    wildcard: bool
    names: tuple[str, ...]


@dataclass(frozen=True)
class InitModuleSpec:
    """Complete input for __init__.mojo generation.

    re_exports defines the ordered re-export manifest. Order determines
    import statement order in the generated __init__.mojo.

    Use PACKAGE_MODULE_EXPORTS for the standard 7-module Vulkan package.
    Pass a custom tuple when generating a subset (e.g. test fixtures).

    Attributes:
        re_exports: Ordered sequence of module re-export entries.
    """

    re_exports: tuple[InitReExport, ...]


# ===--- Write result types ---=== #


@dataclass(frozen=True)
class FileWriteResult:
    """Result of writing a single generated file.

    Returned by write_module and write_init_module. Consumed by S6
    for per-file rows in the generation summary table.

    Attributes:
        filename: Filename written, e.g. "vk_structs.mojo" or "__init__.mojo".
        path: Absolute path of the written file.
        line_count: Number of newline characters in the written UTF-8 content.
        byte_count: Number of bytes written (UTF-8 encoded).
    """

    filename: str
    path: Path
    line_count: int
    byte_count: int


@dataclass(frozen=True)
class PackageWriteResult:
    """Result of writing the complete generated package.

    files is ordered: module files first (in MODULE_ORDER), then
    __init__.mojo last. Consumed by S6 for the generation summary report.

    Attributes:
        output_dir: Directory all files were written to.
        files: One FileWriteResult per file written, in write order.
    """

    output_dir: Path
    files: tuple[FileWriteResult, ...]

    @property
    def total_lines(self) -> int:
        """Sum of line_count across all written files."""
        return sum(f.line_count for f in self.files)


# PACKAGE_MODULE_EXPORTS is defined after InitReExport (depends on it).
PACKAGE_MODULE_EXPORTS: tuple[InitReExport, ...] = (
    InitReExport(MODULE_BASE_TYPES, wildcard=True, names=()),
    InitReExport(MODULE_ENUMS, wildcard=True, names=()),
    InitReExport(MODULE_HANDLES, wildcard=True, names=()),
    InitReExport(MODULE_STRUCTS, wildcard=True, names=()),
    InitReExport(MODULE_UNIONS, wildcard=True, names=()),
    InitReExport(MODULE_LOADER, wildcard=False, names=LOADER_SELECTIVE_EXPORTS),
    InitReExport(MODULE_COMMANDS, wildcard=True, names=()),
)
"""Canonical re-export manifest for the generated Vulkan package.

S5 passes this to write_init_module (or write_package) unchanged.
Tests can reference this constant directly to verify package layout."""


# ===--- Pure formatting functions ---=== #

_HEADER_BORDER: str = "# x-------------------------------------------x #"


def format_file_header(config: WriteConfig) -> list[str]:
    """Return comment-block lines for a generated module file header.

    Produces the machine-readable boxed comment at the top of every generated
    .mojo file (NOT __init__.mojo, which uses a docstring instead).

    Output format:
        # x-------------------------------------------x #
        # | Vulkan 1.3 bindings for Mojo
        # | Generated by vulkan-bindings-gen
        # | Source: vk.xml 1.4.343
        # | Target: Vulkan 1.3
        # | Extensions: VK_EXT_debug_utils, VK_KHR_swapchain
        # x-------------------------------------------x #

    Extensions line rules (in priority order):
        1. all_extensions=True   -> "# | Extensions: all"
        2. extensions non-empty  -> "# | Extensions: {sorted, comma-separated}"
        3. extensions empty      -> Extensions line omitted entirely

    Extensions are sorted alphabetically for deterministic output regardless
    of frozenset iteration order.

    Args:
        config: Shared generation metadata.

    Returns:
        List of source lines without trailing newlines. No trailing blank
        line — callers are responsible for surrounding whitespace.

    Raises:
        ValueError: If config.vk_xml_version is empty (would produce a
            misleading header).
    """
    if not config.vk_xml_version:
        raise ValueError("vk_xml_version must not be empty")

    version = f"{config.target_version.major}.{config.target_version.minor}"
    lines: list[str] = [
        _HEADER_BORDER,
        f"# | Vulkan {version} bindings for Mojo",
        "# | Generated by vulkan-bindings-gen",
        f"# | Source: vk.xml {config.vk_xml_version}",
        f"# | Target: Vulkan {version}",
    ]

    if config.all_extensions:
        lines.append("# | Extensions: all")
    elif config.extensions:
        sorted_exts = ", ".join(sorted(config.extensions))
        lines.append(f"# | Extensions: {sorted_exts}")

    lines.append(_HEADER_BORDER)
    return lines


def format_import_block(
    external_imports: tuple[ExternalImport, ...],
    sibling_imports: tuple[SiblingImport, ...],
) -> list[str]:
    """Return Mojo import statement lines for a module file.

    ExternalImport renders as:
        from <module> import <name1>, <name2>, ...

    SiblingImport renders as:
        from .<module_stem> import <name1>, <name2>, ...

    When both groups are non-empty, a single blank line separates external
    imports (first) from sibling imports (second).

    When both groups are empty, returns an empty list (no output).
    When only one group is non-empty, no blank line is emitted.

    Names within each import line are emitted in declaration order.

    Args:
        external_imports: Non-package imports, in declaration order.
        sibling_imports: Same-package named imports, in declaration order.

    Returns:
        List of import lines without trailing newlines. No leading or
        trailing blank line — assemble_module_source handles whitespace.

    Raises:
        ValueError: If any ExternalImport or SiblingImport has an empty
            names tuple (programming error — cannot emit a valid import
            statement without names).
    """
    for imp in external_imports:
        if not imp.names:
            raise ValueError(
                f"ExternalImport for module '{imp.module}' has empty names tuple"
            )
    for imp in sibling_imports:
        if not imp.names:
            raise ValueError(
                f"SiblingImport for module '{imp.module_stem}' has empty names tuple"
            )

    lines: list[str] = []
    for imp in external_imports:
        lines.append(f"from {imp.module} import {', '.join(imp.names)}")

    if external_imports and sibling_imports:
        lines.append("")

    for imp in sibling_imports:
        lines.append(f"from .{imp.module_stem} import {', '.join(imp.names)}")

    return lines


def assemble_module_source(config: WriteConfig, spec: ModuleSpec) -> str:
    """Assemble a complete .mojo module source string from a ModuleSpec.

    File structure (with imports):
        <header_comment_block>      <- format_file_header output
                                    <- blank line
        <import_block>              <- format_import_block output
                                    <- blank line
        <content_lines>             <- spec.content_lines joined with newlines
                                    <- trailing newline

    Without imports:
        <header_comment_block>
                                    <- blank line
        <content_lines>
                                    <- trailing newline

    When content_lines is empty, the file contains only the header (and
    imports if present), followed by a trailing newline. No extra blank
    line is emitted after a trailing section.

    Args:
        config: Shared generation metadata for the header.
        spec: Per-module spec with filename, imports, and content lines.

    Returns:
        Complete Mojo source string including trailing newline.

    Raises:
        ValueError: If spec.filename is empty or does not end with ".mojo".
        ValueError: Propagated from format_import_block on empty names tuple.
    """
    if not spec.filename or not spec.filename.endswith(".mojo"):
        raise ValueError(
            f"spec.filename must be non-empty and end with '.mojo', "
            f"got {spec.filename!r}"
        )

    parts: list[str] = list(format_file_header(config))

    has_imports = bool(spec.external_imports or spec.sibling_imports)
    if has_imports:
        parts.append("")
        parts.extend(format_import_block(spec.external_imports, spec.sibling_imports))

    if spec.content_lines:
        parts.append("")
        parts.extend(spec.content_lines)

    return "\n".join(parts) + "\n"


def assemble_init_source(config: WriteConfig, init_spec: InitModuleSpec) -> str:
    """Assemble a complete __init__.mojo source string.

    File structure:
        \"\"\"<docstring>\"\"\"         <- module docstring (no comment header)
                                    <- blank line
        <re_export_lines>           <- one entry per InitReExport
                                    <- trailing newline

    Docstring target encoding:
        - all_extensions=True   -> "{version} (all extensions)"
        - extensions non-empty  -> "{version} + {sorted ext names}"
        - extensions empty      -> "{version}"

    InitReExport rendering:
        wildcard=True:
            from .<module_stem> import *

        wildcard=False, len(names) == 1:
            from .<module_stem> import Name

        wildcard=False, len(names) > 1:
            from .<module_stem> import (
                Name1,
                Name2,
            )
            (4-space indent, trailing comma on every name)

    Args:
        config: Shared generation metadata for the docstring version string.
        init_spec: Ordered re-export manifest.

    Returns:
        Complete __init__.mojo source string including trailing newline.

    Raises:
        ValueError: If any InitReExport has wildcard=False and empty names
            (no names for a selective import is invalid).
    """
    for re_export in init_spec.re_exports:
        if not re_export.wildcard and not re_export.names:
            raise ValueError(
                f"InitReExport for module '{re_export.module_stem}' has "
                f"wildcard=False but empty names tuple"
            )

    version = f"Vulkan {config.target_version.major}.{config.target_version.minor}"
    if config.all_extensions:
        target_str = f"{version} (all extensions)"
    elif config.extensions:
        sorted_exts = ", ".join(sorted(config.extensions))
        target_str = f"{version} + {sorted_exts}"
    else:
        target_str = version

    docstring = f'"""{target_str} bindings for Mojo. Generated by vulkan-bindings-gen. Target: {version}."""'
    parts: list[str] = [docstring, ""]

    for re_export in init_spec.re_exports:
        if re_export.wildcard:
            parts.append(f"from .{re_export.module_stem} import *")
        elif len(re_export.names) == 1:
            parts.append(f"from .{re_export.module_stem} import {re_export.names[0]}")
        else:
            name_lines = "\n".join(f"    {name}," for name in re_export.names)
            parts.append(f"from .{re_export.module_stem} import (\n{name_lines}\n)")

    return "\n".join(parts) + "\n"


# ===--- Writer I/O functions ---=== #


def write_module(
    output_dir: Path, config: WriteConfig, spec: ModuleSpec
) -> FileWriteResult:
    """Write a single generated .mojo module file to disk.

    Thin I/O shell over assemble_module_source. Creates output_dir (and any
    missing parent directories) before writing.

    Args:
        output_dir: Directory to write the file into. Created if absent.
        config: Shared generation metadata passed to assemble_module_source.
        spec: Per-module spec with filename, imports, and content lines.

    Returns:
        FileWriteResult with truthful filename, resolved path, line_count
        (newline characters in content), and byte_count (UTF-8 bytes written).

    Raises:
        ValueError: Propagated from assemble_module_source on invalid spec.
        OSError: Propagated directly if the filesystem write fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    content = assemble_module_source(config, spec)
    file_path = output_dir / spec.filename
    file_path.write_text(content, encoding="utf-8")
    resolved = file_path.resolve()
    file_bytes = resolved.read_bytes()
    return FileWriteResult(
        filename=spec.filename,
        path=resolved,
        line_count=content.count("\n"),
        byte_count=len(file_bytes),
    )


def write_init_module(
    output_dir: Path, config: WriteConfig, init_spec: InitModuleSpec
) -> FileWriteResult:
    """Write __init__.mojo to disk.

    Thin I/O shell over assemble_init_source. Creates output_dir (and any
    missing parent directories) before writing.

    Args:
        output_dir: Directory to write __init__.mojo into. Created if absent.
        config: Shared generation metadata passed to assemble_init_source.
        init_spec: Ordered re-export manifest.

    Returns:
        FileWriteResult with filename="__init__.mojo", resolved path,
        line_count (newline characters in content), and byte_count
        (UTF-8 bytes written).

    Raises:
        ValueError: Propagated from assemble_init_source on invalid init_spec.
        OSError: Propagated directly if the filesystem write fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    content = assemble_init_source(config, init_spec)
    file_path = output_dir / "__init__.mojo"
    file_path.write_text(content, encoding="utf-8")
    resolved = file_path.resolve()
    file_bytes = resolved.read_bytes()
    return FileWriteResult(
        filename="__init__.mojo",
        path=resolved,
        line_count=content.count("\n"),
        byte_count=len(file_bytes),
    )


def write_package(
    output_dir: Path,
    config: WriteConfig,
    module_specs: tuple[ModuleSpec, ...],
    init_spec: InitModuleSpec,
) -> PackageWriteResult:
    """Write all module files and __init__.mojo for a complete package.

    Writes module specs first in provided order, then __init__.mojo last.
    Propagates any OSError immediately without rollback — partial writes
    are possible and intentional (no rollback behavior).

    Args:
        output_dir: Directory to write all files into. Created if absent.
        config: Shared generation metadata passed to every write call.
        module_specs: Ordered module specs to write. Written in provided order.
        init_spec: __init__.mojo re-export manifest. Written last.

    Returns:
        PackageWriteResult with files tuple ordered: module results first
        (in provided order), then init result last. total_lines is the
        sum of all file line_counts.

    Raises:
        ValueError: Propagated from any assemble_* call on invalid spec.
        OSError: Propagated directly from any write failure, without wrapping.
    """
    files: list[FileWriteResult] = []
    for spec in module_specs:
        files.append(write_module(output_dir, config, spec))
    files.append(write_init_module(output_dir, config, init_spec))
    return PackageWriteResult(
        output_dir=Path(output_dir),
        files=tuple(files),
    )


# ===--- S5 Pipeline stage boundaries ---=== #


@dataclass(frozen=True)
class RegistryMeta:
    """Scalar metadata loaded from the parsed registry XML.

    Constructed once per run immediately after XML parse. Passed to all
    stages that need registry-level data without threading the raw root element.

    Attributes:
        all_type_names: Registry type map (name -> category string) from
            load_all_type_names. Used by topo_sort_structs and build_target_set.
        api_constants: VK_MAX_* integer constants from load_api_constants.
            Used by generate_struct_mojo for array size resolution.
        known_types: Set of known type names from build_known_types.
            Used by generate_commands and loader generators.
        vk_xml_version: Version string extracted from the registry,
            e.g. "1.4.343". Used by WriteConfig and summary output.
    """

    all_type_names: dict[str, str]
    api_constants: dict[str, int]
    known_types: set[str]
    vk_xml_version: str


@dataclass(frozen=True)
class ExtractedRegistry:
    """Complete unfiltered extraction from one vk.xml parse.

    Created once per run by extract_registry. Contains all types and commands
    present in the full registry, with no version or extension filtering applied.

    Attributes:
        basetypes: (name, type_str, comment) tuples from extract_basetypes.
        enum_values: Enum name -> variant list from collect_enum_values.
        block_types: Block name -> type string map from get_enum_block_types.
        enum_type_names: All enum type names from get_enum_type_names.
        handles: Handle definition tuples from extract_handles (first return).
        handle_aliases: (alias_name, target_name) pairs from extract_handles.
        unions: Union StructDef list from extract_unions (first return).
        union_aliases: (alias_name, target_name) pairs from extract_unions.
        structs: Struct StructDef list from extract_structs (first return), unsorted.
        struct_aliases: (alias_name, target_name) pairs from extract_structs.
        commands: CommandDef list from extract_commands (first return).
        cmd_aliases: (alias_name, target_name) pairs from extract_commands.
    """

    basetypes: list
    enum_values: dict
    block_types: dict
    enum_type_names: set
    handles: list
    handle_aliases: list
    unions: list
    union_aliases: list
    structs: list
    struct_aliases: list
    commands: list
    cmd_aliases: list


@dataclass(frozen=True)
class FilteredRegistry:
    """Target-filtered and post-processed extraction, ready for generation.

    Produced by filter_and_prepare. All list fields are filtered to the
    TargetSet. structs is topologically sorted. union_sizes contains
    C-measured byte sizes for all included unions.

    Attributes:
        basetypes: All base types, unfiltered (always required as foundation).
        enum_values: Filtered enum name -> variant list.
        block_types: Full block type map (used for enum generation).
        enum_type_names: All enum type names (S2 filtering via enum_values keys).
        handles: Filtered handle definition tuples.
        handle_aliases: Filtered handle alias pairs.
        unions: Filtered union StructDef list.
        union_aliases: Filtered union alias pairs.
        union_sizes: dict[name, byte_size] from measure_union_sizes on filtered unions.
        structs: Filtered and topologically sorted StructDef list.
        struct_aliases: Filtered struct alias pairs.
        commands: Filtered CommandDef list.
        cmd_aliases: Filtered command alias pairs.
    """

    basetypes: list
    enum_values: dict
    block_types: dict
    enum_type_names: set
    handles: list
    handle_aliases: list
    unions: list
    union_aliases: list
    union_sizes: dict
    structs: list
    struct_aliases: list
    commands: list
    cmd_aliases: list


# ===--- S5 Stage functions ---=== #


def extract_registry(
    root: ET.Element,
    meta: RegistryMeta,
) -> ExtractedRegistry:
    """Run all extraction functions and bundle their outputs into one container.

    Calls extract_basetypes, collect_enum_values, get_enum_block_types,
    get_enum_type_names, extract_handles, extract_unions, extract_structs,
    and extract_commands. No filtering is applied.

    Args:
        root: Parsed registry XML root element.
        meta: Registry metadata (unused; reserved for future use).

    Returns:
        ExtractedRegistry with all category lists populated.

    Raises:
        Any exception propagated from the underlying extract_* functions.
    """
    basetypes = extract_basetypes(root)
    enum_values = collect_enum_values(root)
    block_types = get_enum_block_types(root)
    enum_type_names = get_enum_type_names(root)
    handles, handle_aliases = extract_handles(root)
    unions, union_aliases = extract_unions(root)
    structs, struct_aliases = extract_structs(root)
    commands, cmd_aliases = extract_commands(root)
    return ExtractedRegistry(
        basetypes=basetypes,
        enum_values=enum_values,
        block_types=block_types,
        enum_type_names=enum_type_names,
        handles=handles,
        handle_aliases=handle_aliases,
        unions=unions,
        union_aliases=union_aliases,
        structs=structs,
        struct_aliases=struct_aliases,
        commands=commands,
        cmd_aliases=cmd_aliases,
    )


def resolve_generate_extensions(
    root: ET.Element,
    config: GenerateConfig,
) -> frozenset[str]:
    """Map GenerateConfig extension selection to a closed, transitive extension set.

    Handles two modes:
    - all_extensions=True: uses all_vulkan_extension_names(root) as the seed,
      then closes via resolve_extension_deps.
    - all_extensions=False: passes config.extensions directly to
      resolve_extension_deps. Returns frozenset() for empty config.extensions.

    Args:
        root: Parsed registry XML root element.
        config: GenerateConfig with version, extensions, and all_extensions fields.

    Returns:
        Closed frozenset of extension names. Empty frozenset for core-only generation.
    """
    if not config.extensions and not config.all_extensions:
        return frozenset()

    if config.all_extensions:
        initial = all_vulkan_extension_names(root)
    else:
        initial = config.extensions

    return resolve_extension_deps(root, initial)


def filter_and_prepare(
    extracted: ExtractedRegistry,
    target: TargetSet,
    meta: RegistryMeta,
) -> FilteredRegistry:
    """Apply TargetSet filtering and post-processing to all extracted categories.

    Filtering: applies filter_by_target and filter_aliases_by_target to every
    filterable category. basetypes pass through unfiltered.

    Post-processing on filtered data:
    - measure_union_sizes(filtered_unions): subprocess measurement on filtered set only.
    - topo_sort_structs(filtered_structs, meta.all_type_names): dependency-ordered
      struct list using the full type registry for correct pointer resolution.

    Args:
        extracted: Complete unfiltered registry from extract_registry.
        target: TargetSet specifying included names.
        meta: Registry metadata; all_type_names passed to topo_sort_structs.

    Returns:
        FilteredRegistry with categories filtered, union_sizes measured, and
        structs topologically sorted.

    Raises:
        RuntimeError: Propagated from topo_sort_structs if a dependency cycle
            is detected.
    """
    filtered_enum_values = {
        name: vals
        for name, vals in extracted.enum_values.items()
        if name in target.types
    }
    filtered_handles = filter_by_target(
        extracted.handles, target.types, key=lambda h: h[0]
    )
    filtered_handle_aliases = filter_aliases_by_target(
        extracted.handle_aliases, target.types
    )
    filtered_unions = filter_by_target(
        extracted.unions, target.types, key=lambda u: u.name
    )
    filtered_union_aliases = filter_aliases_by_target(
        extracted.union_aliases, target.types
    )
    filtered_structs_raw = filter_by_target(
        extracted.structs, target.types, key=lambda s: s.name
    )
    filtered_struct_aliases = filter_aliases_by_target(
        extracted.struct_aliases, target.types
    )
    filtered_commands = filter_by_target(
        extracted.commands, target.commands, key=lambda c: c.name
    )
    filtered_cmd_aliases = filter_aliases_by_target(
        extracted.cmd_aliases, target.commands
    )

    union_sizes = measure_union_sizes(filtered_unions)
    filtered_structs = topo_sort_structs(filtered_structs_raw, meta.all_type_names)

    return FilteredRegistry(
        basetypes=extracted.basetypes,
        enum_values=filtered_enum_values,
        block_types=extracted.block_types,
        enum_type_names=extracted.enum_type_names,
        handles=filtered_handles,
        handle_aliases=filtered_handle_aliases,
        unions=filtered_unions,
        union_aliases=filtered_union_aliases,
        union_sizes=union_sizes,
        structs=filtered_structs,
        struct_aliases=filtered_struct_aliases,
        commands=filtered_commands,
        cmd_aliases=filtered_cmd_aliases,
    )


def build_write_config(
    config: GenerateConfig,
    vk_xml_version: str,
) -> WriteConfig:
    """Construct WriteConfig from GenerateConfig and parsed registry version.

    Args:
        config: Validated GenerateConfig from build_config.
        vk_xml_version: Registry version string, e.g. "1.4.343". Must be non-empty.

    Returns:
        WriteConfig with vk_xml_version, target_version, extensions, and
        all_extensions populated from config.

    Raises:
        ValueError: If vk_xml_version is empty string.
    """
    if not vk_xml_version:
        raise ValueError("vk_xml_version must not be empty")
    return WriteConfig(
        vk_xml_version=vk_xml_version,
        target_version=config.version,
        extensions=config.extensions,
        all_extensions=config.all_extensions,
    )


def build_module_specs(
    root: ET.Element,
    filtered: FilteredRegistry,
    write_config: WriteConfig,
    meta: RegistryMeta,
) -> tuple[ModuleSpec, ...]:
    """Assemble all seven ModuleSpec instances from filtered registry data.

    Returns specs in MODULE_ORDER: vk_base_types, vk_enums, vk_handles,
    vk_structs, vk_unions, vk_loader, vk_commands.

    Struct generation errors are collected as inline # ERROR comments (non-fatal).

    Args:
        root: Registry XML root (needed for bitmask/funcpointer/stdvideo generators).
        filtered: Filtered and post-processed registry data.
        write_config: Shared generation metadata (unused here; passed for consistency).
        meta: Registry metadata; known_types used by command/loader generators.

    Returns:
        Tuple of seven ModuleSpec instances in MODULE_ORDER.
    """
    # Build reverse map: type name -> module stem (for sibling import computation)
    type_to_module: dict[str, str] = {}
    for name, _, _ in filtered.basetypes:
        type_to_module[name] = MODULE_BASE_TYPES
    for name in filtered.enum_values:
        type_to_module[name] = MODULE_ENUMS
    for h in filtered.handles:
        type_to_module[h[0]] = MODULE_HANDLES
    for alias, _ in filtered.handle_aliases:
        type_to_module[alias] = MODULE_HANDLES
    for u in filtered.unions:
        type_to_module[u.name] = MODULE_UNIONS
    for alias, _ in filtered.union_aliases:
        type_to_module[alias] = MODULE_UNIONS
    for s in filtered.structs:
        type_to_module[s.name] = MODULE_STRUCTS
    for alias, _ in filtered.struct_aliases:
        type_to_module[alias] = MODULE_STRUCTS
    for c in filtered.commands:
        type_to_module[c.name] = MODULE_COMMANDS

    def _sibling_imports(
        referenced: set[str],
        current: str,
    ) -> tuple[SiblingImport, ...]:
        by_module: dict[str, list[str]] = {}
        for name in referenced:
            src = type_to_module.get(name)
            if src and src != current:
                by_module.setdefault(src, []).append(name)
        return tuple(
            SiblingImport(module_stem=stem, names=tuple(sorted(names)))
            for stem, names in sorted(by_module.items())
            if names
        )

    # ── vk_base_types ──────────────────────────────────────────────────────────
    ffi_import = ExternalImport(
        module="ffi",
        names=(
            "_Global",
            "c_char",
            "c_int",
            "c_uint",
            "c_float",
            "c_double",
            "c_size_t",
        ),
    )
    base_lines: list[str] = []
    base_lines.extend(generate_base_types(filtered.basetypes))
    base_lines.extend(generate_bitmask_aliases(root))
    base_lines.extend(generate_funcpointer_aliases(root))
    base_lines.extend(generate_stdvideo_types(root))
    base_spec = ModuleSpec(
        filename=f"{MODULE_BASE_TYPES}.mojo",
        external_imports=(ffi_import,),
        sibling_imports=(),
        content_lines=tuple(base_lines),
    )

    # ── vk_enums ───────────────────────────────────────────────────────────────
    enum_lines = generate_enums(
        filtered.enum_values, filtered.block_types, filtered.enum_type_names
    )
    enum_spec = ModuleSpec(
        filename=f"{MODULE_ENUMS}.mojo",
        external_imports=(),
        sibling_imports=(),
        content_lines=tuple(enum_lines),
    )

    # ── vk_handles ─────────────────────────────────────────────────────────────
    handle_lines = generate_handles(filtered.handles, filtered.handle_aliases)
    handle_spec = ModuleSpec(
        filename=f"{MODULE_HANDLES}.mojo",
        external_imports=(),
        sibling_imports=(),
        content_lines=tuple(handle_lines),
    )

    # ── vk_structs ─────────────────────────────────────────────────────────────
    struct_lines: list[str] = []
    struct_member_types: set[str] = set()
    for s in filtered.structs:
        for m in s.members:
            struct_member_types.add(m.type_name)
        try:
            struct_lines.extend(
                generate_struct_mojo(s, meta.all_type_names, meta.api_constants)
            )
        except Exception as e:
            struct_lines.append(f"# ERROR generating {s.name}: {e}")
            struct_lines.append("")
    for alias_name, target_name in sorted(filtered.struct_aliases, key=lambda x: x[0]):
        struct_lines.append(f"comptime {alias_name} = {target_name}")
    struct_sibling_imports = _sibling_imports(struct_member_types, MODULE_STRUCTS)
    struct_spec = ModuleSpec(
        filename=f"{MODULE_STRUCTS}.mojo",
        external_imports=(),
        sibling_imports=struct_sibling_imports,
        content_lines=tuple(struct_lines),
    )

    # ── vk_unions ──────────────────────────────────────────────────────────────
    union_lines: list[str] = []
    union_member_types: set[str] = set()
    for u in filtered.unions:
        for m in u.members:
            union_member_types.add(m.type_name)
        union_lines.extend(generate_union_mojo(u, filtered.union_sizes))
    for alias_name, target_name in sorted(filtered.union_aliases, key=lambda x: x[0]):
        union_lines.append(f"comptime {alias_name} = {target_name}")
    union_sibling_imports = _sibling_imports(union_member_types, MODULE_UNIONS)
    union_spec = ModuleSpec(
        filename=f"{MODULE_UNIONS}.mojo",
        external_imports=(),
        sibling_imports=union_sibling_imports,
        content_lines=tuple(union_lines),
    )

    # ── vk_loader ──────────────────────────────────────────────────────────────
    # Decomposed from generate_loader_section: infra + init functions only (no wrappers)
    loader_lines: list[str] = []
    loader_lines.extend(generate_loader())
    global_cmds = [c for c in filtered.commands if classify_command(c) == "global"]
    instance_cmds = [c for c in filtered.commands if classify_command(c) == "instance"]
    device_cmds = [c for c in filtered.commands if classify_command(c) == "device"]
    loader_lines.extend(generate_init_function("init_vulkan_global", global_cmds))
    loader_lines.extend(generate_init_function("init_vulkan_instance", instance_cmds))
    loader_lines.extend(generate_init_function("init_vulkan_device", device_cmds))
    loader_lines.append(
        "fn init_vulkan(global_load: LoadProc, instance_load: LoadProc, device_load: LoadProc) raises:"
    )
    loader_lines.append("    init_vulkan_global(global_load)")
    loader_lines.append("    init_vulkan_instance(instance_load)")
    loader_lines.append("    init_vulkan_device(device_load)")
    loader_lines.append("")
    loader_spec = ModuleSpec(
        filename=f"{MODULE_LOADER}.mojo",
        external_imports=(ExternalImport(module="ffi", names=("_Global",)),),
        sibling_imports=(),
        content_lines=tuple(loader_lines),
    )

    # ── vk_commands ────────────────────────────────────────────────────────────
    # command type declarations + wrapper functions + wrapper aliases
    cmd_lines: list[str] = []
    cmd_lines.extend(
        generate_commands(filtered.commands, filtered.cmd_aliases, meta.known_types)
    )
    for cmd in sorted(filtered.commands, key=lambda c: c.name):
        cmd_lines.extend(generate_wrapper_fn(cmd, meta.known_types))
    cmd_by_name = {cmd.name: cmd for cmd in filtered.commands}
    for alias_name, target_name in sorted(filtered.cmd_aliases, key=lambda x: x[0]):
        target_cmd = cmd_by_name.get(target_name)
        if target_cmd is None:
            continue
        cmd_lines.extend(
            generate_wrapper_alias(
                alias_name,
                target_name,
                target_cmd,
                meta.known_types,
            )
        )
    cmd_referenced_types = meta.known_types & set(type_to_module.keys())
    cmd_sibling_imports = _sibling_imports(cmd_referenced_types, MODULE_COMMANDS)
    cmd_import_map: dict[str, set[str]] = {
        imp.module_stem: set(imp.names) for imp in cmd_sibling_imports
    }
    cmd_import_map.setdefault(MODULE_LOADER, set()).add("get_fn")
    cmd_sibling_imports = tuple(
        SiblingImport(module_stem=stem, names=tuple(sorted(names)))
        for stem, names in sorted(cmd_import_map.items())
        if names
    )
    cmd_spec = ModuleSpec(
        filename=f"{MODULE_COMMANDS}.mojo",
        external_imports=(),
        sibling_imports=cmd_sibling_imports,
        content_lines=tuple(cmd_lines),
    )

    return (
        base_spec,
        enum_spec,
        handle_spec,
        struct_spec,
        union_spec,
        loader_spec,
        cmd_spec,
    )


def run_generate(
    config: GenerateConfig,
) -> PackageWriteResult:
    """Execute the complete generation pipeline for a GenerateConfig.

    Runs all seven stages in order: parse -> meta load -> extract ->
    resolve extensions -> build target set -> filter -> assemble -> write.

    Args:
        config: Validated GenerateConfig from build_config.

    Returns:
        PackageWriteResult describing every file written.

    Raises:
        OSError: XML file not readable or filesystem write failure.
        ET.ParseError: Malformed vk.xml parse failure.
        RuntimeError: Safety limit exceeded in dependency closure.
        ValueError: Malformed ModuleSpec assembled in stage 5.
    """
    print(f"Parsing: {config.vk_xml}")
    tree = ET.parse(config.vk_xml)
    root = tree.getroot()

    all_type_names = load_all_type_names(root)
    api_constants = load_api_constants(root)
    known_types = build_known_types(root)
    vk_xml_version = extract_registry_version(root)
    print(f"  Registry: {len(all_type_names)} types, {len(api_constants)} constants")

    meta = RegistryMeta(
        all_type_names=all_type_names,
        api_constants=api_constants,
        known_types=known_types,
        vk_xml_version=vk_xml_version,
    )

    extracted = extract_registry(root, meta)
    print(
        f"  Extracted: {len(extracted.structs)} structs, "
        f"{len(extracted.commands)} commands, {len(extracted.handles)} handles, "
        f"{len(extracted.enum_values)} enums, {len(extracted.unions)} unions"
    )

    resolved_exts = resolve_generate_extensions(root, config)
    print(f"  Extensions: {len(resolved_exts)} resolved")

    target, _stats = build_target_set(
        root,
        config.version,
        resolved_exts,
        extracted.structs,
        meta.all_type_names,
    )
    print(f"  Target: {len(target.types)} types, {len(target.commands)} commands")

    filtered = filter_and_prepare(extracted, target, meta)

    write_config = build_write_config(config, meta.vk_xml_version)

    print(f"  Assembling: {len(MODULE_ORDER)} module specs")
    module_specs = build_module_specs(root, filtered, write_config, meta)

    init_spec = InitModuleSpec(re_exports=PACKAGE_MODULE_EXPORTS)
    result = write_package(config.output_dir, write_config, module_specs, init_spec)
    print(
        f"  Written: {len(result.files)} files, "
        f"{result.total_lines} lines to {result.output_dir}"
    )

    ext_type_names = collect_extension_types(root, resolved_exts, config.version)
    ext_command_names = collect_extension_commands(root, resolved_exts, config.version)
    summary = build_generation_summary(
        write_config, filtered, ext_type_names, ext_command_names, result
    )
    print_generation_summary(summary)

    return result


# ===--- S6 Summary report ---=== #


PREVIOUS_SINGLE_FILE_LINE_COUNT: int = 25_500
"""Historical single-file line count from the pre-v2 generator output.

Used in the "Total: N lines (was M in 1 file)" reduction summary line.
Update only if the legacy generator's output size changes significantly.
"""


@dataclass(frozen=True)
class CategoryCount:
    """Count of items in one type/command category, split by core vs. extension.

    Used for each of the six "Types generated:" rows. The ext field drives
    the optional "(N core + M from extensions)" annotation in the renderer.

    Invariant: core + ext == total. Enforced by build_generation_counts.

    Attributes:
        total: Total count of primary items (not aliases) in this category.
        core: Count of items NOT in the ext_type_names / ext_command_names set.
        ext: Count of items IN the ext_type_names / ext_command_names set.
    """

    total: int
    core: int
    ext: int


@dataclass(frozen=True)
class GenerationCounts:
    """Per-category item counts derived from the filtered registry.

    One CategoryCount per generated-type category, mirroring the spec
    "Types generated:" section order. Aliases excluded from all counts.

    Attributes:
        base_types: VkBool32, VkFlags, etc. ext always 0 (core-only in spec).
        enums: Enum types filtered to target version + extensions.
        handles: Handle types filtered to target version + extensions.
        structs: Struct types (topo-sorted). ext > 0 for extension structs.
        unions: Union types. ext is typically 0 (extension unions are rare).
        commands: Command wrappers. ext > 0 when extension commands included.
    """

    base_types: CategoryCount
    enums: CategoryCount
    handles: CategoryCount
    structs: CategoryCount
    unions: CategoryCount
    commands: CategoryCount


@dataclass(frozen=True)
class GenerationSummary:
    """Complete, immutable data for the post-generation console report.

    Produced by build_generation_summary. Consumed by format_generation_summary
    and print_generation_summary. Carries no logic — pure sealed data.

    Attributes:
        target_label: Human-readable target string built by build_target_label.
        source_label: Registry source string, e.g. "vk.xml 1.4.343".
        output_dir: Output directory path as string (verbatim from PackageWriteResult).
        counts: Per-category item counts from build_generation_counts.
        files: Ordered write results from PackageWriteResult.files.
        previous_line_count: Legacy single-file baseline for the reduction line.
    """

    target_label: str
    source_label: str
    output_dir: str
    counts: GenerationCounts
    files: tuple[FileWriteResult, ...]
    previous_line_count: int


def build_target_label(config: WriteConfig) -> str:
    """Build the human-readable target string for the summary Target: row.

    Three cases in priority order:
    1. config.all_extensions=True  -> "Vulkan {major}.{minor} + all extensions"
    2. config.extensions non-empty -> "Vulkan {major}.{minor} + {ext1}, {ext2}, ..."
       Extensions sorted alphabetically for deterministic output.
    3. config.extensions empty     -> "Vulkan {major}.{minor}"

    Args:
        config: WriteConfig with target_version, extensions, and all_extensions.

    Returns:
        Target label string. No trailing whitespace or newline.
    """
    version_str = f"Vulkan {config.target_version}"
    if config.all_extensions:
        return f"{version_str} + all extensions"
    if config.extensions:
        sorted_exts = ", ".join(sorted(config.extensions))
        return f"{version_str} + {sorted_exts}"
    return version_str


def build_generation_counts(
    filtered: FilteredRegistry,
    ext_type_names: frozenset[str],
    ext_command_names: frozenset[str],
) -> GenerationCounts:
    """Compute per-category totals and core/ext splits from the filtered registry.

    For each category, counts primary items only (not aliases). An item is
    classified as ext if its primary name appears in ext_type_names (or
    ext_command_names for commands). All other items are core.

    Asserts core + ext == total for every CategoryCount before returning.

    Args:
        filtered: Target-filtered registry from filter_and_prepare.
        ext_type_names: Names of types from extension require blocks.
        ext_command_names: Names of commands from extension require blocks.

    Returns:
        GenerationCounts with six CategoryCount fields, all invariants satisfied.
    """

    def _count(names: list[str], ext_set: frozenset[str]) -> CategoryCount:
        total = len(names)
        ext = sum(1 for n in names if n in ext_set)
        core = total - ext
        assert core + ext == total, (
            f"CategoryCount invariant violated: {core}+{ext}!={total}"
        )
        return CategoryCount(total=total, core=core, ext=ext)

    base_type_names = [item[0] for item in filtered.basetypes]
    enum_names = list(filtered.enum_values.keys())
    handle_names = [item[0] for item in filtered.handles]
    struct_names = [s.name for s in filtered.structs]
    union_names = [u.name for u in filtered.unions]
    command_names = [c.name for c in filtered.commands]

    return GenerationCounts(
        base_types=_count(base_type_names, ext_type_names),
        enums=_count(enum_names, ext_type_names),
        handles=_count(handle_names, ext_type_names),
        structs=_count(struct_names, ext_type_names),
        unions=_count(union_names, ext_type_names),
        commands=_count(command_names, ext_command_names),
    )


def build_generation_summary(
    write_config: WriteConfig,
    filtered: FilteredRegistry,
    ext_type_names: frozenset[str],
    ext_command_names: frozenset[str],
    write_result: PackageWriteResult,
    previous_line_count: int = PREVIOUS_SINGLE_FILE_LINE_COUNT,
) -> GenerationSummary:
    """Assemble a complete GenerationSummary from pipeline stage outputs.

    Delegates to build_target_label and build_generation_counts. Copies
    write_result.files verbatim (ordering preserved).

    Args:
        write_config: Shared generation metadata (target_version, vk_xml_version,
            extensions, all_extensions).
        filtered: Target-filtered registry for category counts.
        ext_type_names: Extension type names for core/ext split.
        ext_command_names: Extension command names for core/ext split.
        write_result: Package write result (file list + line counts).
        previous_line_count: Historical baseline for reduction comparison.
            Defaults to PREVIOUS_SINGLE_FILE_LINE_COUNT.

    Returns:
        GenerationSummary with all fields populated and all invariants satisfied.
    """
    return GenerationSummary(
        target_label=build_target_label(write_config),
        source_label=f"vk.xml {write_config.vk_xml_version}",
        output_dir=str(write_result.output_dir),
        counts=build_generation_counts(filtered, ext_type_names, ext_command_names),
        files=write_result.files,
        previous_line_count=previous_line_count,
    )


def format_generation_summary(summary: GenerationSummary) -> str:
    """Render a GenerationSummary to the spec-defined multi-section console string.

    Heading uses the version component of target_label ("Vulkan X.Y"), not the
    full label. Split annotations appear only when ext > 0. Line counts use
    thousands separators. Returns a string with exactly one trailing newline.

    Args:
        summary: Assembled GenerationSummary from build_generation_summary.

    Returns:
        Formatted multi-line string including trailing newline.
    """
    # Extract "Vulkan X.Y" from target_label for the heading.
    version_part = " ".join(summary.target_label.split()[:2])
    heading = f"{version_part} bindings generated:"

    lines: list[str] = []
    lines.append(heading)
    lines.append("")
    lines.append(f"  Target:     {summary.target_label}")
    lines.append(f"  Source:     {summary.source_label}")
    lines.append(f"  Output:     {summary.output_dir}")
    lines.append("")
    lines.append("  Types generated:")

    def _type_row(label: str, cc: CategoryCount) -> str:
        count_str = f"{cc.total:>6}"
        if cc.ext > 0:
            return f"    {label:<11}{count_str}  ({cc.core} core + {cc.ext} from extensions)"
        return f"    {label:<11}{count_str}"

    lines.append(_type_row("Base types:", summary.counts.base_types))
    lines.append(_type_row("Enums:", summary.counts.enums))
    lines.append(_type_row("Handles:", summary.counts.handles))
    lines.append(_type_row("Structs:", summary.counts.structs))
    lines.append(_type_row("Unions:", summary.counts.unions))
    lines.append(_type_row("Commands:", summary.counts.commands))

    lines.append("")
    lines.append("  Files written:")
    for file_result in summary.files:
        line_str = f"{file_result.line_count:>6,} lines"
        lines.append(f"    {file_result.filename:<28} {line_str}")

    total_lines = sum(f.line_count for f in summary.files)
    file_count = len(summary.files)
    lines.append("")
    lines.append(
        f"  Total: {total_lines:,} lines across {file_count} files"
        f" (was {summary.previous_line_count:,} in 1 file)"
    )
    lines.append("")
    lines.append(f"  Verify: mojo package {summary.output_dir} -o /tmp/vulkan.mojopkg")
    lines.append("")

    return "\n".join(lines)


def print_generation_summary(summary: GenerationSummary) -> None:
    """Print the generation summary to stdout.

    Thin wrapper around format_generation_summary. Kept separate so
    format_generation_summary remains purely testable without stdout capture.

    Args:
        summary: Assembled GenerationSummary from build_generation_summary.
    """
    print(format_generation_summary(summary), end="")


# ===--- Main generation ---=== #


def main():
    try:
        config = build_config()
    except ConfigError as err:
        print(f"Config error [{err.code}]: {err.message}")
        if err.suggestion:
            print(f"Hint: {err.suggestion}")
        raise SystemExit(1) from err

    if isinstance(config, DiscoveryConfig):
        run_discovery(config)
        return

    try:
        run_generate(config)
    except (OSError, ET.ParseError) as err:
        print(f"Error: {err}")
        raise SystemExit(1) from err
    except (RuntimeError, ValueError) as err:
        print(f"Internal error: {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
