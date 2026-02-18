from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import gen


def _require_callable(name: str) -> Callable[..., object]:
    symbol = getattr(gen, name, None)
    assert callable(symbol), f"Missing S4 API symbol: gen.{name}"
    return symbol


def _require_type(name: str) -> type:
    symbol = getattr(gen, name, None)
    assert isinstance(symbol, type), f"Missing S4 type symbol: gen.{name}"
    return symbol


def _make_write_config(
    *,
    vk_xml_version: str = "1.4.343",
    major: int = 1,
    minor: int = 3,
    extensions: frozenset[str] = frozenset(),
    all_extensions: bool = False,
) -> object:
    write_config_type = _require_type("WriteConfig")
    return write_config_type(
        vk_xml_version=vk_xml_version,
        target_version=gen.VulkanVersion(major, minor),
        extensions=extensions,
        all_extensions=all_extensions,
    )


def _make_external_import(module: str, names: tuple[str, ...]) -> object:
    external_import_type = _require_type("ExternalImport")
    return external_import_type(module=module, names=names)


def _make_sibling_import(module_stem: str, names: tuple[str, ...]) -> object:
    sibling_import_type = _require_type("SiblingImport")
    return sibling_import_type(module_stem=module_stem, names=names)


def _make_module_spec(
    *,
    filename: str,
    external_imports: tuple[object, ...] = (),
    sibling_imports: tuple[object, ...] = (),
    content_lines: tuple[str, ...] = (),
) -> object:
    module_spec_type = _require_type("ModuleSpec")
    return module_spec_type(
        filename=filename,
        external_imports=external_imports,
        sibling_imports=sibling_imports,
        content_lines=content_lines,
    )


def _make_init_re_export(
    module_stem: str, wildcard: bool, names: tuple[str, ...]
) -> object:
    init_re_export_type = _require_type("InitReExport")
    return init_re_export_type(module_stem=module_stem, wildcard=wildcard, names=names)


def _make_init_module_spec(re_exports: tuple[object, ...]) -> object:
    init_module_spec_type = _require_type("InitModuleSpec")
    return init_module_spec_type(re_exports=re_exports)


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_t_01_format_file_header_core_only_omits_extensions_line() -> None:
    format_file_header = _require_callable("format_file_header")
    config = _make_write_config()

    lines = format_file_header(config)

    assert any(line.startswith("# | Source: vk.xml 1.4.343") for line in lines)
    assert any(line.startswith("# | Target: Vulkan 1.3") for line in lines)
    assert not any(line.startswith("# | Extensions:") for line in lines)


def test_t_02_format_file_header_extensions_are_sorted_and_comma_separated() -> None:
    format_file_header = _require_callable("format_file_header")
    config = _make_write_config(extensions=frozenset({"VK_Z", "VK_A"}))

    lines = format_file_header(config)

    assert "# | Extensions: VK_A, VK_Z" in lines


def test_t_03_format_file_header_all_extensions_overrides_list() -> None:
    format_file_header = _require_callable("format_file_header")
    config = _make_write_config(
        extensions=frozenset({"VK_EXT_debug_utils"}),
        all_extensions=True,
    )

    lines = format_file_header(config)

    assert "# | Extensions: all" in lines
    assert not any("VK_EXT_debug_utils" in line for line in lines)


def test_t_04_format_file_header_uses_fixed_symmetric_border() -> None:
    format_file_header = _require_callable("format_file_header")
    config = _make_write_config()

    lines = format_file_header(config)

    expected_border = "# x-------------------------------------------x #"
    assert lines[0] == expected_border
    assert lines[-1] == expected_border


def test_t_05_format_file_header_rejects_empty_vk_xml_version() -> None:
    format_file_header = _require_callable("format_file_header")
    config = _make_write_config(vk_xml_version="")

    with pytest.raises(ValueError):
        format_file_header(config)


def test_t_06_format_import_block_external_only_renders_declaration_order() -> None:
    format_import_block = _require_callable("format_import_block")
    external = (_make_external_import("ffi", ("c_int", "c_float")),)

    lines = format_import_block(external, ())

    assert lines == ["from ffi import c_int, c_float"]


def test_t_07_format_import_block_sibling_only_renders_relative_named_import() -> None:
    format_import_block = _require_callable("format_import_block")
    sibling = (_make_sibling_import("vk_enums", ("VkFormat", "VkImageLayout")),)

    lines = format_import_block((), sibling)

    assert lines == ["from .vk_enums import VkFormat, VkImageLayout"]


def test_t_08_format_import_block_both_groups_have_single_blank_separator() -> None:
    format_import_block = _require_callable("format_import_block")
    external = (_make_external_import("ffi", ("c_int",)),)
    sibling = (_make_sibling_import("vk_enums", ("VkFormat",)),)

    lines = format_import_block(external, sibling)

    assert lines == [
        "from ffi import c_int",
        "",
        "from .vk_enums import VkFormat",
    ]
    assert lines.count("") == 1


def test_t_09_format_import_block_rejects_empty_names_tuples() -> None:
    format_import_block = _require_callable("format_import_block")
    with pytest.raises(ValueError):
        format_import_block((_make_external_import("ffi", ()),), ())
    with pytest.raises(ValueError):
        format_import_block((), (_make_sibling_import("vk_enums", ()),))


def test_t_10_assemble_module_source_no_imports_layout_and_newline() -> None:
    assemble_module_source = _require_callable("assemble_module_source")
    config = _make_write_config()
    spec = _make_module_spec(
        filename="vk_base_types.mojo",
        content_lines=("alias VkBool32 = UInt32",),
    )

    text = assemble_module_source(config, spec)
    lines = _lines(text)
    border = "# x-------------------------------------------x #"
    header_end = lines.index(border, 1)

    assert lines[0] == border
    assert lines[header_end + 1] == ""
    assert lines[header_end + 2] == "alias VkBool32 = UInt32"
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_t_11_assemble_module_source_with_imports_includes_required_boundaries() -> (
    None
):
    assemble_module_source = _require_callable("assemble_module_source")
    config = _make_write_config()
    spec = _make_module_spec(
        filename="vk_structs.mojo",
        external_imports=(_make_external_import("ffi", ("c_int",)),),
        sibling_imports=(_make_sibling_import("vk_enums", ("VkFormat",)),),
        content_lines=("struct VkFoo:", "    pass"),
    )

    text = assemble_module_source(config, spec)
    lines = _lines(text)
    border = "# x-------------------------------------------x #"
    header_end = lines.index(border, 1)
    first_import = lines.index("from ffi import c_int")
    first_content = lines.index("struct VkFoo:")

    assert lines[header_end + 1] == ""
    assert first_import == header_end + 2
    assert lines[first_import + 1] == ""
    assert lines[first_import + 2] == "from .vk_enums import VkFormat"
    assert lines[first_content - 1] == ""


def test_t_12_assemble_module_source_rejects_invalid_filename() -> None:
    assemble_module_source = _require_callable("assemble_module_source")
    config = _make_write_config()
    for filename in ("", "vk_structs", "vk_structs.txt"):
        spec = _make_module_spec(filename=filename, content_lines=("alias X = Int",))
        with pytest.raises(ValueError):
            assemble_module_source(config, spec)


def test_t_13_assemble_module_source_allows_empty_content_with_trailing_newline() -> (
    None
):
    assemble_module_source = _require_callable("assemble_module_source")
    config = _make_write_config()
    specs = (
        _make_module_spec(filename="vk_structs.mojo", content_lines=()),
        _make_module_spec(
            filename="vk_structs.mojo",
            external_imports=(_make_external_import("ffi", ("c_int",)),),
            content_lines=(),
        ),
    )
    for spec in specs:
        text = assemble_module_source(config, spec)
        assert text.endswith("\n")
        assert not text.endswith("\n\n")


def test_t_14_assemble_init_source_wildcard_re_export_line() -> None:
    assemble_init_source = _require_callable("assemble_init_source")
    config = _make_write_config()
    init_spec = _make_init_module_spec((_make_init_re_export("vk_enums", True, ()),))

    text = assemble_init_source(config, init_spec)

    assert "from .vk_enums import *" in _lines(text)


def test_t_15_assemble_init_source_single_selective_name_is_inline() -> None:
    assemble_init_source = _require_callable("assemble_init_source")
    config = _make_write_config()
    init_spec = _make_init_module_spec(
        (_make_init_re_export("vk_loader", False, ("LoadProc",)),)
    )

    text = assemble_init_source(config, init_spec)

    assert "from .vk_loader import LoadProc" in _lines(text)


def test_t_16_assemble_init_source_multi_selective_names_use_indented_block() -> None:
    assemble_init_source = _require_callable("assemble_init_source")
    config = _make_write_config()
    init_spec = _make_init_module_spec(
        (_make_init_re_export("vk_loader", False, ("LoadProc", "FuncPtr")),)
    )

    text = assemble_init_source(config, init_spec)

    assert "from .vk_loader import (\n    LoadProc,\n    FuncPtr,\n)" in text


def test_t_17_assemble_init_source_docstring_target_encoding_modes() -> None:
    assemble_init_source = _require_callable("assemble_init_source")
    init_spec = _make_init_module_spec(((_make_init_re_export("vk_enums", True, ())),))
    cases = (
        (_make_write_config(), "Vulkan 1.3"),
        (
            _make_write_config(extensions=frozenset({"VK_Z", "VK_A"})),
            "Vulkan 1.3 + VK_A, VK_Z",
        ),
        (
            _make_write_config(
                extensions=frozenset({"VK_EXT_debug_utils"}),
                all_extensions=True,
            ),
            "Vulkan 1.3 (all extensions)",
        ),
    )

    for config, target_fragment in cases:
        text = assemble_init_source(config, init_spec)
        assert text.startswith('"""')
        assert target_fragment in text


def test_t_18_assemble_init_source_rejects_empty_selective_names() -> None:
    assemble_init_source = _require_callable("assemble_init_source")
    config = _make_write_config()
    init_spec = _make_init_module_spec((_make_init_re_export("vk_loader", False, ()),))

    with pytest.raises(ValueError):
        assemble_init_source(config, init_spec)


def test_t_19_write_module_writes_exact_assembled_content(tmp_path: Path) -> None:
    write_module = _require_callable("write_module")
    assemble_module_source = _require_callable("assemble_module_source")
    config = _make_write_config()
    spec = _make_module_spec(
        filename="vk_structs.mojo",
        external_imports=(_make_external_import("ffi", ("c_int",)),),
        content_lines=("alias VkBool32 = UInt32",),
    )
    output_dir = tmp_path / "out"

    write_module(output_dir, config, spec)

    expected = assemble_module_source(config, spec)
    actual = (output_dir / "vk_structs.mojo").read_text(encoding="utf-8")
    assert actual == expected


def test_t_20_write_module_returns_truthful_file_metadata(tmp_path: Path) -> None:
    write_module = _require_callable("write_module")
    config = _make_write_config()
    spec = _make_module_spec(
        filename="vk_base_types.mojo",
        content_lines=("alias VkBool32 = UInt32", "alias VkFlags = UInt32"),
    )

    result = write_module(tmp_path / "out", config, spec)
    expected_path = (tmp_path / "out" / spec.filename).resolve()
    file_bytes = expected_path.read_bytes()
    file_text = file_bytes.decode("utf-8")

    assert result.filename == spec.filename
    assert result.path == expected_path
    assert result.line_count == file_text.count("\n")
    assert result.byte_count == len(file_bytes)


def test_t_21_write_module_creates_output_dir_and_propagates_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_module = _require_callable("write_module")
    config = _make_write_config()
    spec = _make_module_spec(
        filename="vk_enums.mojo",
        content_lines=("alias VkBool32 = UInt32",),
    )

    output_dir = tmp_path / "nested" / "out"
    assert not output_dir.exists()
    write_module(output_dir, config, spec)
    assert (output_dir / spec.filename).exists()

    def _raise_oserror(self: Path, *_args: object, **_kwargs: object) -> int:
        raise OSError("disk is full")

    monkeypatch.setattr(Path, "write_text", _raise_oserror)
    with pytest.raises(OSError):
        write_module(tmp_path / "second_out", config, spec)


def test_t_22_write_init_module_writes_expected_file_and_content(
    tmp_path: Path,
) -> None:
    write_init_module = _require_callable("write_init_module")
    assemble_init_source = _require_callable("assemble_init_source")
    config = _make_write_config()
    init_spec = _make_init_module_spec(
        (
            _make_init_re_export("vk_enums", True, ()),
            _make_init_re_export("vk_loader", False, ("LoadProc", "FuncPtr")),
        )
    )

    write_init_module(tmp_path / "out", config, init_spec)

    expected = assemble_init_source(config, init_spec)
    init_path = tmp_path / "out" / "__init__.mojo"
    assert init_path.exists()
    assert init_path.read_text(encoding="utf-8") == expected


def test_t_23_write_init_module_returns_truthful_file_metadata(tmp_path: Path) -> None:
    write_init_module = _require_callable("write_init_module")
    config = _make_write_config()
    init_spec = _make_init_module_spec(((_make_init_re_export("vk_enums", True, ())),))

    result = write_init_module(tmp_path / "out", config, init_spec)

    expected_path = (tmp_path / "out" / "__init__.mojo").resolve()
    file_bytes = expected_path.read_bytes()
    file_text = file_bytes.decode("utf-8")

    assert result.filename == "__init__.mojo"
    assert result.path == expected_path
    assert result.line_count == file_text.count("\n")
    assert result.byte_count == len(file_bytes)


def test_t_24_write_package_preserves_module_order_and_writes_init_last(
    tmp_path: Path,
) -> None:
    write_package = _require_callable("write_package")
    config = _make_write_config()
    module_specs = (
        _make_module_spec(filename="b.mojo", content_lines=("alias B = Int",)),
        _make_module_spec(filename="a.mojo", content_lines=("alias A = Int",)),
        _make_module_spec(filename="c.mojo", content_lines=("alias C = Int",)),
    )
    init_spec = _make_init_module_spec(((_make_init_re_export("vk_enums", True, ())),))

    result = write_package(tmp_path / "out", config, module_specs, init_spec)

    assert tuple(file_result.filename for file_result in result.files) == (
        "b.mojo",
        "a.mojo",
        "c.mojo",
        "__init__.mojo",
    )


def test_t_25_write_package_total_lines_is_sum_of_file_lines(tmp_path: Path) -> None:
    write_package = _require_callable("write_package")
    config = _make_write_config()
    module_specs = (
        _make_module_spec(filename="first.mojo", content_lines=("alias A = Int",)),
        _make_module_spec(
            filename="second.mojo",
            content_lines=("alias B = Int", "alias C = Int"),
        ),
    )
    init_spec = _make_init_module_spec(((_make_init_re_export("vk_enums", True, ())),))

    result = write_package(tmp_path / "out", config, module_specs, init_spec)

    assert result.total_lines == sum(
        file_result.line_count for file_result in result.files
    )


def test_t_26_write_package_propagates_partial_failure_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_package = _require_callable("write_package")
    _require_callable("write_module")
    config = _make_write_config()
    module_specs = (
        _make_module_spec(filename="first.mojo", content_lines=("alias A = Int",)),
        _make_module_spec(filename="second.mojo", content_lines=("alias B = Int",)),
    )
    init_spec = _make_init_module_spec(((_make_init_re_export("vk_enums", True, ())),))

    call_count = {"n": 0}
    original_write_module = gen.write_module

    def _write_then_fail(
        output_dir: Path,
        config_arg: object,
        spec_arg: object,
    ) -> object:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("simulated write failure")
        return original_write_module(output_dir, config_arg, spec_arg)

    monkeypatch.setattr(gen, "write_module", _write_then_fail)
    with pytest.raises(OSError):
        write_package(tmp_path / "out", config, module_specs, init_spec)
    assert (tmp_path / "out" / "first.mojo").exists()


def test_t_27_module_order_matches_canonical_module_constants() -> None:
    module_base_types = getattr(gen, "MODULE_BASE_TYPES", None)
    module_enums = getattr(gen, "MODULE_ENUMS", None)
    module_handles = getattr(gen, "MODULE_HANDLES", None)
    module_structs = getattr(gen, "MODULE_STRUCTS", None)
    module_unions = getattr(gen, "MODULE_UNIONS", None)
    module_loader = getattr(gen, "MODULE_LOADER", None)
    module_commands = getattr(gen, "MODULE_COMMANDS", None)
    module_order = getattr(gen, "MODULE_ORDER", None)

    expected_order = (
        module_base_types,
        module_enums,
        module_handles,
        module_structs,
        module_unions,
        module_loader,
        module_commands,
    )
    assert module_order == expected_order
    assert isinstance(module_order, tuple)
    assert len(module_order) == 7
    assert len(set(module_order)) == 7


def test_t_28_loader_selective_exports_match_canonical_six_name_api() -> None:
    expected = (
        "LoadProc",
        "FuncPtr",
        "init_vulkan",
        "init_vulkan_global",
        "init_vulkan_instance",
        "init_vulkan_device",
    )

    assert getattr(gen, "LOADER_SELECTIVE_EXPORTS", None) == expected


def test_t_29_package_module_exports_match_canonical_wildcard_selective_policy() -> (
    None
):
    init_re_export_type = _require_type("InitReExport")
    package_module_exports = getattr(gen, "PACKAGE_MODULE_EXPORTS", None)
    loader_selective_exports = getattr(gen, "LOADER_SELECTIVE_EXPORTS", None)
    module_base_types = getattr(gen, "MODULE_BASE_TYPES", None)
    module_enums = getattr(gen, "MODULE_ENUMS", None)
    module_handles = getattr(gen, "MODULE_HANDLES", None)
    module_structs = getattr(gen, "MODULE_STRUCTS", None)
    module_unions = getattr(gen, "MODULE_UNIONS", None)
    module_loader = getattr(gen, "MODULE_LOADER", None)
    module_commands = getattr(gen, "MODULE_COMMANDS", None)

    assert isinstance(package_module_exports, tuple)
    assert len(package_module_exports) == 7
    assert all(
        isinstance(entry, init_re_export_type) for entry in package_module_exports
    )

    module_stems = tuple(entry.module_stem for entry in package_module_exports)
    wildcard_flags = tuple(entry.wildcard for entry in package_module_exports)
    names_tuples = tuple(entry.names for entry in package_module_exports)

    assert module_stems == (
        module_base_types,
        module_enums,
        module_handles,
        module_structs,
        module_unions,
        module_loader,
        module_commands,
    )
    assert wildcard_flags == (True, True, True, True, True, False, True)
    assert names_tuples[5] == loader_selective_exports
    assert names_tuples[0] == ()
    assert names_tuples[1] == ()
    assert names_tuples[2] == ()
    assert names_tuples[3] == ()
    assert names_tuples[4] == ()
    assert names_tuples[6] == ()
