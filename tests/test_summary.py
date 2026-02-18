from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import gen


def _require_callable(name: str) -> Callable[..., object]:
    symbol = getattr(gen, name, None)
    assert callable(symbol), f"Missing S6 API symbol: gen.{name}"
    return symbol


def _require_type(name: str) -> type:
    symbol = getattr(gen, name, None)
    assert isinstance(symbol, type), f"Missing S6 type symbol: gen.{name}"
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


def _make_command(name: str) -> gen.CommandDef:
    return gen.CommandDef(
        name=name, return_type="VkResult", return_is_void=False, params=[]
    )


def _make_filtered_registry(
    *,
    basetypes: list[tuple[str, str, str]] | None = None,
    enum_values: dict[str, list[tuple[str, int]]] | None = None,
    block_types: dict[str, str] | None = None,
    enum_type_names: set[str] | None = None,
    handles: list[tuple[str, str, bool]] | None = None,
    handle_aliases: list[tuple[str, str]] | None = None,
    unions: list[gen.StructDef] | None = None,
    union_aliases: list[tuple[str, str]] | None = None,
    union_sizes: dict[str, int] | None = None,
    structs: list[gen.StructDef] | None = None,
    struct_aliases: list[tuple[str, str]] | None = None,
    commands: list[gen.CommandDef] | None = None,
    cmd_aliases: list[tuple[str, str]] | None = None,
) -> object:
    filtered_registry_type = _require_type("FilteredRegistry")
    return filtered_registry_type(
        basetypes=[] if basetypes is None else basetypes,
        enum_values={} if enum_values is None else enum_values,
        block_types={} if block_types is None else block_types,
        enum_type_names=set() if enum_type_names is None else enum_type_names,
        handles=[] if handles is None else handles,
        handle_aliases=[] if handle_aliases is None else handle_aliases,
        unions=[] if unions is None else unions,
        union_aliases=[] if union_aliases is None else union_aliases,
        union_sizes={} if union_sizes is None else union_sizes,
        structs=[] if structs is None else structs,
        struct_aliases=[] if struct_aliases is None else struct_aliases,
        commands=[] if commands is None else commands,
        cmd_aliases=[] if cmd_aliases is None else cmd_aliases,
    )


def _make_file_result(filename: str, line_count: int) -> object:
    file_write_result_type = _require_type("FileWriteResult")
    return file_write_result_type(
        filename=filename,
        path=Path("/tmp") / filename,
        line_count=line_count,
        byte_count=line_count * 10,
    )


def _make_package_result(output_dir: Path, files: tuple[object, ...]) -> object:
    package_write_result_type = _require_type("PackageWriteResult")
    return package_write_result_type(output_dir=output_dir, files=files)


def _make_generation_counts(
    *,
    base_types: tuple[int, int, int] = (0, 0, 0),
    enums: tuple[int, int, int] = (0, 0, 0),
    handles: tuple[int, int, int] = (0, 0, 0),
    structs: tuple[int, int, int] = (0, 0, 0),
    unions: tuple[int, int, int] = (0, 0, 0),
    commands: tuple[int, int, int] = (0, 0, 0),
) -> object:
    category_count_type = _require_type("CategoryCount")
    generation_counts_type = _require_type("GenerationCounts")

    return generation_counts_type(
        base_types=category_count_type(*base_types),
        enums=category_count_type(*enums),
        handles=category_count_type(*handles),
        structs=category_count_type(*structs),
        unions=category_count_type(*unions),
        commands=category_count_type(*commands),
    )


def _make_generation_summary(
    *,
    target_label: str = "Vulkan 1.3",
    source_label: str = "vk.xml 1.4.343",
    output_dir: str = "libs/vulkan/src/",
    counts: object | None = None,
    files: tuple[object, ...] = (),
    previous_line_count: int = 25_500,
) -> object:
    generation_summary_type = _require_type("GenerationSummary")
    return generation_summary_type(
        target_label=target_label,
        source_label=source_label,
        output_dir=output_dir,
        counts=_make_generation_counts() if counts is None else counts,
        files=files,
        previous_line_count=previous_line_count,
    )


def test_t_01_api_surface_symbols_exist() -> None:
    _require_type("CategoryCount")
    _require_type("GenerationCounts")
    _require_type("GenerationSummary")
    _require_callable("build_target_label")
    _require_callable("build_generation_counts")
    _require_callable("build_generation_summary")
    _require_callable("format_generation_summary")
    _require_callable("print_generation_summary")


def test_t_02_build_target_label_core_only_renders_version_only() -> None:
    build_target_label = _require_callable("build_target_label")
    config = _make_write_config()

    assert build_target_label(config) == "Vulkan 1.3"


def test_t_03_build_target_label_extensions_are_sorted_deterministically() -> None:
    build_target_label = _require_callable("build_target_label")
    config = _make_write_config(
        extensions=frozenset({"VK_KHR_swapchain", "VK_EXT_debug_utils"})
    )

    assert (
        build_target_label(config)
        == "Vulkan 1.3 + VK_EXT_debug_utils, VK_KHR_swapchain"
    )


def test_t_04_build_target_label_all_extensions_overrides_explicit_set() -> None:
    build_target_label = _require_callable("build_target_label")
    config = _make_write_config(
        major=1,
        minor=4,
        extensions=frozenset({"VK_EXT_debug_utils"}),
        all_extensions=True,
    )

    assert build_target_label(config) == "Vulkan 1.4 + all extensions"


def test_t_05_build_target_label_version_1_0_formats_without_special_cases() -> None:
    build_target_label = _require_callable("build_target_label")
    config = _make_write_config(major=1, minor=0)

    assert build_target_label(config) == "Vulkan 1.0"


def test_t_06_build_target_label_does_not_depend_on_vk_xml_version_field() -> None:
    build_target_label = _require_callable("build_target_label")
    config = _make_write_config(vk_xml_version="")

    assert build_target_label(config) == "Vulkan 1.3"


def test_t_07_build_generation_counts_empty_extension_sets_make_all_core(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    build_generation_counts = _require_callable("build_generation_counts")
    filtered = _make_filtered_registry(
        basetypes=[("VkBool32", "UInt32", "")],
        enum_values={"VkResult": [("VK_SUCCESS", 0)]},
        handles=[("VkInstance", "", True)],
        structs=[make_struct_def("VkExtent2D", [])],
        unions=[make_struct_def("VkClearColorValue", [])],
        commands=[_make_command("vkCreateInstance")],
    )

    counts = build_generation_counts(filtered, frozenset(), frozenset())

    for category in (
        counts.base_types,
        counts.enums,
        counts.handles,
        counts.structs,
        counts.unions,
        counts.commands,
    ):
        assert category.ext == 0
        assert category.core == category.total


def test_t_08_build_generation_counts_membership_drives_ext_attribution(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    build_generation_counts = _require_callable("build_generation_counts")
    filtered = _make_filtered_registry(
        enum_values={
            "VkCoreEnum": [("VK_CORE", 1)],
            "VkExtEnum": [("VK_EXT", 2)],
        },
        handles=[("VkCoreHandle", "", True), ("VkExtHandle", "", True)],
        structs=[
            make_struct_def("VkCoreStruct", []),
            make_struct_def("VkExtStruct", []),
        ],
        unions=[make_struct_def("VkCoreUnion", []), make_struct_def("VkExtUnion", [])],
        commands=[_make_command("vkCoreCmd"), _make_command("vkExtCmd")],
    )

    counts = build_generation_counts(
        filtered,
        frozenset({"VkExtEnum", "VkExtHandle", "VkExtStruct", "VkExtUnion"}),
        frozenset({"vkExtCmd"}),
    )

    assert counts.enums.ext == 1
    assert counts.handles.ext == 1
    assert counts.structs.ext == 1
    assert counts.unions.ext == 1
    assert counts.commands.ext == 1


def test_t_09_build_generation_counts_closure_added_types_not_in_ext_set_are_core(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    build_generation_counts = _require_callable("build_generation_counts")
    filtered = _make_filtered_registry(
        structs=[
            make_struct_def("VkCoreStruct", []),
            make_struct_def("VkDependencyFromClosure", []),
            make_struct_def("VkExtStruct", []),
        ]
    )

    counts = build_generation_counts(filtered, frozenset({"VkExtStruct"}), frozenset())

    assert counts.structs.total == 3
    assert counts.structs.ext == 1
    assert counts.structs.core == 2


def test_t_10_build_generation_counts_handles_use_name_tuple_index_zero_only() -> None:
    build_generation_counts = _require_callable("build_generation_counts")
    filtered = _make_filtered_registry(
        handles=[
            ("VkCoreHandle", "VK_EXT_debug_utils", True),
            ("VkExtHandle", "", True),
        ]
    )

    counts = build_generation_counts(
        filtered, frozenset({"VkExtHandle", "VK_EXT_debug_utils"}), frozenset()
    )

    assert counts.handles.total == 2
    assert counts.handles.ext == 1
    assert counts.handles.core == 1


def test_t_11_build_generation_counts_enums_count_enum_types_not_variants() -> None:
    build_generation_counts = _require_callable("build_generation_counts")
    filtered = _make_filtered_registry(
        enum_values={
            "VkA": [("VK_A_0", 0), ("VK_A_1", 1)],
            "VkB": [("VK_B_0", 0)],
        }
    )

    counts = build_generation_counts(filtered, frozenset({"VkB"}), frozenset())

    assert counts.enums.total == 2
    assert counts.enums.ext == 1
    assert counts.enums.core == 1


def test_t_12_build_generation_counts_zero_item_categories_return_zero_valid_counts(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    build_generation_counts = _require_callable("build_generation_counts")
    filtered = _make_filtered_registry(
        basetypes=[],
        enum_values={},
        handles=[],
        structs=[make_struct_def("VkOnlyStruct", [])],
        unions=[],
        commands=[],
    )

    counts = build_generation_counts(filtered, frozenset(), frozenset())

    assert (counts.base_types.total, counts.base_types.core, counts.base_types.ext) == (
        0,
        0,
        0,
    )
    assert (counts.enums.total, counts.enums.core, counts.enums.ext) == (0, 0, 0)
    assert (counts.handles.total, counts.handles.core, counts.handles.ext) == (0, 0, 0)
    assert (counts.unions.total, counts.unions.core, counts.unions.ext) == (0, 0, 0)
    assert (counts.commands.total, counts.commands.core, counts.commands.ext) == (
        0,
        0,
        0,
    )


def test_t_13_build_generation_counts_invariant_holds_for_all_categories(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    build_generation_counts = _require_callable("build_generation_counts")
    filtered = _make_filtered_registry(
        basetypes=[("VkA", "UInt32", ""), ("VkB", "UInt32", "")],
        enum_values={"VkCoreEnum": [], "VkExtEnum": []},
        handles=[("VkCoreHandle", "", True), ("VkExtHandle", "", True)],
        structs=[
            make_struct_def("VkCoreStruct", []),
            make_struct_def("VkExtStruct", []),
        ],
        unions=[make_struct_def("VkCoreUnion", []), make_struct_def("VkExtUnion", [])],
        commands=[_make_command("vkCoreCmd"), _make_command("vkExtCmd")],
    )

    counts = build_generation_counts(
        filtered,
        frozenset({"VkExtEnum", "VkExtHandle", "VkExtStruct", "VkExtUnion"}),
        frozenset({"vkExtCmd"}),
    )

    for category in (
        counts.base_types,
        counts.enums,
        counts.handles,
        counts.structs,
        counts.unions,
        counts.commands,
    ):
        assert category.core + category.ext == category.total


def test_t_14_build_generation_summary_copies_source_and_output_metadata(
    tmp_path: Path,
) -> None:
    build_generation_summary = _require_callable("build_generation_summary")
    write_config = _make_write_config(vk_xml_version="1.4.343")
    filtered = _make_filtered_registry()
    write_result = _make_package_result(tmp_path / "out", files=tuple())

    summary = build_generation_summary(
        write_config,
        filtered,
        frozenset(),
        frozenset(),
        write_result,
    )

    assert summary.source_label == "vk.xml 1.4.343"
    assert summary.output_dir == str(write_result.output_dir)


def test_t_15_build_generation_summary_preserves_writer_file_order(
    tmp_path: Path,
) -> None:
    build_generation_summary = _require_callable("build_generation_summary")
    write_result = _make_package_result(
        tmp_path / "out",
        files=(
            _make_file_result("vk_structs.mojo", 100),
            _make_file_result("vk_base_types.mojo", 20),
            _make_file_result("__init__.mojo", 5),
        ),
    )

    summary = build_generation_summary(
        _make_write_config(),
        _make_filtered_registry(),
        frozenset(),
        frozenset(),
        write_result,
    )

    assert tuple(file_result.filename for file_result in summary.files) == (
        "vk_structs.mojo",
        "vk_base_types.mojo",
        "__init__.mojo",
    )


def test_t_16_build_generation_summary_uses_default_previous_line_baseline(
    tmp_path: Path,
) -> None:
    build_generation_summary = _require_callable("build_generation_summary")
    expected = getattr(gen, "PREVIOUS_SINGLE_FILE_LINE_COUNT", None)
    assert expected == 25_500, "Missing/incorrect S6 baseline constant"

    summary = build_generation_summary(
        _make_write_config(),
        _make_filtered_registry(),
        frozenset(),
        frozenset(),
        _make_package_result(tmp_path / "out", files=tuple()),
    )

    assert summary.previous_line_count == 25_500


@pytest.mark.parametrize("baseline", [0, 1234])
def test_t_17_build_generation_summary_uses_explicit_previous_line_override_verbatim(
    baseline: int,
    tmp_path: Path,
) -> None:
    build_generation_summary = _require_callable("build_generation_summary")

    summary = build_generation_summary(
        _make_write_config(),
        _make_filtered_registry(),
        frozenset(),
        frozenset(),
        _make_package_result(tmp_path / "out", files=tuple()),
        previous_line_count=baseline,
    )

    assert summary.previous_line_count == baseline


def test_t_18_build_generation_summary_delegates_count_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_generation_summary = _require_callable("build_generation_summary")
    sentinel_counts = _make_generation_counts(
        base_types=(1, 1, 0),
        enums=(2, 1, 1),
        handles=(3, 2, 1),
        structs=(4, 3, 1),
        unions=(5, 5, 0),
        commands=(6, 4, 2),
    )

    monkeypatch.setattr(gen, "build_generation_counts", lambda *_args: sentinel_counts)

    summary = build_generation_summary(
        _make_write_config(),
        _make_filtered_registry(),
        frozenset(),
        frozenset(),
        _make_package_result(tmp_path / "out", files=tuple()),
    )

    assert summary.counts is sentinel_counts


def test_t_19_format_generation_summary_emits_full_section_skeleton_in_order() -> None:
    format_generation_summary = _require_callable("format_generation_summary")
    summary = _make_generation_summary(
        target_label="Vulkan 1.3 + VK_EXT_debug_utils",
        counts=_make_generation_counts(
            base_types=(1, 1, 0),
            enums=(2, 1, 1),
            handles=(1, 1, 0),
            structs=(1, 1, 0),
            unions=(0, 0, 0),
            commands=(2, 1, 1),
        ),
        files=(_make_file_result("vk_enums.mojo", 1624),),
    )

    output = format_generation_summary(summary)

    heading = output.index("Vulkan 1.3 bindings generated:")
    target = output.index("Target:")
    source = output.index("Source:")
    out_row = output.index("Output:")
    types = output.index("Types generated:")
    files = output.index("Files written:")
    total = output.index("Total:")
    verify = output.index("Verify:")

    assert heading < target < source < out_row < types < files < total < verify


def test_t_20_format_generation_summary_heading_uses_version_only_not_full_target() -> (
    None
):
    format_generation_summary = _require_callable("format_generation_summary")
    summary = _make_generation_summary(
        target_label="Vulkan 1.3 + VK_EXT_debug_utils, VK_KHR_swapchain",
    )

    output = format_generation_summary(summary)

    assert "Vulkan 1.3 bindings generated:" in output
    assert (
        "Vulkan 1.3 + VK_EXT_debug_utils, VK_KHR_swapchain bindings generated:"
        not in output
    )
    assert "Target:     Vulkan 1.3 + VK_EXT_debug_utils, VK_KHR_swapchain" in output


def test_t_21_format_generation_summary_split_suffix_appears_only_when_ext_positive() -> (
    None
):
    format_generation_summary = _require_callable("format_generation_summary")
    summary = _make_generation_summary(
        counts=_make_generation_counts(
            base_types=(7, 7, 0),
            enums=(128, 116, 12),
            handles=(30, 30, 0),
            structs=(320, 294, 26),
            unions=(2, 2, 0),
            commands=(235, 215, 20),
        )
    )

    output = format_generation_summary(summary)

    assert "Enums:" in output and "(116 core + 12 from extensions)" in output
    assert "Structs:" in output and "(294 core + 26 from extensions)" in output
    assert "Commands:" in output and "(215 core + 20 from extensions)" in output
    assert "Base types:" in output and "Base types:     7\n" in output


def test_t_22_format_generation_summary_uses_thousands_separators_for_lines_and_totals() -> (
    None
):
    format_generation_summary = _require_callable("format_generation_summary")
    summary = _make_generation_summary(
        files=(
            _make_file_result("vk_enums.mojo", 1624),
            _make_file_result("vk_structs.mojo", 3842),
            _make_file_result("vk_commands.mojo", 2714),
            _make_file_result("__init__.mojo", 14),
        ),
        previous_line_count=25_500,
    )

    output = format_generation_summary(summary)

    assert "1,624 lines" in output
    assert "3,842 lines" in output
    assert "2,714 lines" in output
    assert "8,194" in output
    assert "25,500" in output


def test_t_23_format_generation_summary_files_section_preserves_input_tuple_order() -> (
    None
):
    format_generation_summary = _require_callable("format_generation_summary")
    summary = _make_generation_summary(
        files=(
            _make_file_result("vk_structs.mojo", 10),
            _make_file_result("vk_base_types.mojo", 20),
            _make_file_result("__init__.mojo", 30),
        )
    )

    output = format_generation_summary(summary)

    struct_i = output.index("vk_structs.mojo")
    base_i = output.index("vk_base_types.mojo")
    init_i = output.index("__init__.mojo")
    assert struct_i < base_i < init_i


@pytest.mark.parametrize("out_dir", ["/tmp/out", "/tmp/out/"])
def test_t_24_format_generation_summary_verify_command_uses_output_dir_verbatim(
    out_dir: str,
) -> None:
    format_generation_summary = _require_callable("format_generation_summary")
    summary = _make_generation_summary(output_dir=out_dir)

    output = format_generation_summary(summary)

    assert f"Verify: mojo package {out_dir} -o /tmp/vulkan.mojopkg" in output


def test_t_25_format_generation_summary_is_deterministic_and_single_newline_terminated() -> (
    None
):
    format_generation_summary = _require_callable("format_generation_summary")
    summary = _make_generation_summary(files=(_make_file_result("vk_a.mojo", 1),))

    first = format_generation_summary(summary)
    second = format_generation_summary(summary)

    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


def test_t_26_print_generation_summary_prints_formatter_output_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_generation_summary = _require_callable("print_generation_summary")
    sentinel = "line one\nline two\n"

    monkeypatch.setattr(gen, "format_generation_summary", lambda _summary: sentinel)

    print_generation_summary(_make_generation_summary())
    output = capsys.readouterr().out

    assert output == sentinel
