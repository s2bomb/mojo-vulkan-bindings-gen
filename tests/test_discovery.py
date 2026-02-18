from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import gen


def _require_callable(name: str) -> Callable[..., object]:
    symbol = getattr(gen, name, None)
    assert callable(symbol), f"Missing S3 API symbol: gen.{name}"
    return symbol


def _require_type(name: str) -> type:
    symbol = getattr(gen, name, None)
    assert isinstance(symbol, type), f"Missing S3 type symbol: gen.{name}"
    return symbol


def _make_version_summary(
    *,
    major: int,
    minor: int,
    delta_type_count: int,
    delta_command_count: int,
    cumulative_type_count: int,
    cumulative_command_count: int,
    is_base: bool,
) -> object:
    version_summary_type = _require_type("VersionSummary")
    return version_summary_type(
        version=gen.VulkanVersion(major, minor),
        delta_type_count=delta_type_count,
        delta_command_count=delta_command_count,
        cumulative_type_count=cumulative_type_count,
        cumulative_command_count=cumulative_command_count,
        is_base=is_base,
    )


def _make_extension_summary(
    *,
    name: str,
    ext_type: str,
    type_count: int,
    command_count: int,
    depends_raw: str,
    promoted_to: str | None,
) -> object:
    extension_summary_type = _require_type("ExtensionSummary")
    return extension_summary_type(
        name=name,
        ext_type=ext_type,
        type_count=type_count,
        command_count=command_count,
        depends_raw=depends_raw,
        promoted_to=promoted_to,
    )


def _make_type_entry(name: str, category: str) -> object:
    type_entry_type = _require_type("TypeEntry")
    return type_entry_type(name=name, category=category)


def _make_extension_detail(
    *,
    summary: object,
    types: tuple[object, ...],
    commands: tuple[str, ...],
) -> object:
    extension_detail_type = _require_type("ExtensionDetail")
    return extension_detail_type(summary=summary, types=types, commands=commands)


def _write_registry(tmp_path: Path, inner_xml: str) -> Path:
    vk_xml = tmp_path / "vk.xml"
    vk_xml.write_text(f"<registry>{inner_xml}</registry>\n", encoding="utf-8")
    return vk_xml


def _make_discovery_config(
    *,
    command: str,
    vk_xml: Path,
    filter_text: str | None,
    info_extension: str | None,
) -> object:
    discovery_config_type = _require_type("DiscoveryConfig")
    return discovery_config_type(
        command=command,
        filter_text=filter_text,
        info_extension=info_extension,
        vk_xml=vk_xml,
    )


def test_t_01_extract_registry_version_returns_major_minor_patch_when_available(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    extract_registry_version = _require_callable("extract_registry_version")
    root = make_registry_root(
        """
        <feature api="vulkan" number="1.3"/>
        <feature api="vulkan" number="1.4"/>
        <enums name="API Constants">
            <enum name="VK_HEADER_VERSION" value="343"/>
        </enums>
        """
    )

    assert extract_registry_version(root) == "1.4.343"


def test_t_02_extract_registry_version_returns_major_minor_when_patch_missing(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    extract_registry_version = _require_callable("extract_registry_version")
    root = make_registry_root(
        """
        <feature api="vulkan" number="1.2"/>
        <feature api="vulkan" number="1.4"/>
        <enums name="API Constants">
            <enum name="VK_SOME_OTHER_CONSTANT" value="1"/>
        </enums>
        """
    )

    assert extract_registry_version(root) == "1.4"


def test_t_03_extract_registry_version_returns_unknown_without_vulkan_features(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    extract_registry_version = _require_callable("extract_registry_version")
    root = make_registry_root('<feature api="vulkansc" number="1.0"/>')

    assert extract_registry_version(root) == "unknown"


def test_t_04_gather_version_summaries_returns_fixed_ordered_rows(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_version_summaries = _require_callable("gather_version_summaries")
    all_versions = getattr(gen, "ALL_VULKAN_VERSIONS", None)
    assert isinstance(
        all_versions, tuple
    ), "Missing S3 constant: gen.ALL_VULKAN_VERSIONS"

    root = make_registry_root(
        """
        <feature api="vulkan" number="1.0"><require><type name="Vk10"/></require></feature>
        <feature api="vulkan" number="1.2"><require><type name="Vk12"/></require></feature>
        """
    )
    summaries = gather_version_summaries(root)

    assert len(summaries) == len(all_versions)
    assert [getattr(row, "version") for row in summaries] == list(all_versions)


def test_t_05_gather_version_summaries_delta_cumulative_math_is_consistent(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_version_summaries = _require_callable("gather_version_summaries")
    root = make_registry_root(
        """
        <feature api="vulkan" number="1.0">
            <require>
                <type name="VkA"/>
                <command name="vkA"/>
            </require>
        </feature>
        <feature api="vulkan" number="1.1">
            <require>
                <type name="VkB"/>
                <command name="vkB"/>
            </require>
        </feature>
        <feature api="vulkan" number="1.2">
            <require>
                <type name="VkC"/>
                <command name="vkC"/>
            </require>
        </feature>
        """
    )
    summaries = gather_version_summaries(root)

    for index in range(1, len(summaries)):
        prev = summaries[index - 1]
        curr = summaries[index]
        assert getattr(curr, "cumulative_type_count") == getattr(
            prev, "cumulative_type_count"
        ) + getattr(curr, "delta_type_count")
        assert getattr(curr, "cumulative_command_count") == getattr(
            prev, "cumulative_command_count"
        ) + getattr(curr, "delta_command_count")


def test_t_06_gather_version_summaries_marks_only_1_0_as_base(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_version_summaries = _require_callable("gather_version_summaries")
    root = make_registry_root('<feature api="vulkan" number="1.0"/>')
    summaries = gather_version_summaries(root)

    base_rows = [row for row in summaries if getattr(row, "is_base")]
    assert len(base_rows) == 1
    assert getattr(base_rows[0], "version") == gen.VulkanVersion(1, 0)


def test_t_07_gather_extension_summaries_filters_to_vulkan_and_sorts(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_extension_summaries = _require_callable("gather_extension_summaries")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_Z_ONLY_SC" supported="vulkansc"/>
            <extension name="VK_B_MIXED" supported="vulkan,vulkansc"/>
            <extension name="VK_A_VK" supported="vulkan"/>
        </extensions>
        """
    )
    summaries = gather_extension_summaries(root)
    names = [getattr(summary, "name") for summary in summaries]

    assert names == ["VK_A_VK", "VK_B_MIXED"]


def test_t_08_gather_extension_summaries_counts_raw_require_members(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_extension_summaries = _require_callable("gather_extension_summaries")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_EXT_RAW_COUNTS" supported="vulkan">
                <require>
                    <type name="VkA"/>
                    <command name="vkA"/>
                </require>
                <require depends="VK_VERSION_1_3">
                    <type name="VkB"/>
                    <command name="vkB"/>
                </require>
            </extension>
        </extensions>
        """
    )

    summaries = gather_extension_summaries(root)
    assert len(summaries) == 1
    assert getattr(summaries[0], "type_count") == 2
    assert getattr(summaries[0], "command_count") == 2


def test_t_09_gather_extension_summaries_maps_metadata_fields_verbatim(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_extension_summaries = _require_callable("gather_extension_summaries")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan"/>
            <extension
                name="VK_B"
                supported="vulkan"
                type="device"
                depends="VK_KHR_surface+VK_VERSION_1_1"
            />
            <extension
                name="VK_C"
                supported="vulkan"
                promotedto="VK_VERSION_1_3"
            />
        </extensions>
        """
    )
    summaries = gather_extension_summaries(root)
    by_name = {getattr(summary, "name"): summary for summary in summaries}

    assert getattr(by_name["VK_A"], "ext_type") == ""
    assert getattr(by_name["VK_A"], "depends_raw") == ""
    assert getattr(by_name["VK_A"], "promoted_to") is None
    assert getattr(by_name["VK_B"], "ext_type") == "device"
    assert getattr(by_name["VK_B"], "depends_raw") == "VK_KHR_surface+VK_VERSION_1_1"
    assert getattr(by_name["VK_C"], "promoted_to") == "VK_VERSION_1_3"


def test_t_10_gather_extension_summaries_empty_section_returns_empty(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_extension_summaries = _require_callable("gather_extension_summaries")
    root = make_registry_root("<extensions />")

    assert gather_extension_summaries(root) == []


def test_t_11_filter_extensions_by_text_is_case_insensitive_and_stable() -> None:
    filter_extensions_by_text = _require_callable("filter_extensions_by_text")
    summaries = [
        _make_extension_summary(
            name="VK_KHR_swapchain",
            ext_type="device",
            type_count=13,
            command_count=9,
            depends_raw="",
            promoted_to=None,
        ),
        _make_extension_summary(
            name="VK_EXT_swapchain_colorspace",
            ext_type="instance",
            type_count=5,
            command_count=2,
            depends_raw="",
            promoted_to=None,
        ),
        _make_extension_summary(
            name="VK_KHR_surface",
            ext_type="instance",
            type_count=7,
            command_count=3,
            depends_raw="",
            promoted_to=None,
        ),
    ]

    filtered = filter_extensions_by_text(summaries, "SwapChain")

    assert [getattr(summary, "name") for summary in filtered] == [
        "VK_KHR_swapchain",
        "VK_EXT_swapchain_colorspace",
    ]


def test_t_12_filter_extensions_by_text_empty_filter_is_no_op() -> None:
    filter_extensions_by_text = _require_callable("filter_extensions_by_text")
    summaries = [
        _make_extension_summary(
            name="VK_A",
            ext_type="",
            type_count=1,
            command_count=1,
            depends_raw="",
            promoted_to=None,
        ),
        _make_extension_summary(
            name="VK_B",
            ext_type="",
            type_count=2,
            command_count=2,
            depends_raw="",
            promoted_to=None,
        ),
        _make_extension_summary(
            name="VK_C",
            ext_type="",
            type_count=3,
            command_count=3,
            depends_raw="",
            promoted_to=None,
        ),
    ]

    filtered = filter_extensions_by_text(summaries, "")

    assert filtered == summaries


def test_t_13_filter_extensions_by_text_no_match_returns_empty() -> None:
    filter_extensions_by_text = _require_callable("filter_extensions_by_text")
    summaries = [
        _make_extension_summary(
            name="VK_KHR_surface",
            ext_type="instance",
            type_count=7,
            command_count=3,
            depends_raw="",
            promoted_to=None,
        )
    ]

    assert filter_extensions_by_text(summaries, "swapchain") == []


def test_t_14_gather_extension_detail_unknown_extension_returns_none(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_extension_detail = _require_callable("gather_extension_detail")
    root = make_registry_root(
        '<extensions><extension name="VK_A" supported="vulkan"/></extensions>'
    )

    assert gather_extension_detail(root, "VK_NOT_PRESENT") is None


def test_t_15_gather_extension_detail_deduplicates_and_keeps_counts_consistent(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_extension_detail = _require_callable("gather_extension_detail")
    root = make_registry_root(
        """
        <types>
            <type category="handle" name="VkA"/>
            <type category="struct" name="VkB"/>
        </types>
        <extensions>
            <extension name="VK_A" supported="vulkan" type="device">
                <require>
                    <type name="VkA"/>
                    <command name="vkA"/>
                </require>
                <require>
                    <type name="VkA"/>
                    <type name="VkB"/>
                    <command name="vkA"/>
                    <command name="vkB"/>
                </require>
            </extension>
        </extensions>
        """
    )

    detail = gather_extension_detail(root, "VK_A")
    assert detail is not None

    types = getattr(detail, "types")
    commands = getattr(detail, "commands")
    summary = getattr(detail, "summary")
    assert len(types) == 2
    assert len(commands) == 2
    assert len(types) == getattr(summary, "type_count")
    assert len(commands) == getattr(summary, "command_count")


def test_t_16_gather_extension_detail_maps_type_categories_and_keeps_unknowns(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_extension_detail = _require_callable("gather_extension_detail")
    root = make_registry_root(
        """
        <types>
            <type category="handle" name="VkKnownHandle"/>
            <type category="struct" name="VkKnownStruct"/>
        </types>
        <extensions>
            <extension name="VK_A" supported="vulkan">
                <require>
                    <type name="VkKnownHandle"/>
                    <type name="VkKnownStruct"/>
                    <type name="VkUnknownType"/>
                </require>
            </extension>
        </extensions>
        """
    )

    detail = gather_extension_detail(root, "VK_A")
    assert detail is not None

    categories = {
        getattr(entry, "name"): getattr(entry, "category")
        for entry in getattr(detail, "types")
    }
    assert categories["VkKnownHandle"] == "handle"
    assert categories["VkKnownStruct"] == "struct"
    assert categories["VkUnknownType"] == ""


def test_t_17_gather_extension_detail_preserves_first_appearance_order(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    gather_extension_detail = _require_callable("gather_extension_detail")
    root = make_registry_root(
        """
        <types>
            <type category="struct" name="VkB"/>
            <type category="struct" name="VkA"/>
            <type category="struct" name="VkC"/>
        </types>
        <extensions>
            <extension name="VK_ORDER" supported="vulkan">
                <require>
                    <type name="VkB"/>
                    <type name="VkA"/>
                    <command name="vkB"/>
                    <command name="vkA"/>
                </require>
                <require>
                    <type name="VkC"/>
                    <type name="VkA"/>
                    <command name="vkC"/>
                    <command name="vkA"/>
                </require>
            </extension>
        </extensions>
        """
    )

    detail = gather_extension_detail(root, "VK_ORDER")
    assert detail is not None
    assert [getattr(entry, "name") for entry in getattr(detail, "types")] == [
        "VkB",
        "VkA",
        "VkC",
    ]
    assert list(getattr(detail, "commands")) == ["vkB", "vkA", "vkC"]


def test_t_18_format_versions_table_header_and_labels_match_contract() -> None:
    format_versions_table = _require_callable("format_versions_table")
    summaries = [
        _make_version_summary(
            major=1,
            minor=0,
            delta_type_count=538,
            delta_command_count=215,
            cumulative_type_count=538,
            cumulative_command_count=215,
            is_base=True,
        ),
        _make_version_summary(
            major=1,
            minor=1,
            delta_type_count=104,
            delta_command_count=28,
            cumulative_type_count=642,
            cumulative_command_count=243,
            is_base=False,
        ),
        _make_version_summary(
            major=1,
            minor=2,
            delta_type_count=62,
            delta_command_count=13,
            cumulative_type_count=704,
            cumulative_command_count=256,
            is_base=False,
        ),
    ]

    output = format_versions_table(summaries, "1.4.343")

    assert output.startswith("Vulkan versions in vk.xml 1.4.343:")
    assert "(base)" in output
    assert "(642 total)" in output


def test_t_19_format_versions_table_plus_prefix_applies_only_to_delta_rows() -> None:
    format_versions_table = _require_callable("format_versions_table")
    summaries = [
        _make_version_summary(
            major=1,
            minor=0,
            delta_type_count=10,
            delta_command_count=5,
            cumulative_type_count=10,
            cumulative_command_count=5,
            is_base=True,
        ),
        _make_version_summary(
            major=1,
            minor=1,
            delta_type_count=2,
            delta_command_count=1,
            cumulative_type_count=12,
            cumulative_command_count=6,
            is_base=False,
        ),
    ]

    output = format_versions_table(summaries, "1.4.343")
    row_lines = [line for line in output.splitlines() if line.strip().startswith("1.")]
    row_by_version = {line.strip().split()[0]: line for line in row_lines}

    assert "+" not in row_by_version["1.0"]
    assert "+" in row_by_version["1.1"]


def test_t_20_format_versions_table_is_deterministic_and_newline_terminated() -> None:
    format_versions_table = _require_callable("format_versions_table")
    summaries = [
        _make_version_summary(
            major=1,
            minor=0,
            delta_type_count=1,
            delta_command_count=1,
            cumulative_type_count=1,
            cumulative_command_count=1,
            is_base=True,
        )
    ]

    first = format_versions_table(summaries, "1.0.0")
    second = format_versions_table(summaries, "1.0.0")

    assert first == second
    assert first.endswith("\n")
    assert not first.endswith("\n\n")


def test_t_21_format_extensions_table_header_count_matches_input_size() -> None:
    format_extensions_table = _require_callable("format_extensions_table")
    summaries = [
        _make_extension_summary(
            name="VK_A",
            ext_type="device",
            type_count=1,
            command_count=1,
            depends_raw="",
            promoted_to=None,
        ),
        _make_extension_summary(
            name="VK_B",
            ext_type="instance",
            type_count=2,
            command_count=2,
            depends_raw="",
            promoted_to=None,
        ),
        _make_extension_summary(
            name="VK_C",
            ext_type="",
            type_count=3,
            command_count=3,
            depends_raw="",
            promoted_to=None,
        ),
    ]

    output = format_extensions_table(summaries, "1.4.343")

    assert output.startswith("3 Vulkan extensions in vk.xml 1.4.343:")


def test_t_22_format_extensions_table_annotation_precedence_is_promoted_then_depends() -> (
    None
):
    format_extensions_table = _require_callable("format_extensions_table")
    summaries = [
        _make_extension_summary(
            name="VK_A",
            ext_type="device",
            type_count=1,
            command_count=1,
            depends_raw="VK_KHR_surface",
            promoted_to="VK_VERSION_1_3",
        ),
        _make_extension_summary(
            name="VK_B",
            ext_type="instance",
            type_count=1,
            command_count=1,
            depends_raw="VK_KHR_surface",
            promoted_to=None,
        ),
        _make_extension_summary(
            name="VK_C",
            ext_type="",
            type_count=1,
            command_count=1,
            depends_raw="",
            promoted_to=None,
        ),
    ]

    output = format_extensions_table(summaries, "1.4.343")
    lines = output.splitlines()
    row_a = next(line for line in lines if "VK_A" in line)
    row_b = next(line for line in lines if "VK_B" in line)
    row_c = next(line for line in lines if "VK_C" in line)

    assert "promoted:" in row_a
    assert "depends:" not in row_a
    assert "depends:" in row_b
    assert "promoted:" not in row_b
    assert "depends:" not in row_c
    assert "promoted:" not in row_c


def test_t_23_format_extensions_table_does_not_filter_rows() -> None:
    format_extensions_table = _require_callable("format_extensions_table")
    summaries = [
        _make_extension_summary(
            name="VK_KHR_surface",
            ext_type="instance",
            type_count=1,
            command_count=1,
            depends_raw="",
            promoted_to=None,
        ),
        _make_extension_summary(
            name="VK_KHR_swapchain",
            ext_type="device",
            type_count=2,
            command_count=2,
            depends_raw="VK_KHR_surface",
            promoted_to=None,
        ),
    ]

    output = format_extensions_table(summaries, "1.4.343")

    assert "2 Vulkan extensions in vk.xml 1.4.343:" in output
    assert "VK_KHR_surface" in output
    assert "VK_KHR_swapchain" in output


def test_t_24_format_extension_detail_header_and_promoted_no_rendering() -> None:
    format_extension_detail = _require_callable("format_extension_detail")
    summary = _make_extension_summary(
        name="VK_NAME",
        ext_type="device",
        type_count=1,
        command_count=1,
        depends_raw="",
        promoted_to=None,
    )
    detail = _make_extension_detail(
        summary=summary,
        types=(_make_type_entry("VkThing", "struct"),),
        commands=("vkThing",),
    )

    output = format_extension_detail(detail)
    lines = output.splitlines()

    assert lines[0] == "VK_NAME (device extension)"
    assert "Promoted: no" in output


def test_t_25_format_extension_detail_depends_line_strips_version_tokens() -> None:
    format_extension_detail = _require_callable("format_extension_detail")
    summary = _make_extension_summary(
        name="VK_NAME",
        ext_type="device",
        type_count=1,
        command_count=1,
        depends_raw="VK_KHR_surface+VK_VERSION_1_1",
        promoted_to=None,
    )
    detail = _make_extension_detail(
        summary=summary,
        types=(_make_type_entry("VkThing", "struct"),),
        commands=("vkThing",),
    )

    output = format_extension_detail(detail)

    assert "Depends:" in output
    assert "VK_KHR_surface" in output
    assert "VK_VERSION_1_1" not in output


def test_t_26_format_extension_detail_section_counts_match_payload_lengths() -> None:
    format_extension_detail = _require_callable("format_extension_detail")
    summary = _make_extension_summary(
        name="VK_NAME",
        ext_type="instance",
        type_count=2,
        command_count=0,
        depends_raw="",
        promoted_to=None,
    )
    detail = _make_extension_detail(
        summary=summary,
        types=(
            _make_type_entry("VkA", "struct"),
            _make_type_entry("VkB", "handle"),
        ),
        commands=(),
    )

    output = format_extension_detail(detail)

    assert "Types (2):" in output
    assert "Commands (0):" in output


def test_t_27_run_discovery_list_versions_prints_to_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_discovery = _require_callable("run_discovery")
    vk_xml = _write_registry(
        tmp_path,
        """
        <feature api="vulkan" number="1.4"/>
        <enums name="API Constants"><enum name="VK_HEADER_VERSION" value="343"/></enums>
        """,
    )
    config = _make_discovery_config(
        command="list-versions",
        filter_text=None,
        info_extension=None,
        vk_xml=vk_xml,
    )

    run_discovery(config)
    captured = capsys.readouterr()

    assert "Vulkan versions in vk.xml" in captured.out
    assert captured.err == ""


def test_t_28_run_discovery_list_extensions_applies_filter_only_when_present(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_discovery = _require_callable("run_discovery")
    vk_xml = _write_registry(
        tmp_path,
        """
        <feature api="vulkan" number="1.4"/>
        <enums name="API Constants"><enum name="VK_HEADER_VERSION" value="343"/></enums>
        <extensions>
            <extension name="VK_KHR_surface" supported="vulkan"/>
            <extension name="VK_KHR_swapchain" supported="vulkan"/>
            <extension name="VK_EXT_swapchain_colorspace" supported="vulkan"/>
        </extensions>
        """,
    )

    unfiltered_config = _make_discovery_config(
        command="list-extensions",
        filter_text=None,
        info_extension=None,
        vk_xml=vk_xml,
    )
    run_discovery(unfiltered_config)
    unfiltered = capsys.readouterr()

    filtered_config = _make_discovery_config(
        command="list-extensions",
        filter_text="swapchain",
        info_extension=None,
        vk_xml=vk_xml,
    )
    run_discovery(filtered_config)
    filtered = capsys.readouterr()

    assert "3 Vulkan extensions in vk.xml" in unfiltered.out
    assert "VK_KHR_surface" in unfiltered.out
    assert "2 Vulkan extensions in vk.xml" in filtered.out
    assert "VK_KHR_surface" not in filtered.out
    assert "VK_KHR_swapchain" in filtered.out
    assert "VK_EXT_swapchain_colorspace" in filtered.out


def test_t_29_run_discovery_info_unknown_extension_exits_with_canonical_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_discovery = _require_callable("run_discovery")
    vk_xml = _write_registry(
        tmp_path,
        """
        <feature api="vulkan" number="1.4"/>
        <enums name="API Constants"><enum name="VK_HEADER_VERSION" value="343"/></enums>
        <extensions>
            <extension name="VK_KHR_surface" supported="vulkan"/>
        </extensions>
        """,
    )
    missing_name = "VK_NOT_REAL_EXT"
    config = _make_discovery_config(
        command="info",
        filter_text=None,
        info_extension=missing_name,
        vk_xml=vk_xml,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_discovery(config)
    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert captured.out == ""
    assert (
        captured.err
        == f"Error: extension '{missing_name}' not found in vk.xml 1.4.343\n"
    )


def test_t_30_run_discovery_propagates_xml_parse_errors(
    tmp_path: Path,
) -> None:
    run_discovery = _require_callable("run_discovery")
    malformed = tmp_path / "bad.xml"
    malformed.write_text("<registry><feature>", encoding="utf-8")
    config = _make_discovery_config(
        command="list-versions",
        filter_text=None,
        info_extension=None,
        vk_xml=malformed,
    )

    with pytest.raises(ET.ParseError):
        run_discovery(config)


def test_t_31_run_discovery_emits_only_result_output_without_generation_chatter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_discovery = _require_callable("run_discovery")
    vk_xml = _write_registry(
        tmp_path,
        """
        <types>
            <type category="struct" name="VkSurfaceKHR"/>
        </types>
        <feature api="vulkan" number="1.4"/>
        <enums name="API Constants"><enum name="VK_HEADER_VERSION" value="343"/></enums>
        <extensions>
            <extension name="VK_KHR_surface" supported="vulkan" type="instance">
                <require>
                    <type name="VkSurfaceKHR"/>
                    <command name="vkDestroySurfaceKHR"/>
                </require>
            </extension>
        </extensions>
        """,
    )

    commands = [
        _make_discovery_config(
            command="list-versions",
            filter_text=None,
            info_extension=None,
            vk_xml=vk_xml,
        ),
        _make_discovery_config(
            command="list-extensions",
            filter_text=None,
            info_extension=None,
            vk_xml=vk_xml,
        ),
        _make_discovery_config(
            command="info",
            filter_text=None,
            info_extension="VK_KHR_surface",
            vk_xml=vk_xml,
        ),
    ]

    for config in commands:
        run_discovery(config)
        captured = capsys.readouterr()
        assert "Parsing:" not in captured.out
        assert "Type registry:" not in captured.out
        assert "Command classification:" not in captured.out
        assert captured.err == ""

    run_discovery(commands[0])
    versions = capsys.readouterr()
    assert "Vulkan versions in vk.xml" in versions.out

    run_discovery(commands[1])
    extensions = capsys.readouterr()
    assert "Vulkan extensions in vk.xml" in extensions.out

    run_discovery(commands[2])
    detail = capsys.readouterr()
    assert "VK_KHR_surface (instance extension)" in detail.out


def test_main_discovery_delegates_to_run_discovery_not_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Regression guard: main() must call run_discovery, not print the old stub."""
    vk_xml = _write_registry(
        tmp_path,
        """
        <feature api="vulkan" number="1.4"/>
        <enums name="API Constants"><enum name="VK_HEADER_VERSION" value="343"/></enums>
        """,
    )
    monkeypatch.setattr(
        sys, "argv", ["gen.py", "--list-versions", "--vk-xml", str(vk_xml)]
    )

    gen.main()
    captured = capsys.readouterr()

    assert "not implemented" not in captured.out
    assert "Vulkan versions in vk.xml 1.4.343" in captured.out
    assert captured.err == ""
