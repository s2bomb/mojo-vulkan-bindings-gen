from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import gen


def _require_callable(name: str) -> Callable[..., object]:
    symbol = getattr(gen, name, None)
    assert callable(symbol), f"Missing S2 API symbol: gen.{name}"
    return symbol


def _require_type(name: str) -> type:
    symbol = getattr(gen, name, None)
    assert isinstance(symbol, type), f"Missing S2 type symbol: gen.{name}"
    return symbol


@pytest.mark.parametrize("depends_str", ["", "   "])
def test_t_01_parse_extension_depends_blank_returns_empty(depends_str: str) -> None:
    parse_extension_depends = _require_callable("parse_extension_depends")

    assert parse_extension_depends(depends_str) == frozenset()


@pytest.mark.parametrize(
    ("depends_str", "expected"),
    [
        ("VK_A+VK_B", frozenset({"VK_A", "VK_B"})),
        ("VK_A,VK_B", frozenset({"VK_A", "VK_B"})),
        ("VK_A+VK_B,VK_C", frozenset({"VK_A", "VK_B", "VK_C"})),
    ],
)
def test_t_02_parse_extension_depends_flattens_and_or(
    depends_str: str,
    expected: frozenset[str],
) -> None:
    parse_extension_depends = _require_callable("parse_extension_depends")

    assert parse_extension_depends(depends_str) == expected


def test_t_03_parse_extension_depends_deduplicates_tokens() -> None:
    parse_extension_depends = _require_callable("parse_extension_depends")

    assert parse_extension_depends("VK_A+VK_A,VK_A") == frozenset({"VK_A"})


def test_t_04_parse_extension_depends_does_not_over_filter_tokens() -> None:
    parse_extension_depends = _require_callable("parse_extension_depends")

    parsed = parse_extension_depends("VK_KHR_surface+VK_EXT_debug_utils")
    assert parsed == frozenset({"VK_KHR_surface", "VK_EXT_debug_utils"})


def test_t_05_all_vulkan_extension_names_filters_by_supported(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    all_vulkan_extension_names = _require_callable("all_vulkan_extension_names")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan"/>
            <extension name="VK_B" supported="vulkansc"/>
            <extension name="VK_C" supported="disabled"/>
            <extension name="VK_D" supported="vulkan,vulkansc"/>
        </extensions>
        """
    )

    assert all_vulkan_extension_names(root) == frozenset({"VK_A", "VK_D"})


def test_t_06_all_vulkan_extension_names_empty_section_returns_empty(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    all_vulkan_extension_names = _require_callable("all_vulkan_extension_names")
    root = make_registry_root("<extensions />")

    assert all_vulkan_extension_names(root) == frozenset()


def test_t_07_resolve_extension_deps_empty_input_returns_empty(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root("<extensions />")

    assert resolve_extension_deps(root, frozenset()) == frozenset()


def test_t_08_resolve_extension_deps_resolves_transitive_chain(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_C" supported="vulkan"/>
            <extension name="VK_A" supported="vulkan" depends="VK_C"/>
            <extension name="VK_B" supported="vulkan" depends="VK_A"/>
        </extensions>
        """
    )

    assert resolve_extension_deps(root, frozenset({"VK_B"})) == frozenset(
        {"VK_A", "VK_B", "VK_C"}
    )


def test_t_09_resolve_extension_deps_unknown_names_pass_through(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        '<extensions><extension name="VK_A" supported="vulkan"/></extensions>'
    )

    resolved = resolve_extension_deps(root, frozenset({"VK_NONEXISTENT"}))
    assert "VK_NONEXISTENT" in resolved


def test_t_10_resolve_extension_deps_cycles_terminate(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan" depends="VK_B"/>
            <extension name="VK_B" supported="vulkan" depends="VK_A"/>
        </extensions>
        """
    )

    assert resolve_extension_deps(root, frozenset({"VK_A"})) == frozenset(
        {"VK_A", "VK_B"}
    )


def test_t_11_resolve_extension_deps_is_idempotent(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_C" supported="vulkan"/>
            <extension name="VK_B" supported="vulkan" depends="VK_C"/>
            <extension name="VK_A" supported="vulkan" depends="VK_B"/>
        </extensions>
        """
    )

    resolved_once = resolve_extension_deps(root, frozenset({"VK_A"}))
    resolved_twice = resolve_extension_deps(root, resolved_once)
    assert resolved_once == resolved_twice


def _version_fixture_root(
    make_registry_root: Callable[[str], ET.Element],
) -> ET.Element:
    return make_registry_root(
        """
        <feature api="vulkan" number="1.0">
            <require>
                <type name="VkType10"/>
                <command name="vkCmd10"/>
            </require>
        </feature>
        <feature api="vulkan" number="1.1">
            <require>
                <type name="VkType11"/>
                <command name="vkCmd11"/>
            </require>
        </feature>
        <feature api="vulkan" number="1.3">
            <require>
                <type name="VkType13"/>
                <command name="vkCmd13"/>
            </require>
        </feature>
        <feature api="vulkansc" number="1.3">
            <require>
                <type name="VkScOnly"/>
                <command name="vkScOnly"/>
            </require>
        </feature>
        """
    )


def test_t_12_collect_version_types_collects_all_lower_or_equal_versions(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    collect_version_types = _require_callable("collect_version_types")
    root = _version_fixture_root(make_registry_root)

    collected = collect_version_types(root, gen.VulkanVersion(1, 3))
    assert collected == frozenset({"VkType10", "VkType11", "VkType13"})


def test_t_13_collect_version_types_excludes_higher_versions(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    collect_version_types = _require_callable("collect_version_types")
    root = _version_fixture_root(make_registry_root)

    collected = collect_version_types(root, gen.VulkanVersion(1, 2))
    assert "VkType13" not in collected


def test_t_14_collect_version_commands_mirrors_version_type_selection(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    collect_version_commands = _require_callable("collect_version_commands")
    root = _version_fixture_root(make_registry_root)

    collected = collect_version_commands(root, gen.VulkanVersion(1, 1))
    assert collected == frozenset({"vkCmd10", "vkCmd11"})


def test_t_15_collect_version_skips_malformed_entries_missing_name(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    collect_version_types = _require_callable("collect_version_types")
    collect_version_commands = _require_callable("collect_version_commands")
    root = make_registry_root(
        """
        <feature api="vulkan" number="1.1">
            <require>
                <type/>
                <command/>
                <type name="VkGoodType"/>
                <command name="vkGoodCommand"/>
            </require>
        </feature>
        """
    )

    assert collect_version_types(root, gen.VulkanVersion(1, 1)) == frozenset(
        {"VkGoodType"}
    )
    assert collect_version_commands(root, gen.VulkanVersion(1, 1)) == frozenset(
        {"vkGoodCommand"}
    )


def test_t_16_collect_extension_types_filters_by_name_and_includes_satisfied_requires(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    collect_extension_types = _require_callable("collect_extension_types")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan">
                <require><type name="VkA"/></require>
                <require depends="VK_VERSION_1_1"><type name="VkA11"/></require>
            </extension>
            <extension name="VK_B" supported="vulkan">
                <require><type name="VkB"/></require>
            </extension>
        </extensions>
        """
    )

    collected = collect_extension_types(
        root,
        frozenset({"VK_A"}),
        gen.VulkanVersion(1, 1),
    )
    assert collected == frozenset({"VkA", "VkA11"})


def test_t_17_collect_extension_types_empty_input_returns_empty(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    collect_extension_types = _require_callable("collect_extension_types")
    root = make_registry_root(
        '<extensions><extension name="VK_A" supported="vulkan"/></extensions>'
    )

    assert (
        collect_extension_types(root, frozenset(), gen.VulkanVersion(1, 3))
        == frozenset()
    )


def test_t_18_collect_extension_commands_mirrors_extension_eligibility(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    collect_extension_commands = _require_callable("collect_extension_commands")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan">
                <require><command name="vkA"/></require>
                <require depends="VK_VERSION_1_1"><command name="vkA11"/></require>
            </extension>
            <extension name="VK_B" supported="vulkan">
                <require><command name="vkB"/></require>
            </extension>
        </extensions>
        """
    )

    collected = collect_extension_commands(
        root,
        frozenset({"VK_A"}),
        gen.VulkanVersion(1, 1),
    )
    assert collected == frozenset({"vkA", "vkA11"})


def test_t_19_collect_extension_excludes_non_vulkan_and_unsatisfied_requires(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    collect_extension_types = _require_callable("collect_extension_types")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan">
                <require depends="VK_VERSION_1_2"><type name="VkA12"/></require>
            </extension>
            <extension name="VK_SC_ONLY" supported="vulkansc">
                <require><type name="VkScOnly"/></require>
            </extension>
        </extensions>
        """
    )

    collected = collect_extension_types(
        root,
        frozenset({"VK_A", "VK_SC_ONLY"}),
        gen.VulkanVersion(1, 1),
    )
    assert collected == frozenset()


def test_t_20_close_type_deps_adds_non_pointer_value_dependencies(
    make_struct_member: Callable[..., gen.StructMember],
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    close_type_deps = _require_callable("close_type_deps")

    structs = [
        make_struct_def(
            "VkA",
            [make_struct_member(name="b", type_name="VkB", is_pointer=False)],
        ),
        make_struct_def("VkB", []),
    ]
    closed, stats = close_type_deps(
        frozenset({"VkA"}),
        structs,
        {"VkA": "struct", "VkB": "struct"},
    )

    assert "VkB" in closed
    assert "VkB" in getattr(stats, "added_types")


def test_t_21_close_type_deps_ignores_pointer_members(
    make_struct_member: Callable[..., gen.StructMember],
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    close_type_deps = _require_callable("close_type_deps")
    structs = [
        make_struct_def(
            "VkA",
            [make_struct_member(name="b", type_name="VkB", is_pointer=True)],
        ),
        make_struct_def("VkB", []),
    ]
    closed, _stats = close_type_deps(
        frozenset({"VkA"}),
        structs,
        {"VkA": "struct", "VkB": "struct"},
    )

    assert "VkB" not in closed


def test_t_22_close_type_deps_adds_non_pointer_array_members(
    make_struct_member: Callable[..., gen.StructMember],
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    close_type_deps = _require_callable("close_type_deps")
    structs = [
        make_struct_def(
            "VkA",
            [
                make_struct_member(
                    name="b",
                    type_name="VkB",
                    is_pointer=False,
                    array_size="4",
                )
            ],
        ),
        make_struct_def("VkB", []),
    ]
    closed, _stats = close_type_deps(
        frozenset({"VkA"}),
        structs,
        {"VkA": "struct", "VkB": "struct"},
    )

    assert "VkB" in closed


def test_t_23_close_type_deps_reaches_fixed_point_multi_hop(
    make_struct_member: Callable[..., gen.StructMember],
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    close_type_deps = _require_callable("close_type_deps")
    structs = [
        make_struct_def(
            "VkA",
            [make_struct_member(name="b", type_name="VkB", is_pointer=False)],
        ),
        make_struct_def(
            "VkB",
            [make_struct_member(name="c", type_name="VkC", is_pointer=False)],
        ),
        make_struct_def("VkC", []),
    ]
    closed, stats = close_type_deps(
        frozenset({"VkA"}),
        structs,
        {"VkA": "struct", "VkB": "struct", "VkC": "struct"},
    )

    assert closed == frozenset({"VkA", "VkB", "VkC"})
    assert getattr(stats, "iteration_count") >= 2


def test_t_24_close_type_deps_only_adds_registry_known_types(
    make_struct_member: Callable[..., gen.StructMember],
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    close_type_deps = _require_callable("close_type_deps")
    structs = [
        make_struct_def(
            "VkA",
            [
                make_struct_member(name="x", type_name="uint32_t", is_pointer=False),
                make_struct_member(name="y", type_name="PlatformFoo", is_pointer=False),
            ],
        )
    ]
    closed, _stats = close_type_deps(frozenset({"VkA"}), structs, {"VkA": "struct"})

    assert "uint32_t" not in closed
    assert "PlatformFoo" not in closed


def test_t_25_close_type_deps_has_safety_limit_backstop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_type_deps = _require_callable("close_type_deps")
    assert hasattr(gen, "_TYPE_CLOSURE_SAFETY_LIMIT"), (
        "Missing S2 safety-limit constant: gen._TYPE_CLOSURE_SAFETY_LIMIT"
    )

    monkeypatch.setattr(gen, "_TYPE_CLOSURE_SAFETY_LIMIT", 0)
    with pytest.raises(RuntimeError):
        close_type_deps(frozenset({"VkA"}), [], {"VkA": "struct"})


def test_t_26_build_target_set_assembles_unions_and_closure(
    make_registry_root: Callable[[str], ET.Element],
    make_struct_member: Callable[..., gen.StructMember],
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    build_target_set = _require_callable("build_target_set")
    _require_type("TargetSet")
    _require_type("TypeClosureStats")
    root = make_registry_root(
        """
        <feature api="vulkan" number="1.1">
            <require>
                <type name="VkCoreType"/>
                <command name="vkCoreCmd"/>
            </require>
        </feature>
        <extensions>
            <extension name="VK_A" supported="vulkan">
                <require>
                    <type name="VkExtType"/>
                    <command name="vkExtCmd"/>
                </require>
            </extension>
        </extensions>
        """
    )
    structs = [
        make_struct_def(
            "VkExtType",
            [make_struct_member(name="dep", type_name="VkTransit", is_pointer=False)],
        ),
        make_struct_def("VkTransit", []),
    ]

    target, _stats = build_target_set(
        root,
        gen.VulkanVersion(1, 1),
        frozenset({"VK_A"}),
        structs,
        {"VkCoreType": "struct", "VkExtType": "struct", "VkTransit": "struct"},
    )

    assert getattr(target, "types") == frozenset(
        {"VkCoreType", "VkExtType", "VkTransit"}
    )
    assert getattr(target, "commands") == frozenset({"vkCoreCmd", "vkExtCmd"})


@pytest.mark.skipif(
    not gen.DEFAULT_VK_XML.exists(),
    reason="integration: requires real vk.xml",
)
def test_t_27_build_target_set_compile_safety_invariant_real_vkxml() -> None:
    build_target_set = _require_callable("build_target_set")

    root = ET.parse(gen.DEFAULT_VK_XML).getroot()
    all_type_names = gen.load_all_type_names(root)
    structs, _aliases = gen.extract_structs(root)
    target, _stats = build_target_set(
        root,
        gen.VulkanVersion(1, 3),
        frozenset(),
        structs,
        all_type_names,
    )

    target_types = getattr(target, "types")
    struct_map = {s.name: s for s in structs}
    for type_name in target_types:
        struct_def = struct_map.get(type_name)
        if struct_def is None:
            continue
        for member in struct_def.members:
            if member.is_pointer:
                continue
            if member.type_name in all_type_names:
                assert member.type_name in target_types


@pytest.mark.skipif(
    not gen.DEFAULT_VK_XML.exists(),
    reason="integration: requires real vk.xml",
)
def test_t_28_build_target_set_is_deterministic_real_vkxml() -> None:
    build_target_set = _require_callable("build_target_set")

    root = ET.parse(gen.DEFAULT_VK_XML).getroot()
    all_type_names = gen.load_all_type_names(root)
    structs, _aliases = gen.extract_structs(root)

    first_target, first_stats = build_target_set(
        root,
        gen.VulkanVersion(1, 3),
        frozenset(),
        structs,
        all_type_names,
    )
    second_target, second_stats = build_target_set(
        root,
        gen.VulkanVersion(1, 3),
        frozenset(),
        structs,
        all_type_names,
    )

    assert first_target == second_target
    assert first_stats == second_stats


@pytest.mark.skipif(
    not gen.DEFAULT_VK_XML.exists(),
    reason="integration: requires real vk.xml",
)
def test_t_29_promoted_extension_is_no_op_relative_to_core_only() -> None:
    build_target_set = _require_callable("build_target_set")
    resolve_extension_deps = _require_callable("resolve_extension_deps")

    root = ET.parse(gen.DEFAULT_VK_XML).getroot()
    all_type_names = gen.load_all_type_names(root)
    structs, _aliases = gen.extract_structs(root)

    core_only, _stats = build_target_set(
        root,
        gen.VulkanVersion(1, 3),
        frozenset(),
        structs,
        all_type_names,
    )
    promoted_resolved = resolve_extension_deps(
        root,
        frozenset({"VK_KHR_dynamic_rendering"}),
    )
    with_promoted, _stats = build_target_set(
        root,
        gen.VulkanVersion(1, 3),
        promoted_resolved,
        structs,
        all_type_names,
    )

    assert core_only == with_promoted


def test_t_30_build_target_set_propagates_closure_runtime_errors(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_target_set = _require_callable("build_target_set")
    _require_callable("close_type_deps")
    root = make_registry_root('<feature api="vulkan" number="1.0"><require/></feature>')

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("closure failed")

    monkeypatch.setattr(gen, "close_type_deps", _raise)
    with pytest.raises(RuntimeError):
        build_target_set(root, gen.VulkanVersion(1, 0), frozenset(), [], {})


@dataclass(frozen=True)
class _NamedItem:
    name: str


def test_t_31_filter_by_target_preserves_order_and_filters_membership() -> None:
    filter_by_target = _require_callable("filter_by_target")

    items = [_NamedItem("A"), _NamedItem("B"), _NamedItem("C")]
    filtered = filter_by_target(
        items, frozenset({"A", "C"}), key=lambda item: item.name
    )
    assert filtered == [_NamedItem("A"), _NamedItem("C")]


def test_t_32_filter_by_target_returns_new_list_and_handles_no_matches() -> None:
    filter_by_target = _require_callable("filter_by_target")

    items = [_NamedItem("A"), _NamedItem("B")]
    filtered = filter_by_target(items, frozenset({"Z"}), key=lambda item: item.name)
    assert filtered == []
    assert filtered is not items
    assert items == [_NamedItem("A"), _NamedItem("B")]


def test_t_33_filter_aliases_by_target_includes_when_alias_in_target() -> None:
    filter_aliases_by_target = _require_callable("filter_aliases_by_target")

    aliases = [("VkFooKHR", "VkFoo")]
    filtered = filter_aliases_by_target(aliases, frozenset({"VkFooKHR"}))
    assert filtered == aliases


def test_t_34_filter_aliases_by_target_includes_when_target_in_target() -> None:
    filter_aliases_by_target = _require_callable("filter_aliases_by_target")

    aliases = [("VkFooKHR", "VkFoo")]
    filtered = filter_aliases_by_target(aliases, frozenset({"VkFoo"}))
    assert filtered == aliases


def test_t_35_filter_aliases_by_target_excludes_when_neither_side_in_target() -> None:
    filter_aliases_by_target = _require_callable("filter_aliases_by_target")

    aliases = [("VkFooKHR", "VkFoo")]
    filtered = filter_aliases_by_target(aliases, frozenset())
    assert filtered == []
