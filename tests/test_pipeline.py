from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import gen


def _require_callable(name: str) -> Callable[..., object]:
    symbol = getattr(gen, name, None)
    assert callable(symbol), f"Missing S5 API symbol: gen.{name}"
    return symbol


def _require_type(name: str) -> type:
    symbol = getattr(gen, name, None)
    assert isinstance(symbol, type), f"Missing S5 type symbol: gen.{name}"
    return symbol


def _make_registry_meta(
    *,
    all_type_names: dict[str, str] | None = None,
    api_constants: dict[str, int] | None = None,
    known_types: set[str] | None = None,
    vk_xml_version: str = "1.4.343",
) -> object:
    registry_meta_type = _require_type("RegistryMeta")
    return registry_meta_type(
        all_type_names={} if all_type_names is None else all_type_names,
        api_constants={} if api_constants is None else api_constants,
        known_types=set() if known_types is None else known_types,
        vk_xml_version=vk_xml_version,
    )


def _make_extracted_registry(
    *,
    basetypes: list[tuple[str, str, str]] | None = None,
    enum_values: dict[str, list[tuple[str, int]]] | None = None,
    block_types: dict[str, str] | None = None,
    enum_type_names: set[str] | None = None,
    handles: list[tuple[str, str, bool]] | None = None,
    handle_aliases: list[tuple[str, str]] | None = None,
    unions: list[gen.StructDef] | None = None,
    union_aliases: list[tuple[str, str]] | None = None,
    structs: list[gen.StructDef] | None = None,
    struct_aliases: list[tuple[str, str]] | None = None,
    commands: list[gen.CommandDef] | None = None,
    cmd_aliases: list[tuple[str, str]] | None = None,
) -> object:
    extracted_registry_type = _require_type("ExtractedRegistry")
    return extracted_registry_type(
        basetypes=[] if basetypes is None else basetypes,
        enum_values={} if enum_values is None else enum_values,
        block_types={} if block_types is None else block_types,
        enum_type_names=set() if enum_type_names is None else enum_type_names,
        handles=[] if handles is None else handles,
        handle_aliases=[] if handle_aliases is None else handle_aliases,
        unions=[] if unions is None else unions,
        union_aliases=[] if union_aliases is None else union_aliases,
        structs=[] if structs is None else structs,
        struct_aliases=[] if struct_aliases is None else struct_aliases,
        commands=[] if commands is None else commands,
        cmd_aliases=[] if cmd_aliases is None else cmd_aliases,
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


def _make_target_set(types: frozenset[str], commands: frozenset[str]) -> gen.TargetSet:
    target_set_type = _require_type("TargetSet")
    return target_set_type(types=types, commands=commands)


def _make_generate_config(tmp_path: Path) -> gen.GenerateConfig:
    vk_xml = tmp_path / "vk.xml"
    vk_xml.write_text("<registry/>\n", encoding="utf-8")
    headers = tmp_path / "headers"
    headers.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "out"
    return gen.GenerateConfig(
        version=gen.VulkanVersion(1, 3),
        extensions=frozenset(),
        all_extensions=False,
        vk_xml=vk_xml,
        vulkan_headers=headers,
        output_dir=output_dir,
    )


def _make_write_config() -> gen.WriteConfig:
    return gen.WriteConfig(
        vk_xml_version="1.4.343",
        target_version=gen.VulkanVersion(1, 3),
        extensions=frozenset(),
        all_extensions=False,
    )


def _make_command(name: str) -> gen.CommandDef:
    return gen.CommandDef(
        name=name, return_type="VkResult", return_is_void=False, params=[]
    )


def test_t_00_api_surface_symbols_exist() -> None:
    for type_name in ("RegistryMeta", "ExtractedRegistry", "FilteredRegistry"):
        _require_type(type_name)
    for callable_name in (
        "extract_registry",
        "resolve_generate_extensions",
        "filter_and_prepare",
        "build_module_specs",
        "build_write_config",
        "run_generate",
    ):
        _require_callable(callable_name)


def test_t_00_api_surface_migration_guard_module_order_excludes_vk_monolith() -> None:
    assert "vk" not in gen.MODULE_ORDER
    assert all(stem.startswith("vk_") for stem in gen.MODULE_ORDER)


def test_t_01_extract_registry_calls_extractors_and_bundles_unchanged(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract_registry = _require_callable("extract_registry")
    root = make_registry_root("<types/>")
    meta = _make_registry_meta()

    basetypes = [("VkBaseA", "UInt32", "")]
    enum_values = {"VkEnum": [("VK_ENUM_A", 1)]}
    block_types = {"VkEnum": "enum"}
    enum_type_names = {"VkEnum"}
    handles = [("VkHandle", "", True)]
    handle_aliases = [("VkHandleAlias", "VkHandle")]
    unions = [gen.StructDef("VkUnion", [])]
    union_aliases = [("VkUnionAlias", "VkUnion")]
    structs = [gen.StructDef("VkStruct", [])]
    struct_aliases = [("VkStructAlias", "VkStruct")]
    commands = [_make_command("vkCmdX")]
    cmd_aliases = [("vkCmdAlias", "vkCmdX")]

    monkeypatch.setattr(gen, "extract_basetypes", lambda _root: basetypes)
    monkeypatch.setattr(gen, "collect_enum_values", lambda _root: enum_values)
    monkeypatch.setattr(gen, "get_enum_block_types", lambda _root: block_types)
    monkeypatch.setattr(gen, "get_enum_type_names", lambda _root: enum_type_names)
    monkeypatch.setattr(gen, "extract_handles", lambda _root: (handles, handle_aliases))
    monkeypatch.setattr(gen, "extract_unions", lambda _root: (unions, union_aliases))
    monkeypatch.setattr(gen, "extract_structs", lambda _root: (structs, struct_aliases))
    monkeypatch.setattr(gen, "extract_commands", lambda _root: (commands, cmd_aliases))

    extracted = extract_registry(root, meta)

    assert extracted.basetypes == basetypes
    assert extracted.enum_values == enum_values
    assert extracted.block_types == block_types
    assert extracted.enum_type_names == enum_type_names
    assert extracted.handles == handles
    assert extracted.handle_aliases == handle_aliases
    assert extracted.unions == unions
    assert extracted.union_aliases == union_aliases
    assert extracted.structs == structs
    assert extracted.struct_aliases == struct_aliases
    assert extracted.commands == commands
    assert extracted.cmd_aliases == cmd_aliases


def test_t_02_extract_registry_does_not_filter_by_target(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract_registry = _require_callable("extract_registry")
    root = make_registry_root("<types/>")
    meta = _make_registry_meta()
    structs = [gen.StructDef("VkCore", []), gen.StructDef("VkExtOnly", [])]

    monkeypatch.setattr(gen, "extract_basetypes", lambda _root: [])
    monkeypatch.setattr(gen, "collect_enum_values", lambda _root: {})
    monkeypatch.setattr(gen, "get_enum_block_types", lambda _root: {})
    monkeypatch.setattr(gen, "get_enum_type_names", lambda _root: set())
    monkeypatch.setattr(gen, "extract_handles", lambda _root: ([], []))
    monkeypatch.setattr(gen, "extract_unions", lambda _root: ([], []))
    monkeypatch.setattr(gen, "extract_structs", lambda _root: (structs, []))
    monkeypatch.setattr(gen, "extract_commands", lambda _root: ([], []))

    extracted = extract_registry(root, meta)

    assert [s.name for s in extracted.structs] == ["VkCore", "VkExtOnly"]


def test_t_03_extract_registry_propagates_extractor_exception(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extract_registry = _require_callable("extract_registry")
    root = make_registry_root("<types/>")
    meta = _make_registry_meta()

    monkeypatch.setattr(gen, "extract_basetypes", lambda _root: [])

    def _boom(_root: ET.Element) -> dict[str, list[tuple[str, int]]]:
        raise RuntimeError("extract boom")

    monkeypatch.setattr(gen, "collect_enum_values", _boom)

    with pytest.raises(RuntimeError, match="extract boom"):
        extract_registry(root, meta)


def test_t_04_resolve_generate_extensions_all_extensions_uses_full_registry(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_generate_extensions = _require_callable("resolve_generate_extensions")
    root = make_registry_root("<extensions/>")
    config = gen.GenerateConfig(
        version=gen.VulkanVersion(1, 3),
        extensions=frozenset({"VK_X"}),
        all_extensions=True,
        vk_xml=Path("vk.xml"),
        vulkan_headers=Path("headers"),
        output_dir=Path("out"),
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        gen,
        "all_vulkan_extension_names",
        lambda _root: frozenset({"VK_A", "VK_B"}),
    )

    def _resolve(_root: ET.Element, initial: frozenset[str]) -> frozenset[str]:
        seen["initial"] = initial
        return frozenset({"VK_A", "VK_B", "VK_C"})

    monkeypatch.setattr(gen, "resolve_extension_deps", _resolve)

    resolved = resolve_generate_extensions(root, config)

    assert seen["initial"] == frozenset({"VK_A", "VK_B"})
    assert resolved == frozenset({"VK_A", "VK_B", "VK_C"})


def test_t_05_resolve_generate_extensions_explicit_mode_uses_config_extensions(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_generate_extensions = _require_callable("resolve_generate_extensions")
    root = make_registry_root("<extensions/>")
    config = gen.GenerateConfig(
        version=gen.VulkanVersion(1, 3),
        extensions=frozenset({"VK_KHR_swapchain"}),
        all_extensions=False,
        vk_xml=Path("vk.xml"),
        vulkan_headers=Path("headers"),
        output_dir=Path("out"),
    )
    seen: dict[str, object] = {"all_called": False}

    def _all_extensions(_root: ET.Element) -> frozenset[str]:
        seen["all_called"] = True
        return frozenset({"VK_UNUSED"})

    def _resolve(_root: ET.Element, initial: frozenset[str]) -> frozenset[str]:
        seen["initial"] = initial
        return frozenset({"VK_KHR_swapchain", "VK_KHR_surface"})

    monkeypatch.setattr(gen, "all_vulkan_extension_names", _all_extensions)
    monkeypatch.setattr(gen, "resolve_extension_deps", _resolve)

    resolved = resolve_generate_extensions(root, config)

    assert seen["all_called"] is False
    assert seen["initial"] == frozenset({"VK_KHR_swapchain"})
    assert resolved == frozenset({"VK_KHR_swapchain", "VK_KHR_surface"})


def test_t_06_resolve_generate_extensions_empty_explicit_yields_empty(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_generate_extensions = _require_callable("resolve_generate_extensions")
    root = make_registry_root("<extensions/>")
    config = gen.GenerateConfig(
        version=gen.VulkanVersion(1, 3),
        extensions=frozenset(),
        all_extensions=False,
        vk_xml=Path("vk.xml"),
        vulkan_headers=Path("headers"),
        output_dir=Path("out"),
    )

    assert resolve_generate_extensions(root, config) == frozenset()


def test_t_07_resolve_generate_extensions_does_not_add_validation_layer(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_generate_extensions = _require_callable("resolve_generate_extensions")
    root = make_registry_root("<extensions/>")
    config = gen.GenerateConfig(
        version=gen.VulkanVersion(1, 3),
        extensions=frozenset({"VK_UNKNOWN"}),
        all_extensions=False,
        vk_xml=Path("vk.xml"),
        vulkan_headers=Path("headers"),
        output_dir=Path("out"),
    )
    expected = frozenset({"VK_UNKNOWN", "VK_PROMOTED"})
    monkeypatch.setattr(gen, "resolve_extension_deps", lambda _root, _exts: expected)

    assert resolve_generate_extensions(root, config) == expected


def test_t_08_filter_and_prepare_basetypes_passthrough_and_category_targeting(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    filter_and_prepare = _require_callable("filter_and_prepare")
    extracted = _make_extracted_registry(
        basetypes=[("VkBool32", "UInt32", ""), ("VkExtType", "UInt64", "")],
        enum_values={"VkCoreEnum": [], "VkExtEnum": []},
        block_types={"VkCoreEnum": "enum", "VkExtEnum": "enum"},
        enum_type_names={"VkCoreEnum", "VkExtEnum"},
        handles=[("VkCoreHandle", "", True), ("VkExtHandle", "", True)],
        unions=[make_struct_def("VkCoreUnion", []), make_struct_def("VkExtUnion", [])],
        structs=[
            make_struct_def("VkCoreStruct", []),
            make_struct_def("VkExtStruct", []),
        ],
        commands=[_make_command("vkCoreCmd"), _make_command("vkExtCmd")],
    )
    target = _make_target_set(
        types=frozenset({"VkCoreEnum", "VkCoreHandle", "VkCoreUnion", "VkCoreStruct"}),
        commands=frozenset({"vkCoreCmd"}),
    )
    meta = _make_registry_meta(all_type_names={"VkCoreStruct": "struct"})

    filtered = filter_and_prepare(extracted, target, meta)

    assert filtered.basetypes == extracted.basetypes
    assert set(filtered.enum_values.keys()) == {"VkCoreEnum"}
    assert [h[0] for h in filtered.handles] == ["VkCoreHandle"]
    assert [u.name for u in filtered.unions] == ["VkCoreUnion"]
    assert [s.name for s in filtered.structs] == ["VkCoreStruct"]
    assert [c.name for c in filtered.commands] == ["vkCoreCmd"]


def test_t_09_filter_and_prepare_alias_filtering_uses_either_side_membership(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    filter_and_prepare = _require_callable("filter_and_prepare")
    extracted = _make_extracted_registry(
        unions=[make_struct_def("VkUnion", [])],
        structs=[make_struct_def("VkStruct", [])],
        handle_aliases=[
            ("VkAliasIn", "VkNope"),
            ("VkNope", "VkTargetIn"),
            ("VkNopeA", "VkNopeB"),
        ],
        union_aliases=[
            ("VkUAliasIn", "VkNope"),
            ("VkNope", "VkUTargetIn"),
            ("VkNopeA", "VkNopeB"),
        ],
        struct_aliases=[
            ("VkSAliasIn", "VkNope"),
            ("VkNope", "VkSTargetIn"),
            ("VkNopeA", "VkNopeB"),
        ],
        cmd_aliases=[
            ("vkAliasIn", "vkNope"),
            ("vkNope", "vkTargetIn"),
            ("vkNopeA", "vkNopeB"),
        ],
    )
    target = _make_target_set(
        types=frozenset(
            {
                "VkAliasIn",
                "VkTargetIn",
                "VkUAliasIn",
                "VkUTargetIn",
                "VkSAliasIn",
                "VkSTargetIn",
            }
        ),
        commands=frozenset({"vkAliasIn", "vkTargetIn"}),
    )
    meta = _make_registry_meta()

    filtered = filter_and_prepare(extracted, target, meta)

    assert filtered.handle_aliases == [
        ("VkAliasIn", "VkNope"),
        ("VkNope", "VkTargetIn"),
    ]
    assert filtered.union_aliases == [
        ("VkUAliasIn", "VkNope"),
        ("VkNope", "VkUTargetIn"),
    ]
    assert filtered.struct_aliases == [
        ("VkSAliasIn", "VkNope"),
        ("VkNope", "VkSTargetIn"),
    ]
    assert filtered.cmd_aliases == [("vkAliasIn", "vkNope"), ("vkNope", "vkTargetIn")]


def test_t_10_filter_and_prepare_measures_union_sizes_on_filtered_unions_only(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filter_and_prepare = _require_callable("filter_and_prepare")
    extracted = _make_extracted_registry(
        unions=[make_struct_def("VkCoreUnion", []), make_struct_def("VkExtUnion", [])],
        structs=[],
    )
    target = _make_target_set(types=frozenset({"VkCoreUnion"}), commands=frozenset())
    meta = _make_registry_meta()
    seen: dict[str, list[str]] = {}

    def _measure(unions: list[gen.StructDef]) -> dict[str, int]:
        seen["union_names"] = [u.name for u in unions]
        return {u.name: 16 for u in unions}

    monkeypatch.setattr(gen, "measure_union_sizes", _measure)

    filtered = filter_and_prepare(extracted, target, meta)

    assert seen["union_names"] == ["VkCoreUnion"]
    assert filtered.union_sizes == {"VkCoreUnion": 16}


def test_t_11_filter_and_prepare_topo_sort_uses_filtered_structs_and_full_type_map(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filter_and_prepare = _require_callable("filter_and_prepare")
    core_struct = make_struct_def("VkCoreStruct", [])
    ext_struct = make_struct_def("VkExtStruct", [])
    extracted = _make_extracted_registry(structs=[core_struct, ext_struct], unions=[])
    target = _make_target_set(types=frozenset({"VkCoreStruct"}), commands=frozenset())
    all_type_names = {"VkCoreStruct": "struct", "VkExtStruct": "struct"}
    meta = _make_registry_meta(all_type_names=all_type_names)
    seen: dict[str, object] = {}

    def _topo(
        structs: list[gen.StructDef], type_map: dict[str, str]
    ) -> list[gen.StructDef]:
        seen["struct_names"] = [s.name for s in structs]
        seen["type_map"] = type_map
        return list(structs)

    monkeypatch.setattr(gen, "topo_sort_structs", _topo)

    filter_and_prepare(extracted, target, meta)

    assert seen["struct_names"] == ["VkCoreStruct"]
    assert seen["type_map"] == all_type_names


def test_t_12_filter_and_prepare_returns_postprocessed_outputs(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filter_and_prepare = _require_callable("filter_and_prepare")
    struct_a = make_struct_def("VkA", [])
    struct_b = make_struct_def("VkB", [])
    union_a = make_struct_def("VkU", [])
    extracted = _make_extracted_registry(structs=[struct_a, struct_b], unions=[union_a])
    target = _make_target_set(
        types=frozenset({"VkA", "VkB", "VkU"}), commands=frozenset()
    )
    meta = _make_registry_meta(all_type_names={"VkA": "struct", "VkB": "struct"})

    monkeypatch.setattr(gen, "measure_union_sizes", lambda _unions: {"VkU": 64})
    monkeypatch.setattr(
        gen, "topo_sort_structs", lambda _structs, _type_map: [struct_b, struct_a]
    )

    filtered = filter_and_prepare(extracted, target, meta)

    assert filtered.union_sizes == {"VkU": 64}
    assert [s.name for s in filtered.structs] == ["VkB", "VkA"]


def test_t_13_filter_and_prepare_propagates_topo_cycle_error(
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filter_and_prepare = _require_callable("filter_and_prepare")
    extracted = _make_extracted_registry(
        structs=[make_struct_def("VkA", [])], unions=[]
    )
    target = _make_target_set(types=frozenset({"VkA"}), commands=frozenset())
    meta = _make_registry_meta(all_type_names={"VkA": "struct"})

    def _boom(
        _structs: list[gen.StructDef], _type_map: dict[str, str]
    ) -> list[gen.StructDef]:
        raise RuntimeError("cycle")

    monkeypatch.setattr(gen, "topo_sort_structs", _boom)

    with pytest.raises(RuntimeError, match="cycle"):
        filter_and_prepare(extracted, target, meta)


def test_t_14_build_module_specs_returns_seven_specs_in_module_order(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    build_module_specs = _require_callable("build_module_specs")
    root = make_registry_root("<types/>")
    filtered = _make_filtered_registry()
    specs = build_module_specs(
        root, filtered, _make_write_config(), _make_registry_meta()
    )

    assert len(specs) == len(gen.MODULE_ORDER)
    assert tuple(spec.filename for spec in specs) == tuple(
        f"{stem}.mojo" for stem in gen.MODULE_ORDER
    )
    assert all(spec.filename != "vk.mojo" for spec in specs)


def test_t_15_build_module_specs_base_types_includes_root_based_generators(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module_specs = _require_callable("build_module_specs")
    root = make_registry_root("<types/>")
    filtered = _make_filtered_registry(basetypes=[("VkBool32", "UInt32", "")])

    monkeypatch.setattr(gen, "generate_base_types", lambda _items: ["BASE_TAG"])
    monkeypatch.setattr(gen, "generate_bitmask_aliases", lambda _root: ["BITMASK_TAG"])
    monkeypatch.setattr(
        gen, "generate_funcpointer_aliases", lambda _root: ["FUNCPTR_TAG"]
    )
    monkeypatch.setattr(gen, "generate_stdvideo_types", lambda _root: ["STDVIDEO_TAG"])

    specs = build_module_specs(
        root, filtered, _make_write_config(), _make_registry_meta()
    )
    by_name = {spec.filename: spec for spec in specs}
    base_lines = by_name["vk_base_types.mojo"].content_lines

    assert "BASE_TAG" in base_lines
    assert "BITMASK_TAG" in base_lines
    assert "FUNCPTR_TAG" in base_lines
    assert "STDVIDEO_TAG" in base_lines
    for filename, spec in by_name.items():
        if filename == "vk_base_types.mojo":
            continue
        assert "BITMASK_TAG" not in spec.content_lines
        assert "FUNCPTR_TAG" not in spec.content_lines
        assert "STDVIDEO_TAG" not in spec.content_lines


def test_t_16_build_module_specs_preserves_loader_commands_split(
    make_registry_root: Callable[[str], ET.Element],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module_specs = _require_callable("build_module_specs")
    root = make_registry_root("<types/>")
    cmd = _make_command("vkCmdX")
    filtered = _make_filtered_registry(
        commands=[cmd], cmd_aliases=[("vkCmdAlias", "vkCmdX")]
    )

    monkeypatch.setattr(gen, "generate_loader", lambda: ["LOADER_INFRA_TAG"])
    monkeypatch.setattr(
        gen,
        "generate_init_function",
        lambda _fn_name, _commands: ["LOADER_INIT_TAG"],
    )
    monkeypatch.setattr(
        gen, "generate_commands", lambda _cmds, _aliases, _types: ["COMMAND_ALIAS_TAG"]
    )
    monkeypatch.setattr(
        gen, "generate_wrapper_fn", lambda _cmd, _types: ["WRAPPER_FN_TAG"]
    )
    monkeypatch.setattr(
        gen,
        "generate_wrapper_alias",
        lambda _name, _target, _target_cmd, _types: ["WRAPPER_ALIAS_TAG"],
    )

    def _legacy_loader_section(*_args: object, **_kwargs: object) -> list[str]:
        raise AssertionError(
            "generate_loader_section must not be called by S5 assembly"
        )

    monkeypatch.setattr(gen, "generate_loader_section", _legacy_loader_section)

    specs = build_module_specs(
        root, filtered, _make_write_config(), _make_registry_meta()
    )
    by_name = {spec.filename: spec for spec in specs}

    loader_lines = by_name["vk_loader.mojo"].content_lines
    commands_lines = by_name["vk_commands.mojo"].content_lines

    assert "LOADER_INFRA_TAG" in loader_lines
    assert "LOADER_INIT_TAG" in loader_lines
    assert "WRAPPER_FN_TAG" not in loader_lines
    assert "WRAPPER_ALIAS_TAG" not in loader_lines

    assert "COMMAND_ALIAS_TAG" in commands_lines
    assert "WRAPPER_FN_TAG" in commands_lines
    assert "WRAPPER_ALIAS_TAG" in commands_lines
    assert "LOADER_INFRA_TAG" not in commands_lines


def test_t_16a_build_module_specs_loader_exports_init_vulkan_entrypoint(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    build_module_specs = _require_callable("build_module_specs")
    root = make_registry_root("<types/>")
    specs = build_module_specs(
        root, _make_filtered_registry(), _make_write_config(), _make_registry_meta()
    )
    by_name = {spec.filename: spec for spec in specs}
    loader_lines = by_name["vk_loader.mojo"].content_lines

    assert any(line.startswith("fn init_vulkan(") for line in loader_lines)
    assert "    init_vulkan_global(global_load)" in loader_lines
    assert "    init_vulkan_instance(instance_load)" in loader_lines
    assert "    init_vulkan_device(device_load)" in loader_lines


def test_t_16b_build_module_specs_emits_wrapper_aliases_from_target_command(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    build_module_specs = _require_callable("build_module_specs")
    root = make_registry_root("<types/>")
    target_cmd = gen.CommandDef(
        name="vkQueueSubmit",
        return_type="VkResult",
        return_is_void=False,
        params=[gen.CommandParam("count", "uint32_t", 0, False)],
    )
    filtered = _make_filtered_registry(
        commands=[target_cmd],
        cmd_aliases=[("vkQueueSubmitAlias", "vkQueueSubmit")],
    )
    meta = _make_registry_meta(known_types={"VkResult"})

    specs = build_module_specs(root, filtered, _make_write_config(), meta)
    by_name = {spec.filename: spec for spec in specs}
    cmd_lines = by_name["vk_commands.mojo"].content_lines

    assert "fn queue_submit_alias(count: UInt32) raises -> VkResult:" in cmd_lines
    assert "    return queue_submit(count)" in cmd_lines


def test_t_17_build_module_specs_sibling_imports_are_selective_and_bounded(
    make_registry_root: Callable[[str], ET.Element],
    make_struct_member: Callable[..., gen.StructMember],
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    build_module_specs = _require_callable("build_module_specs")
    root = make_registry_root("<types/>")
    filtered = _make_filtered_registry(
        basetypes=[("VkBool32", "UInt32", "")],
        enum_values={"VkEnum": [("VK_ENUM_A", 1)]},
        block_types={"VkEnum": "enum"},
        enum_type_names={"VkEnum"},
        handles=[("VkHandle", "", True)],
        structs=[
            make_struct_def(
                "VkStruct",
                [
                    make_struct_member(name="flag", type_name="VkBool32"),
                    make_struct_member(name="h", type_name="VkHandle"),
                ],
            )
        ],
        commands=[_make_command("vkCmdX")],
    )
    meta = _make_registry_meta(
        known_types={"VkStruct", "VkHandle", "VkBool32", "VkEnum"}
    )

    specs = build_module_specs(root, filtered, _make_write_config(), meta)
    sibling_imports = [imp for spec in specs for imp in spec.sibling_imports]

    assert sibling_imports
    for imp in sibling_imports:
        assert imp.names
        assert imp.module_stem in gen.MODULE_ORDER


def test_t_18_build_module_specs_struct_errors_are_inline_not_fatal(
    make_registry_root: Callable[[str], ET.Element],
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_module_specs = _require_callable("build_module_specs")
    root = make_registry_root("<types/>")
    filtered = _make_filtered_registry(
        structs=[
            make_struct_def("VkBadStruct", []),
            make_struct_def("VkGoodStruct", []),
        ]
    )

    def _generate_struct(struct: gen.StructDef, *_args: object) -> list[str]:
        if struct.name == "VkBadStruct":
            raise RuntimeError("bad struct")
        return [f"struct {struct.name}:", "    pass"]

    monkeypatch.setattr(gen, "generate_struct_mojo", _generate_struct)

    specs = build_module_specs(
        root, filtered, _make_write_config(), _make_registry_meta()
    )
    structs_spec = next(spec for spec in specs if spec.filename == "vk_structs.mojo")

    assert any(
        "# ERROR generating VkBadStruct: bad struct" in line
        for line in structs_spec.content_lines
    )
    assert any("struct VkGoodStruct:" in line for line in structs_spec.content_lines)


def test_t_19_build_write_config_maps_generate_config_fields_exactly(
    tmp_path: Path,
) -> None:
    build_write_config = _require_callable("build_write_config")
    config = gen.GenerateConfig(
        version=gen.VulkanVersion(1, 2),
        extensions=frozenset({"VK_KHR_swapchain"}),
        all_extensions=True,
        vk_xml=tmp_path / "vk.xml",
        vulkan_headers=tmp_path / "headers",
        output_dir=tmp_path / "out",
    )

    write_config = build_write_config(config, "1.4.343")

    assert write_config.target_version == config.version
    assert write_config.extensions == config.extensions
    assert write_config.all_extensions is True
    assert write_config.vk_xml_version == "1.4.343"


def test_t_20_build_write_config_rejects_empty_vk_xml_version(tmp_path: Path) -> None:
    build_write_config = _require_callable("build_write_config")
    config = _make_generate_config(tmp_path)

    with pytest.raises(ValueError):
        build_write_config(config, "")


def test_t_21_run_generate_executes_stages_in_strict_linear_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_generate = _require_callable("run_generate")
    config = _make_generate_config(tmp_path)
    calls: list[str] = []
    root = ET.Element("registry")

    class _DummyTree:
        def getroot(self) -> ET.Element:
            return root

    monkeypatch.setattr(gen.ET, "parse", lambda _path: _DummyTree())
    monkeypatch.setattr(
        gen,
        "load_all_type_names",
        lambda _root: calls.append("load_all_type_names") or {"VkStruct": "struct"},
    )
    monkeypatch.setattr(
        gen,
        "load_api_constants",
        lambda _root: calls.append("load_api_constants") or {"VK_MAX": 1},
    )
    monkeypatch.setattr(
        gen,
        "build_known_types",
        lambda _root: calls.append("build_known_types") or {"VkStruct"},
    )
    monkeypatch.setattr(
        gen,
        "extract_registry_version",
        lambda _root: calls.append("extract_registry_version") or "1.4.343",
    )
    monkeypatch.setattr(
        gen,
        "extract_registry",
        lambda _root, _meta: calls.append("extract_registry")
        or _make_extracted_registry(),
    )
    monkeypatch.setattr(
        gen,
        "resolve_generate_extensions",
        lambda _root, _cfg: calls.append("resolve_generate_extensions") or frozenset(),
    )
    monkeypatch.setattr(
        gen,
        "build_target_set",
        lambda *_args: calls.append("build_target_set")
        or (
            _make_target_set(frozenset(), frozenset()),
            gen.TypeClosureStats(0, 0, frozenset(), 1),
        ),
    )
    monkeypatch.setattr(
        gen,
        "filter_and_prepare",
        lambda _extracted, _target, _meta: calls.append("filter_and_prepare")
        or _make_filtered_registry(),
    )
    monkeypatch.setattr(
        gen,
        "build_write_config",
        lambda _cfg, _version: calls.append("build_write_config")
        or _make_write_config(),
    )
    monkeypatch.setattr(
        gen,
        "build_module_specs",
        lambda _root, _filtered, _write_cfg, _meta: calls.append("build_module_specs")
        or tuple(),
    )
    monkeypatch.setattr(
        gen,
        "write_package",
        lambda *_args, **_kwargs: calls.append("write_package")
        or gen.PackageWriteResult(output_dir=config.output_dir, files=tuple()),
    )

    run_generate(config)

    assert calls == [
        "load_all_type_names",
        "load_api_constants",
        "build_known_types",
        "extract_registry_version",
        "extract_registry",
        "resolve_generate_extensions",
        "build_target_set",
        "filter_and_prepare",
        "build_write_config",
        "build_module_specs",
        "write_package",
    ]


def test_t_22_run_generate_build_target_set_uses_full_extracted_structs_and_full_type_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_struct_def: Callable[[str, list[gen.StructMember]], gen.StructDef],
) -> None:
    run_generate = _require_callable("run_generate")
    config = _make_generate_config(tmp_path)
    root = ET.Element("registry")

    class _DummyTree:
        def getroot(self) -> ET.Element:
            return root

    full_structs = [make_struct_def("VkCore", []), make_struct_def("VkExt", [])]
    all_type_names = {"VkCore": "struct", "VkExt": "struct"}
    seen: dict[str, object] = {}

    monkeypatch.setattr(gen.ET, "parse", lambda _path: _DummyTree())
    monkeypatch.setattr(gen, "load_all_type_names", lambda _root: all_type_names)
    monkeypatch.setattr(gen, "load_api_constants", lambda _root: {})
    monkeypatch.setattr(gen, "build_known_types", lambda _root: set())
    monkeypatch.setattr(gen, "extract_registry_version", lambda _root: "1.4.343")
    monkeypatch.setattr(
        gen,
        "extract_registry",
        lambda _root, _meta: _make_extracted_registry(structs=full_structs),
    )
    monkeypatch.setattr(
        gen, "resolve_generate_extensions", lambda _root, _cfg: frozenset()
    )

    def _build_target_set(
        _root: ET.Element,
        _version: gen.VulkanVersion,
        _resolved: frozenset[str],
        structs: list[gen.StructDef],
        type_map: dict[str, str],
    ) -> tuple[gen.TargetSet, gen.TypeClosureStats]:
        seen["structs"] = structs
        seen["type_map"] = type_map
        return _make_target_set(frozenset(), frozenset()), gen.TypeClosureStats(
            0, 0, frozenset(), 1
        )

    monkeypatch.setattr(gen, "build_target_set", _build_target_set)
    monkeypatch.setattr(
        gen, "filter_and_prepare", lambda _e, _t, _m: _make_filtered_registry()
    )
    monkeypatch.setattr(
        gen, "build_write_config", lambda _cfg, _v: _make_write_config()
    )
    monkeypatch.setattr(gen, "build_module_specs", lambda _r, _f, _w, _m: tuple())
    monkeypatch.setattr(
        gen,
        "write_package",
        lambda *_args, **_kwargs: gen.PackageWriteResult(
            output_dir=config.output_dir, files=tuple()
        ),
    )

    run_generate(config)

    assert seen["structs"] == full_structs
    assert seen["type_map"] == all_type_names


def test_t_23_run_generate_write_package_call_shape_uses_canonical_init_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_generate = _require_callable("run_generate")
    config = _make_generate_config(tmp_path)
    root = ET.Element("registry")

    class _DummyTree:
        def getroot(self) -> ET.Element:
            return root

    write_config = _make_write_config()
    module_spec = gen.ModuleSpec(
        filename="vk_base_types.mojo",
        external_imports=tuple(),
        sibling_imports=tuple(),
        content_lines=tuple(),
    )
    seen: dict[str, object] = {}

    monkeypatch.setattr(gen.ET, "parse", lambda _path: _DummyTree())
    monkeypatch.setattr(gen, "load_all_type_names", lambda _root: {})
    monkeypatch.setattr(gen, "load_api_constants", lambda _root: {})
    monkeypatch.setattr(gen, "build_known_types", lambda _root: set())
    monkeypatch.setattr(gen, "extract_registry_version", lambda _root: "1.4.343")
    monkeypatch.setattr(
        gen, "extract_registry", lambda _root, _meta: _make_extracted_registry()
    )
    monkeypatch.setattr(
        gen, "resolve_generate_extensions", lambda _root, _cfg: frozenset()
    )
    monkeypatch.setattr(
        gen,
        "build_target_set",
        lambda *_args: (
            _make_target_set(frozenset(), frozenset()),
            gen.TypeClosureStats(0, 0, frozenset(), 1),
        ),
    )
    monkeypatch.setattr(
        gen, "filter_and_prepare", lambda _e, _t, _m: _make_filtered_registry()
    )
    monkeypatch.setattr(gen, "build_write_config", lambda _cfg, _v: write_config)
    monkeypatch.setattr(
        gen, "build_module_specs", lambda _r, _f, _w, _m: (module_spec,)
    )

    def _write_package(
        output_dir: Path,
        cfg: gen.WriteConfig,
        module_specs: tuple[gen.ModuleSpec, ...],
        init_spec: gen.InitModuleSpec,
    ) -> gen.PackageWriteResult:
        seen["output_dir"] = output_dir
        seen["config"] = cfg
        seen["module_specs"] = module_specs
        seen["init_spec"] = init_spec
        return gen.PackageWriteResult(output_dir=output_dir, files=tuple())

    monkeypatch.setattr(gen, "write_package", _write_package)

    run_generate(config)

    assert seen["output_dir"] == config.output_dir
    assert seen["config"] is write_config
    assert seen["module_specs"] == (module_spec,)
    assert seen["init_spec"].re_exports == gen.PACKAGE_MODULE_EXPORTS


def test_t_24_run_generate_returns_write_result_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_generate = _require_callable("run_generate")
    config = _make_generate_config(tmp_path)
    root = ET.Element("registry")

    class _DummyTree:
        def getroot(self) -> ET.Element:
            return root

    sentinel = gen.PackageWriteResult(output_dir=config.output_dir, files=tuple())

    monkeypatch.setattr(gen.ET, "parse", lambda _path: _DummyTree())
    monkeypatch.setattr(gen, "load_all_type_names", lambda _root: {})
    monkeypatch.setattr(gen, "load_api_constants", lambda _root: {})
    monkeypatch.setattr(gen, "build_known_types", lambda _root: set())
    monkeypatch.setattr(gen, "extract_registry_version", lambda _root: "1.4.343")
    monkeypatch.setattr(
        gen, "extract_registry", lambda _root, _meta: _make_extracted_registry()
    )
    monkeypatch.setattr(
        gen, "resolve_generate_extensions", lambda _root, _cfg: frozenset()
    )
    monkeypatch.setattr(
        gen,
        "build_target_set",
        lambda *_args: (
            _make_target_set(frozenset(), frozenset()),
            gen.TypeClosureStats(0, 0, frozenset(), 1),
        ),
    )
    monkeypatch.setattr(
        gen, "filter_and_prepare", lambda _e, _t, _m: _make_filtered_registry()
    )
    monkeypatch.setattr(
        gen, "build_write_config", lambda _cfg, _v: _make_write_config()
    )
    monkeypatch.setattr(gen, "build_module_specs", lambda _r, _f, _w, _m: tuple())
    monkeypatch.setattr(gen, "write_package", lambda *_args, **_kwargs: sentinel)

    result = run_generate(config)

    assert result is sentinel


@pytest.mark.parametrize(
    ("stage", "error"),
    [
        ("parse", OSError("no such file")),
        ("target", RuntimeError("closure safety")),
        ("assembly", ValueError("bad spec")),
    ],
)
def test_t_25_run_generate_propagates_stage_errors(
    stage: str,
    error: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_generate = _require_callable("run_generate")
    config = _make_generate_config(tmp_path)
    root = ET.Element("registry")

    class _DummyTree:
        def getroot(self) -> ET.Element:
            return root

    if stage == "parse":
        monkeypatch.setattr(gen.ET, "parse", lambda _path: (_ for _ in ()).throw(error))
    else:
        monkeypatch.setattr(gen.ET, "parse", lambda _path: _DummyTree())
        monkeypatch.setattr(gen, "load_all_type_names", lambda _root: {})
        monkeypatch.setattr(gen, "load_api_constants", lambda _root: {})
        monkeypatch.setattr(gen, "build_known_types", lambda _root: set())
        monkeypatch.setattr(gen, "extract_registry_version", lambda _root: "1.4.343")
        monkeypatch.setattr(
            gen, "extract_registry", lambda _root, _meta: _make_extracted_registry()
        )
        monkeypatch.setattr(
            gen, "resolve_generate_extensions", lambda _root, _cfg: frozenset()
        )
        if stage == "target":
            monkeypatch.setattr(
                gen, "build_target_set", lambda *_args: (_ for _ in ()).throw(error)
            )
        else:
            monkeypatch.setattr(
                gen,
                "build_target_set",
                lambda *_args: (
                    _make_target_set(frozenset(), frozenset()),
                    gen.TypeClosureStats(0, 0, frozenset(), 1),
                ),
            )
            monkeypatch.setattr(
                gen, "filter_and_prepare", lambda _e, _t, _m: _make_filtered_registry()
            )
            monkeypatch.setattr(
                gen, "build_write_config", lambda _cfg, _v: _make_write_config()
            )
            monkeypatch.setattr(
                gen, "build_module_specs", lambda *_args: (_ for _ in ()).throw(error)
            )
            monkeypatch.setattr(
                gen,
                "write_package",
                lambda *_args, **_kwargs: gen.PackageWriteResult(
                    output_dir=config.output_dir, files=tuple()
                ),
            )

    with pytest.raises(type(error), match=str(error)):
        run_generate(config)


def test_t_26_main_discovery_short_circuits_generate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = gen.DiscoveryConfig(
        command="list-versions",
        filter_text=None,
        info_extension=None,
        vk_xml=Path("vk.xml"),
    )
    calls = {"discovery": 0, "generate": 0}

    monkeypatch.setattr(gen, "build_config", lambda: discovery)
    monkeypatch.setattr(
        gen,
        "run_discovery",
        lambda _cfg: calls.__setitem__("discovery", calls["discovery"] + 1),
    )
    monkeypatch.setattr(
        gen,
        "run_generate",
        lambda _cfg: calls.__setitem__("generate", calls["generate"] + 1),
        raising=False,
    )

    gen.main()

    assert calls["discovery"] == 1
    assert calls["generate"] == 0


def test_t_27_main_generate_path_calls_run_generate_once_and_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _make_generate_config(tmp_path)
    calls = {"generate": 0}

    monkeypatch.setattr(gen, "build_config", lambda: config)

    def _run_generate(_cfg: object) -> gen.PackageWriteResult:
        calls["generate"] += 1
        print("Vulkan 1.3 bindings generated:")
        return gen.PackageWriteResult(output_dir=config.output_dir, files=tuple())

    monkeypatch.setattr(
        gen,
        "run_generate",
        _run_generate,
        raising=False,
    )
    monkeypatch.setattr(
        gen.ET,
        "parse",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("main should delegate parse to run_generate")
        ),
    )

    gen.main()
    output = capsys.readouterr().out

    assert calls["generate"] == 1
    assert "Vulkan 1.3 bindings generated:" in output
    assert "Generated:" not in output


@pytest.mark.parametrize(
    "suggestion",
    [None, "Use one of: 1.0, 1.1, 1.2, 1.3, 1.4."],
)
def test_t_28_main_maps_config_errors_to_exit_1_with_message_format(
    suggestion: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    err = gen.ConfigError(
        "INVALID_VERSION", "Unsupported Vulkan version: 9.9", suggestion
    )
    monkeypatch.setattr(gen, "build_config", lambda: (_ for _ in ()).throw(err))

    with pytest.raises(SystemExit) as exc_info:
        gen.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert "Config error [INVALID_VERSION]: Unsupported Vulkan version: 9.9" in output
    if suggestion is None:
        assert "Hint:" not in output
    else:
        assert f"Hint: {suggestion}" in output


@pytest.mark.parametrize(
    ("error", "expected_text"),
    [
        (OSError("no such file"), "no such file"),
        (ET.ParseError("malformed xml"), "malformed xml"),
    ],
)
def test_t_29_main_maps_read_errors_to_exit_1_with_user_facing_message(
    error: Exception,
    expected_text: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _make_generate_config(tmp_path)
    monkeypatch.setattr(gen, "build_config", lambda: config)
    monkeypatch.setattr(
        gen,
        "run_generate",
        lambda _cfg: (_ for _ in ()).throw(error),
        raising=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        gen.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert "Error" in output
    assert expected_text in output


def test_t_30_main_maps_runtimeerror_to_dependency_internal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _make_generate_config(tmp_path)
    monkeypatch.setattr(gen, "build_config", lambda: config)
    monkeypatch.setattr(
        gen,
        "run_generate",
        lambda _cfg: (_ for _ in ()).throw(RuntimeError("closure safety")),
        raising=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        gen.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert "Internal error" in output
    assert "closure safety" in output


def test_t_31_main_maps_valueerror_to_assembly_internal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _make_generate_config(tmp_path)
    monkeypatch.setattr(gen, "build_config", lambda: config)
    monkeypatch.setattr(
        gen,
        "run_generate",
        lambda _cfg: (_ for _ in ()).throw(ValueError("bad spec")),
        raising=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        gen.main()

    output = capsys.readouterr().out
    assert exc_info.value.code == 1
    assert "Internal error" in output
    assert "bad spec" in output


def test_t_32_main_branch_execution_is_exclusive_per_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"discovery": 0, "generate": 0}
    discovery = gen.DiscoveryConfig(
        command="list-versions",
        filter_text=None,
        info_extension=None,
        vk_xml=Path("vk.xml"),
    )
    generate = _make_generate_config(tmp_path)

    monkeypatch.setattr(
        gen,
        "run_discovery",
        lambda _cfg: calls.__setitem__("discovery", calls["discovery"] + 1),
    )
    monkeypatch.setattr(
        gen,
        "run_generate",
        lambda _cfg: calls.__setitem__("generate", calls["generate"] + 1)
        or gen.PackageWriteResult(output_dir=generate.output_dir, files=tuple()),
        raising=False,
    )

    monkeypatch.setattr(gen, "build_config", lambda: discovery)
    gen.main()
    assert calls == {"discovery": 1, "generate": 0}

    monkeypatch.setattr(gen, "build_config", lambda: generate)
    monkeypatch.setattr(
        gen.ET,
        "parse",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("main should delegate parse to run_generate")
        ),
    )
    gen.main()
    assert calls == {"discovery": 1, "generate": 1}
