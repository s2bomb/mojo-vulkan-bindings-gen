from __future__ import annotations

from collections.abc import Callable
import xml.etree.ElementTree as ET

import gen


def _require_callable(name: str) -> Callable[..., object]:
    symbol = getattr(gen, name, None)
    assert callable(symbol), f"Missing S7 API symbol: gen.{name}"
    return symbol


def test_resolve_deps_no_dependency(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        '<extensions><extension name="VK_A" supported="vulkan"/></extensions>'
    )

    assert resolve_extension_deps(root, frozenset({"VK_A"})) == frozenset({"VK_A"})


def test_resolve_deps_direct_dependency(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan"/>
            <extension name="VK_B" supported="vulkan" depends="VK_A"/>
        </extensions>
        """
    )

    assert resolve_extension_deps(root, frozenset({"VK_B"})) == frozenset(
        {"VK_A", "VK_B"}
    )


def test_resolve_deps_transitive_chain(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan"/>
            <extension name="VK_B" supported="vulkan" depends="VK_A"/>
            <extension name="VK_C" supported="vulkan" depends="VK_B"/>
        </extensions>
        """
    )

    assert resolve_extension_deps(root, frozenset({"VK_C"})) == frozenset(
        {"VK_A", "VK_B", "VK_C"}
    )


def test_resolve_deps_diamond_deduplication(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_C" supported="vulkan"/>
            <extension name="VK_B" supported="vulkan" depends="VK_C"/>
            <extension name="VK_A" supported="vulkan" depends="VK_B,VK_C"/>
        </extensions>
        """
    )

    assert resolve_extension_deps(root, frozenset({"VK_A"})) == frozenset(
        {"VK_A", "VK_B", "VK_C"}
    )


def test_resolve_deps_multiple_inputs_merged(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        """
        <extensions>
            <extension name="VK_A" supported="vulkan"/>
            <extension name="VK_B" supported="vulkan" depends="VK_A"/>
            <extension name="VK_C" supported="vulkan"/>
            <extension name="VK_D" supported="vulkan" depends="VK_C"/>
        </extensions>
        """
    )

    assert resolve_extension_deps(root, frozenset({"VK_B", "VK_D"})) == frozenset(
        {"VK_A", "VK_B", "VK_C", "VK_D"}
    )


def test_resolve_deps_already_promoted_extension_included(
    make_registry_root: Callable[[str], ET.Element],
) -> None:
    resolve_extension_deps = _require_callable("resolve_extension_deps")
    root = make_registry_root(
        """
        <extensions>
            <extension
                name="VK_KHR_dynamic_rendering"
                supported="vulkan"
                promotedto="VK_VERSION_1_3"
            />
        </extensions>
        """
    )

    resolved = resolve_extension_deps(root, frozenset({"VK_KHR_dynamic_rendering"}))
    assert "VK_KHR_dynamic_rendering" in resolved
