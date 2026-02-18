import argparse
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest

import gen

GENERATOR_DIR = Path(__file__).resolve().parent.parent
if str(GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATOR_DIR))


@pytest.fixture
def existing_paths(tmp_path: Path) -> dict[str, Path]:
    vk_xml = tmp_path / "vk.xml"
    vk_xml.write_text("<registry />\n", encoding="utf-8")

    vulkan_headers = tmp_path / "Vulkan-Headers" / "include"
    vulkan_headers.mkdir(parents=True)

    output_dir = tmp_path / "out"
    return {
        "vk_xml": vk_xml,
        "vulkan_headers": vulkan_headers,
        "output_dir": output_dir,
    }


@pytest.fixture
def missing_path(tmp_path: Path) -> Path:
    return tmp_path / "missing"


@pytest.fixture
def make_args(existing_paths: dict[str, Path]) -> Callable[..., argparse.Namespace]:
    def _make_args(**overrides: object) -> argparse.Namespace:
        base_args: dict[str, object] = {
            "version": None,
            "ext": None,
            "all_extensions": False,
            "vk_xml": existing_paths["vk_xml"],
            "vulkan_headers": existing_paths["vulkan_headers"],
            "output_dir": existing_paths["output_dir"],
            "list_versions": False,
            "list_extensions": False,
            "info": None,
            "filter": None,
        }
        base_args.update(overrides)
        return argparse.Namespace(**base_args)

    return _make_args


@pytest.fixture
def make_registry_root() -> Callable[[str], ET.Element]:
    def _make_registry_root(inner_xml: str) -> ET.Element:
        return ET.fromstring(f"<registry>{inner_xml}</registry>")

    return _make_registry_root


@pytest.fixture
def make_struct_member() -> Callable[..., gen.StructMember]:
    def _make_struct_member(
        *,
        name: str,
        type_name: str,
        is_pointer: bool = False,
        is_double_pointer: bool = False,
        array_size: str | None = None,
        stype_value: str | None = None,
        bitwidth: int | None = None,
    ) -> gen.StructMember:
        return gen.StructMember(
            name=name,
            type_name=type_name,
            is_pointer=is_pointer,
            is_double_pointer=is_double_pointer,
            array_size=array_size,
            stype_value=stype_value,
            bitwidth=bitwidth,
        )

    return _make_struct_member


@pytest.fixture
def make_struct_def() -> Callable[[str, list[gen.StructMember]], gen.StructDef]:
    def _make_struct_def(name: str, members: list[gen.StructMember]) -> gen.StructDef:
        return gen.StructDef(name, members)

    return _make_struct_def
