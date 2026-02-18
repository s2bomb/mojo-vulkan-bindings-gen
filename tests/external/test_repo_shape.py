from __future__ import annotations

from pathlib import Path


_README_ANCHORS = (
    # One-line summary
    "Generate typed Vulkan FFI bindings for Mojo",
    # Prerequisites section
    "Prerequisites",
    # Quick start — all four required flags present
    "--version",
    "--vk-xml",
    "--vulkan-headers",
    "--output-dir",
    # 8-file output table
    "__init__.mojo",
    "vk_base_types.mojo",
    "vk_enums.mojo",
    "vk_handles.mojo",
    "vk_structs.mojo",
    "vk_unions.mojo",
    "vk_loader.mojo",
    "vk_commands.mojo",
    # Discovery examples section
    "--list-versions",
    "--list-extensions",
    "--info",
    # Testing instructions
    "pytest",
    # License section
    "MIT",
)

_AGENTS_MULTIFILE_ANCHORS = (
    "__init__.mojo",
    "vk_base_types.mojo",
    "vk_enums.mojo",
    "vk_handles.mojo",
    "vk_structs.mojo",
    "vk_unions.mojo",
    "vk_loader.mojo",
    "vk_commands.mojo",
    "--vk-xml",
    "--vulkan-headers",
    "--output-dir",
)


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_t_25_required_oss_publication_artifacts_exist() -> None:
    tool_root = _tool_root()
    required_paths = {
        "gen.py",
        "README.md",
        "LICENSE",
        "examples/generate_1_3.sh",
        "examples/generate_with_ext.sh",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_cli.py",
        "tests/test_version.py",
        "tests/test_extension.py",
        "tests/test_extension_deps.py",
        "tests/test_target_set.py",
        "tests/test_discovery.py",
        "tests/test_writer.py",
        "tests/test_pipeline.py",
        "tests/test_summary.py",
        "tests/fixtures/vk_minimal.xml",
        "tests/fixtures/fake_headers/vulkan/vulkan.h",
        "tests/external/test_external_cli.py",
        "tests/external/test_repo_shape.py",
    }

    missing = sorted(path for path in required_paths if not (tool_root / path).exists())
    assert missing == []


def test_t_26_forbidden_oss_artifacts_are_absent() -> None:
    tool_root = _tool_root()

    forbidden_paths = (
        "thoughts",
        "libs/vulkan",
        "vk.xml",
        "Vulkan-Headers",
    )
    for relative_path in forbidden_paths:
        assert not (tool_root / relative_path).exists()

    assert list(tool_root.rglob("*.mojopkg")) == []


def test_t_27_readme_includes_required_sections_and_canonical_quick_start() -> None:
    readme = _tool_root() / "README.md"
    assert readme.exists(), "README.md must exist"
    content = readme.read_text(encoding="utf-8")
    missing = [anchor for anchor in _README_ANCHORS if anchor not in content]
    assert missing == [], f"README.md missing required anchors: {missing}"


def test_t_28_agents_md_reflects_multifile_output_and_explicit_path_flags() -> None:
    agents_md = _tool_root() / "AGENTS.md"
    assert agents_md.exists(), "AGENTS.md must exist"
    content = agents_md.read_text(encoding="utf-8")
    missing = [anchor for anchor in _AGENTS_MULTIFILE_ANCHORS if anchor not in content]
    assert missing == [], f"AGENTS.md missing required anchors: {missing}"
