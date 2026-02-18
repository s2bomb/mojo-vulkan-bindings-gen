from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


EXPECTED_FILES = {
    "__init__.mojo",
    "vk_base_types.mojo",
    "vk_enums.mojo",
    "vk_handles.mojo",
    "vk_structs.mojo",
    "vk_unions.mojo",
    "vk_loader.mojo",
    "vk_commands.mojo",
}


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fixture_vk_xml() -> Path:
    return _tool_root() / "tests" / "fixtures" / "vk_minimal.xml"


def _fixture_headers() -> Path:
    return _tool_root() / "tests" / "fixtures" / "fake_headers"


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    run_cwd = _tool_root() if cwd is None else cwd
    return subprocess.run(
        [sys.executable, "gen.py", *args],
        cwd=run_cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_generate(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "--version",
            "1.0",
            "--vk-xml",
            str(_fixture_vk_xml().resolve()),
            "--vulkan-headers",
            str(_fixture_headers().resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ]
    )


def test_t_01_generate_with_explicit_paths_writes_expected_surface(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "generated"

    result = _run_generate(output_dir)

    assert result.returncode == 0
    assert "Vulkan 1.0 bindings generated:" in result.stdout
    assert "Total:" in result.stdout
    assert {p.name for p in output_dir.glob("*.mojo")} == EXPECTED_FILES


def test_t_02_list_versions_succeeds_without_vulkan_headers(tmp_path: Path) -> None:
    output_dir = tmp_path / "unused-output"

    result = _run(
        [
            "--list-versions",
            "--vk-xml",
            str(_fixture_vk_xml().resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ]
    )

    assert result.returncode == 0
    assert "Vulkan versions in vk.xml" in result.stdout
    assert not output_dir.exists()


def test_t_03_list_extensions_is_read_only_operation(tmp_path: Path) -> None:
    output_dir = tmp_path / "unused-output"

    result = _run(
        [
            "--list-extensions",
            "--vk-xml",
            str(_fixture_vk_xml().resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ]
    )

    assert result.returncode == 0
    assert "Vulkan extensions" in result.stdout
    assert not output_dir.exists()


def test_t_04_info_prints_extension_detail_contract() -> None:
    result = _run(
        [
            "--info",
            "VK_KHR_swapchain",
            "--vk-xml",
            str(_fixture_vk_xml().resolve()),
        ]
    )

    assert result.returncode == 0
    assert "VK_KHR_swapchain" in result.stdout
    assert "Promoted:" in result.stdout


def test_t_05_unknown_flag_returns_argparse_usage_code() -> None:
    result = _run(["--not-a-flag"])

    assert result.returncode == 2


def test_t_06_mutually_exclusive_extension_flags_return_usage_error(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "generated"

    result = _run(
        [
            "--version",
            "1.0",
            "--ext",
            "VK_KHR_swapchain",
            "--all-extensions",
            "--vk-xml",
            str(_fixture_vk_xml().resolve()),
            "--vulkan-headers",
            str(_fixture_headers().resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ]
    )

    assert result.returncode == 2


def test_t_07_generate_mode_requires_version_even_with_valid_paths() -> None:
    result = _run(
        [
            "--vk-xml",
            str(_fixture_vk_xml().resolve()),
            "--vulkan-headers",
            str(_fixture_headers().resolve()),
        ]
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "MISSING_VERSION" in combined_output


def test_t_08_standalone_missing_vk_xml_degrades_without_traceback(
    tmp_path: Path,
) -> None:
    isolated_gen = tmp_path / "gen.py"
    shutil.copy2(_tool_root() / "gen.py", isolated_gen)

    result = subprocess.run(
        [sys.executable, str(isolated_gen), "--version", "1.3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "PATH_NOT_FOUND" in combined_output
    assert "--vk-xml" in combined_output
    assert "Traceback (most recent call last)" not in combined_output


def test_t_09_help_stable_surface_includes_public_flags() -> None:
    result = _run(["--help"])

    combined_output = result.stdout + result.stderr
    assert result.returncode == 0
    for flag in (
        "--version",
        "--vk-xml",
        "--vulkan-headers",
        "--output-dir",
        "--list-versions",
        "--list-extensions",
        "--info",
    ):
        assert flag in combined_output


def test_t_18_generated_file_set_is_exactly_eight_non_empty_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "generated"
    result = _run_generate(output_dir)

    assert result.returncode == 0
    generated = list(output_dir.glob("*.mojo"))
    assert {p.name for p in generated} == EXPECTED_FILES
    for file_path in generated:
        assert file_path.stat().st_size > 0


def test_t_19_generated_files_include_provenance_header_anchors(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "generated"
    result = _run_generate(output_dir)

    assert result.returncode == 0
    for file_path in output_dir.glob("*.mojo"):
        content = file_path.read_text(encoding="utf-8")
        assert "Generated by vulkan-bindings-gen" in content
        assert "Target: Vulkan" in content


def test_t_20_generated_init_contains_no_legacy_vk_route(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"
    result = _run_generate(output_dir)

    assert result.returncode == 0
    init_content = (output_dir / "__init__.mojo").read_text(encoding="utf-8")
    assert "from .vk import *" not in init_content
    assert "vk.mojo" not in init_content


def test_t_21_generated_init_reexports_all_required_module_stems(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "generated"
    result = _run_generate(output_dir)

    assert result.returncode == 0
    init_content = (output_dir / "__init__.mojo").read_text(encoding="utf-8")
    for stem in (
        "vk_base_types",
        "vk_enums",
        "vk_handles",
        "vk_structs",
        "vk_unions",
        "vk_loader",
        "vk_commands",
    ):
        assert f"from .{stem}" in init_content
