from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import gen


def _assert_config_code(exc_info: pytest.ExceptionInfo[Exception], code: str) -> None:
    err = exc_info.value
    assert getattr(err, "code") == code
    assert getattr(err, "code") in gen.VALID_ERROR_CODES


def test_import_gen_module_smoke() -> None:
    assert callable(gen.main)


def test_build_argument_parser_exposes_s1_surface_and_defaults() -> None:
    parser = gen.build_argument_parser()
    option_actions = {
        option: action for action in parser._actions for option in action.option_strings
    }

    expected_options = {
        "--version",
        "--ext",
        "--all-extensions",
        "--vk-xml",
        "--vulkan-headers",
        "--output-dir",
        "--list-versions",
        "--list-extensions",
        "--info",
        "--filter",
    }

    assert expected_options.issubset(option_actions.keys())
    assert option_actions["--vk-xml"].default == gen.DEFAULT_VK_XML
    assert option_actions["--vulkan-headers"].default == gen.DEFAULT_VULKAN_HEADERS
    assert option_actions["--output-dir"].default == gen.DEFAULT_OUTPUT_DIR
    assert option_actions["--list-versions"].default is False
    assert option_actions["--list-extensions"].default is False
    assert option_actions["--info"].default is None
    assert option_actions["--filter"].default is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--ext", "VK_KHR_swapchain", "--all-extensions"],
        ["--list-versions", "--list-extensions"],
    ],
)
def test_parse_args_enforces_argparse_mutual_exclusion(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        gen.parse_args(argv)

    assert exc_info.value.code == 2


def test_parse_args_maps_valid_argv_without_semantic_validation(
    existing_paths: dict[str, Path],
) -> None:
    argv = [
        "--version",
        "1.3",
        "--ext",
        "VK_KHR_swapchain",
        "--vk-xml",
        str(existing_paths["vk_xml"]),
    ]

    args = gen.parse_args(argv)

    assert args.version == "1.3"
    assert args.ext == [["VK_KHR_swapchain"]]
    assert isinstance(args.vk_xml, Path)
    assert args.vk_xml == existing_paths["vk_xml"]


def test_parse_args_accepts_space_separated_extensions_for_single_ext_flag() -> None:
    args = gen.parse_args(
        ["--ext", "VK_KHR_swapchain", "VK_EXT_debug_utils", "--version", "1.3"]
    )

    assert args.ext == [["VK_KHR_swapchain", "VK_EXT_debug_utils"]]


def test_validate_config_normalizes_space_separated_extensions_from_parser(
    existing_paths: dict[str, Path],
) -> None:
    args = gen.parse_args(
        [
            "--version",
            "1.3",
            "--ext",
            "VK_KHR_swapchain",
            "VK_EXT_debug_utils",
            "--vk-xml",
            str(existing_paths["vk_xml"]),
            "--vulkan-headers",
            str(existing_paths["vulkan_headers"]),
        ]
    )

    config = gen.validate_config(args)

    assert isinstance(config, gen.GenerateConfig)
    assert config.extensions == frozenset({"VK_KHR_swapchain", "VK_EXT_debug_utils"})


def test_parse_args_unknown_flag_exits_with_code_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        gen.parse_args(["--not-a-flag"])

    assert exc_info.value.code == 2


def test_validate_path_exists_accepts_existing_path(
    existing_paths: dict[str, Path],
) -> None:
    path = existing_paths["vk_xml"]
    assert gen.validate_path_exists(path, "--vk-xml") == path


def test_validate_path_exists_rejects_none_with_path_not_found() -> None:
    with pytest.raises(gen.ConfigError) as exc_info:
        gen.validate_path_exists(None, "--vk-xml")

    _assert_config_code(exc_info, "PATH_NOT_FOUND")
    assert "--vk-xml" in getattr(exc_info.value, "message")


def test_validate_path_exists_raises_path_not_found(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist.xml"

    with pytest.raises(gen.ConfigError) as exc_info:
        gen.validate_path_exists(missing, "--vk-xml")

    _assert_config_code(exc_info, "PATH_NOT_FOUND")
    assert "--vk-xml" in getattr(exc_info.value, "message")
    assert str(missing) in getattr(exc_info.value, "message")


def test_validate_config_generate_mode_returns_generate_config(
    make_args: Callable[..., object],
) -> None:
    args = make_args(version="1.3", ext=["VK_KHR_swapchain"])

    config = gen.validate_config(args)

    assert isinstance(config, gen.GenerateConfig)
    assert config.version == gen.VulkanVersion(1, 3)
    assert config.extensions == frozenset({"VK_KHR_swapchain"})
    assert config.all_extensions is False


def test_validate_config_discovery_mode_skips_vulkan_headers_path_check(
    make_args: Callable[..., object],
    missing_path: Path,
) -> None:
    args = make_args(
        list_versions=True,
        vulkan_headers=missing_path,
        output_dir=missing_path,
    )

    config = gen.validate_config(args)

    assert isinstance(config, gen.DiscoveryConfig)
    assert config.command == "list-versions"
    assert config.filter_text is None
    assert config.info_extension is None


def test_validate_config_inferred_generate_mode_requires_version(
    make_args: Callable[..., object],
) -> None:
    args = make_args()

    with pytest.raises(gen.ConfigError) as exc_info:
        gen.validate_config(args)

    _assert_config_code(exc_info, "MISSING_VERSION")


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": "1.3", "list_versions": True},
        {"ext": ["VK_KHR_swapchain"], "list_versions": True},
        {"all_extensions": True, "list_extensions": True},
    ],
)
def test_validate_config_rejects_cross_mode_conflicts(
    make_args: Callable[..., object],
    overrides: dict[str, object],
) -> None:
    args = make_args(**overrides)

    with pytest.raises(gen.ConfigError) as exc_info:
        gen.validate_config(args)

    _assert_config_code(exc_info, "CONFLICT_GENERATE_DISCOVERY")


@pytest.mark.parametrize(
    "overrides",
    [
        {"filter": "swap"},
        {"filter": "swap", "list_versions": True},
        {"filter": "swap", "info": "VK_KHR_swapchain"},
    ],
)
def test_validate_config_filter_requires_list_extensions(
    make_args: Callable[..., object],
    overrides: dict[str, object],
) -> None:
    args = make_args(**overrides)

    with pytest.raises(gen.ConfigError) as exc_info:
        gen.validate_config(args)

    _assert_config_code(exc_info, "FILTER_WITHOUT_LIST")


def test_validate_config_path_policy_differs_by_mode(
    make_args: Callable[..., object],
    missing_path: Path,
) -> None:
    generate_args = make_args(version="1.3", vulkan_headers=missing_path)

    with pytest.raises(gen.ConfigError) as exc_info:
        gen.validate_config(generate_args)

    _assert_config_code(exc_info, "PATH_NOT_FOUND")

    discovery_args = make_args(list_versions=True, vulkan_headers=missing_path)
    discovery_config = gen.validate_config(discovery_args)
    assert isinstance(discovery_config, gen.DiscoveryConfig)


def test_validate_config_returns_frozen_dataclasses(
    make_args: Callable[..., object],
) -> None:
    generate_config = gen.validate_config(make_args(version="1.3"))
    discovery_config = gen.validate_config(make_args(list_versions=True))

    with pytest.raises(FrozenInstanceError):
        generate_config.version = gen.VulkanVersion(1, 4)

    with pytest.raises(FrozenInstanceError):
        discovery_config.command = "info"


def test_build_config_generate_success_composes_parse_and_validate(
    existing_paths: dict[str, Path],
) -> None:
    argv = [
        "--version",
        "1.3",
        "--ext",
        "VK_KHR_swapchain",
        "--vk-xml",
        str(existing_paths["vk_xml"]),
        "--vulkan-headers",
        str(existing_paths["vulkan_headers"]),
    ]

    config = gen.build_config(argv)

    assert isinstance(config, gen.GenerateConfig)
    assert config.version == gen.VulkanVersion(1, 3)
    assert config.extensions == frozenset({"VK_KHR_swapchain"})


def test_build_config_accepts_single_ext_flag_with_multiple_values(
    existing_paths: dict[str, Path],
) -> None:
    argv = [
        "--version",
        "1.3",
        "--ext",
        "VK_KHR_swapchain",
        "VK_EXT_debug_utils",
        "--vk-xml",
        str(existing_paths["vk_xml"]),
        "--vulkan-headers",
        str(existing_paths["vulkan_headers"]),
    ]

    config = gen.build_config(argv)

    assert isinstance(config, gen.GenerateConfig)
    assert config.extensions == frozenset({"VK_KHR_swapchain", "VK_EXT_debug_utils"})


@pytest.mark.parametrize(
    "argv",
    [
        ["--not-a-flag"],
        ["--ext", "VK_KHR_swapchain", "--all-extensions"],
    ],
)
def test_build_config_propagates_argparse_usage_errors(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        gen.build_config(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("argv", "expected_code"),
    [
        ([], "MISSING_VERSION"),
        (["--filter", "swap"], "FILTER_WITHOUT_LIST"),
        (["--version", "1.5"], "INVALID_VERSION"),
    ],
)
def test_build_config_propagates_semantic_errors(
    argv: list[str],
    expected_code: str,
) -> None:
    with pytest.raises(gen.ConfigError) as exc_info:
        gen.build_config(argv)

    _assert_config_code(exc_info, expected_code)


def test_error_taxonomy_is_bounded_and_machine_readable(
    make_args: Callable[..., object],
    missing_path: Path,
) -> None:
    expected_codes = {
        "INVALID_VERSION",
        "MISSING_VERSION",
        "INVALID_EXTENSION_NAME",
        "CONFLICT_EXT_FLAGS",
        "CONFLICT_GENERATE_DISCOVERY",
        "FILTER_WITHOUT_LIST",
        "PATH_NOT_FOUND",
    }

    assert gen.VALID_ERROR_CODES == expected_codes

    scenarios = [
        ("invalid-version", lambda: gen.parse_version("1.5"), "INVALID_VERSION"),
        (
            "missing-version",
            lambda: gen.validate_config(make_args()),
            "MISSING_VERSION",
        ),
        (
            "invalid-extension",
            lambda: gen.validate_extension_name("not_an_extension"),
            "INVALID_EXTENSION_NAME",
        ),
        (
            "conflict-ext-flags",
            lambda: gen.validate_config(
                make_args(
                    version="1.3",
                    ext=["VK_KHR_swapchain"],
                    all_extensions=True,
                )
            ),
            "CONFLICT_EXT_FLAGS",
        ),
        (
            "conflict-generate-discovery",
            lambda: gen.validate_config(make_args(version="1.3", list_versions=True)),
            "CONFLICT_GENERATE_DISCOVERY",
        ),
        (
            "filter-without-list",
            lambda: gen.validate_config(make_args(filter="swap")),
            "FILTER_WITHOUT_LIST",
        ),
        (
            "path-not-found",
            lambda: gen.validate_path_exists(missing_path, "--vk-xml"),
            "PATH_NOT_FOUND",
        ),
    ]

    for _name, invoke, expected_code in scenarios:
        with pytest.raises(gen.ConfigError) as exc_info:
            invoke()
        _assert_config_code(exc_info, expected_code)
