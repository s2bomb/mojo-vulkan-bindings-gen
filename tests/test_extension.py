import pytest

import gen


@pytest.mark.parametrize(
    "name",
    [
        "VK_KHR_swapchain",
        "VK_EXT_debug_utils",
        "VK_KHR_8bit_storage",
    ],
)
def test_validate_extension_name_accepts_valid_names(name: str) -> None:
    assert gen.validate_extension_name(name) == name


@pytest.mark.parametrize("name", ["khr_swapchain", "VK_KHR", "not_an_extension"])
def test_validate_extension_name_rejects_malformed_names(name: str) -> None:
    with pytest.raises(gen.ConfigError) as exc_info:
        gen.validate_extension_name(name)

    assert exc_info.value.code == "INVALID_EXTENSION_NAME"
    assert "VK_" in (exc_info.value.suggestion or "")
