import pytest

import gen


def test_parse_version_supported_returns_typed_version() -> None:
    version = gen.parse_version("1.3")

    assert version.major == 1
    assert version.minor == 3
    assert str(version) == "1.3"


@pytest.mark.parametrize("value", ["1.5", "foo", ""])
def test_parse_version_invalid_raises_invalid_version(value: str) -> None:
    with pytest.raises(gen.ConfigError) as exc_info:
        gen.parse_version(value)

    assert exc_info.value.code == "INVALID_VERSION"
    assert "1.0" in (exc_info.value.suggestion or "")
    assert "1.4" in (exc_info.value.suggestion or "")


def test_vulkan_version_ordering_contract() -> None:
    v13 = gen.VulkanVersion(1, 3)
    v14 = gen.VulkanVersion(1, 4)

    assert v13 < v14
    assert v13 <= v14
    assert v14 > v13
    assert not (v14 <= v13)
