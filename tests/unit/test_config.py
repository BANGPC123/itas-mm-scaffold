import pytest

from src.utils.config import ConfigError, load_config


def test_load_existing_config_returns_dict():
    config = load_config("perception")
    assert isinstance(config, dict)
    assert "sign_detector" in config


def test_load_missing_config_raises_config_error():
    with pytest.raises(ConfigError):
        load_config("this_config_does_not_exist")


def test_all_four_stage_configs_load_without_error():
    for name in ["perception", "context", "reasoning", "interaction"]:
        config = load_config(name)
        assert config is not None
