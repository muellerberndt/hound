"""
Tests for the config_loader module.
"""

import os
import tempfile
from pathlib import Path
import pytest
import yaml
from unittest.mock import patch, MagicMock

from utils.config_loader import load_config


class TestConfigLoader:
    """Test suite for config_loader functionality."""

    def test_load_explicit_config_path(self):
        """Test loading config from explicitly provided path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "models": {
                    "agent": {"platform": "OpenAI", "model": "gpt-5"},
                    "strategist": {"platform": "OpenAI", "model": "gpt-5-mini"},
                },
                "analysis": {"max_iterations": 10},
            }
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            # Load config from explicit path
            loaded_config = load_config(config_path)

            # Verify config was loaded correctly
            assert loaded_config == config_data
            assert loaded_config["models"]["agent"]["model"] == "gpt-5"
            assert loaded_config["analysis"]["max_iterations"] == 10
        finally:
            # Clean up
            os.unlink(config_path)

    def test_load_from_environment_variable(self):
        """Test loading config from HOUND_CONFIG environment variable."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "models": {"reporting": {"platform": "Anthropic", "model": "claude-3-sonnet"}},
                "debug": True,
            }
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Set environment variable
            os.environ["HOUND_CONFIG"] = config_path

            # Load config (no explicit path provided)
            loaded_config = load_config()

            # Verify config was loaded from environment
            assert loaded_config == config_data
            assert loaded_config["models"]["reporting"]["model"] == "claude-3-sonnet"
            assert loaded_config["debug"] is True
        finally:
            # Clean up
            os.environ.pop("HOUND_CONFIG", None)
            os.unlink(config_path)

    def test_load_from_current_directory(self):
        """Test loading config.yaml from current directory."""
        # Save current directory
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Change to temp directory
                os.chdir(tmpdir)

                # Create config.yaml in current directory
                config_data = {
                    "models": {"agent": {"platform": "Local", "model": "llama-3"}},
                    "features": {"auto_confirm": False},
                }
                config_path = Path("config.yaml")
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f)

                # Load config (no explicit path, no env var)
                loaded_config = load_config()

                # Verify config was loaded from current directory
                assert loaded_config == config_data
                assert loaded_config["models"]["agent"]["model"] == "llama-3"
            finally:
                # Restore original directory
                os.chdir(original_cwd)

    def test_load_from_hound_directory(self):
        """Test loading config.yaml from hound directory."""
        # Mock the path resolution to avoid dependency on actual file structure
        with patch("utils.config_loader.Path") as MockPath:
            # Setup mock paths
            mock_hound_dir = MagicMock()
            mock_config_file = MagicMock()
            mock_config_file.exists.return_value = True

            # Configure mock to return our test config
            config_data = {
                "models": {"junior": {"platform": "OpenAI", "model": "gpt-4"}},
                "timeout": 300,
            }

            # Mock the file reading
            with patch("builtins.open", create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value = mock_file
                mock_file.__exit__.return_value = None
                mock_open.return_value = mock_file

                with patch("yaml.safe_load") as mock_yaml_load:
                    mock_yaml_load.return_value = config_data

                    # Setup path mocking
                    MockPath.cwd.return_value = MagicMock()
                    MockPath.cwd.return_value.__truediv__.return_value.exists.return_value = False
                    MockPath.return_value.exists.return_value = False
                    MockPath.__file__ = MagicMock()
                    MockPath.__file__.parent.parent = mock_hound_dir
                    mock_hound_dir.__truediv__.return_value = mock_config_file

                    # Load config
                    loaded_config = load_config()

                    # Should return the mocked config
                    assert loaded_config == config_data

    def test_fallback_to_example_config(self):
        """Test fallback to config.example.yaml."""
        with patch("utils.config_loader.Path") as MockPath:
            # Mock all paths to not exist except example
            mock_example_file = MagicMock()
            mock_example_file.exists.return_value = True

            config_data = {
                "models": {"default": {"platform": "Example", "model": "example-model"}},
                "example": True,
            }

            with patch("builtins.open", create=True) as mock_open:
                mock_file = MagicMock()
                mock_file.__enter__.return_value = mock_file
                mock_file.__exit__.return_value = None
                mock_open.return_value = mock_file

                with patch("yaml.safe_load") as mock_yaml_load:
                    mock_yaml_load.return_value = config_data

                    # Setup complex path mocking
                    MockPath.cwd.return_value = MagicMock()
                    MockPath.cwd.return_value.__truediv__.return_value.exists.return_value = False
                    MockPath.return_value.exists.return_value = False

                    mock_hound_dir = MagicMock()
                    MockPath.__file__ = MagicMock()
                    MockPath.__file__.parent.parent = mock_hound_dir

                    # config.yaml doesn't exist
                    mock_config = MagicMock()
                    mock_config.exists.return_value = False

                    # config.example.yaml exists
                    mock_example = MagicMock()
                    mock_example.exists.return_value = True

                    def side_effect(arg):
                        if arg == "config.yaml":
                            return mock_config
                        elif arg == "config.example.yaml":
                            return mock_example
                        return MagicMock()

                    mock_hound_dir.__truediv__.side_effect = side_effect

                    # Load config
                    loaded_config = load_config()

                    assert loaded_config == config_data

    def test_empty_dict_fallback(self):
        """Test that empty dict is returned when no config found."""
        with patch("utils.config_loader.Path") as MockPath:
            # Mock all paths to not exist
            MockPath.cwd.return_value = MagicMock()
            MockPath.cwd.return_value.__truediv__.return_value.exists.return_value = False
            MockPath.return_value.exists.return_value = False
            MockPath.__file__ = MagicMock()

            mock_hound_dir = MagicMock()
            MockPath.__file__.parent.parent = mock_hound_dir

            mock_file = MagicMock()
            mock_file.exists.return_value = False
            mock_hound_dir.__truediv__.return_value = mock_file

            # Load config
            loaded_config = load_config()

            # Should return empty dict
            assert loaded_config == {}

    def test_priority_order(self):
        """Test that config sources are checked in correct priority order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple config files
            explicit_config = Path(tmpdir) / "explicit.yaml"
            env_config = Path(tmpdir) / "env.yaml"
            cwd_config = Path(tmpdir) / "config.yaml"

            explicit_data = {"source": "explicit", "priority": 1}
            env_data = {"source": "env", "priority": 2}
            cwd_data = {"source": "cwd", "priority": 3}

            with open(explicit_config, "w") as f:
                yaml.dump(explicit_data, f)
            with open(env_config, "w") as f:
                yaml.dump(env_data, f)
            with open(cwd_config, "w") as f:
                yaml.dump(cwd_data, f)

            # Test 1: Explicit path takes priority
            loaded = load_config(explicit_config)
            assert loaded["source"] == "explicit"

            # Test 2: Environment variable takes priority over cwd
            os.environ["HOUND_CONFIG"] = str(env_config)
            try:
                original_cwd = os.getcwd()
                os.chdir(tmpdir)
                loaded = load_config()  # No explicit path
                assert loaded["source"] == "env"
            finally:
                os.chdir(original_cwd)
                os.environ.pop("HOUND_CONFIG", None)

            # Test 3: Current directory config when no env var
            try:
                original_cwd = os.getcwd()
                os.chdir(tmpdir)
                loaded = load_config()  # No explicit path, no env var
                assert loaded["source"] == "cwd"
            finally:
                os.chdir(original_cwd)

    def test_yaml_parse_error_handling(self):
        """Test handling of invalid YAML files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Write invalid YAML
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)

        try:
            # Should raise an error or return empty dict depending on implementation
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            os.unlink(config_path)

    def test_nonexistent_explicit_path(self):
        """Test behavior when explicit path doesn't exist."""
        nonexistent_path = Path("/tmp/nonexistent_config_12345.yaml")

        # When explicit path doesn't exist, it falls back to default search
        # So we just verify it returns a dict (could be empty or default config)
        loaded_config = load_config(nonexistent_path)

        # Should return a dictionary (empty or with default config)
        assert isinstance(loaded_config, dict)
