"""
Configuration Tests

Comprehensive tests for the application configuration including:
- Settings loading and validation
- Environment variable handling
- Default values
- Type conversions
- Cache behavior
"""

import os
import pytest
from unittest.mock import patch

from app.config import Settings, get_settings, settings


class TestSettingsDefaults:
    """Tests for default configuration values."""
    
    def test_default_host(self):
        """Test default host is 0.0.0.0."""
        s = Settings()
        assert s.HOST == "0.0.0.0"
    
    def test_default_port(self):
        """Test default port is 8000."""
        s = Settings()
        assert s.PORT == 8000
    
    def test_default_workers(self):
        """Test default workers is 1."""
        s = Settings()
        assert s.WORKERS == 1
    
    def test_default_debug(self):
        """Test debug is disabled by default."""
        s = Settings()
        assert s.DEBUG is False
    
    def test_default_gpu_enabled(self):
        """Test GPU is enabled by default."""
        s = Settings()
        assert s.USE_GPU is True
    
    def test_default_cuda_device(self):
        """Test default CUDA device is 0."""
        s = Settings()
        assert s.CUDA_VISIBLE_DEVICES == "0"
    
    def test_default_model_name(self):
        """Test default model is resnet50."""
        s = Settings()
        assert s.MODEL_NAME == "resnet50"
    
    def test_default_batch_size(self):
        """Test default batch size is 8."""
        s = Settings()
        assert s.BATCH_SIZE == 8
    
    def test_default_max_file_size(self):
        """Test default max file size is 10MB."""
        s = Settings()
        assert s.MAX_FILE_SIZE == 10 * 1024 * 1024  # 10MB
    
    def test_default_max_batch_size(self):
        """Test default max batch size is 20."""
        s = Settings()
        assert s.MAX_BATCH_SIZE == 20
    
    def test_default_allowed_extensions(self):
        """Test default allowed extensions."""
        s = Settings()
        expected = {".jpg", ".jpeg", ".png", ".webp"}
        assert s.ALLOWED_EXTENSIONS == expected
    
    def test_default_log_level(self):
        """Test default log level is INFO."""
        s = Settings()
        assert s.LOG_LEVEL == "INFO"
    
    def test_default_api_prefix(self):
        """Test default API prefix."""
        s = Settings()
        assert s.API_PREFIX == "/api/v1"


class TestSettingsEnvironment:
    """Tests for environment variable loading."""
    
    def test_host_from_env(self):
        """Test HOST can be set from environment."""
        with patch.dict(os.environ, {"HOST": "127.0.0.1"}):
            s = Settings()
            assert s.HOST == "127.0.0.1"
    
    def test_port_from_env(self):
        """Test PORT can be set from environment."""
        with patch.dict(os.environ, {"PORT": "9000"}):
            s = Settings()
            assert s.PORT == 9000
    
    def test_debug_from_env(self):
        """Test DEBUG can be set from environment."""
        with patch.dict(os.environ, {"DEBUG": "true"}):
            s = Settings()
            assert s.DEBUG is True
    
    def test_use_gpu_from_env(self):
        """Test USE_GPU can be set from environment."""
        with patch.dict(os.environ, {"USE_GPU": "false"}):
            s = Settings()
            assert s.USE_GPU is False
    
    def test_model_name_from_env(self):
        """Test MODEL_NAME can be set from environment."""
        with patch.dict(os.environ, {"MODEL_NAME": "resnet101"}):
            s = Settings()
            assert s.MODEL_NAME == "resnet101"
    
    def test_batch_size_from_env(self):
        """Test BATCH_SIZE can be set from environment."""
        with patch.dict(os.environ, {"BATCH_SIZE": "16"}):
            s = Settings()
            assert s.BATCH_SIZE == 16
    
    def test_max_file_size_from_env(self):
        """Test MAX_FILE_SIZE can be set from environment."""
        with patch.dict(os.environ, {"MAX_FILE_SIZE": "20971520"}):  # 20MB
            s = Settings()
            assert s.MAX_FILE_SIZE == 20971520
    
    def test_log_level_from_env(self):
        """Test LOG_LEVEL can be set from environment."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            s = Settings()
            assert s.LOG_LEVEL == "DEBUG"


class TestSettingsValidation:
    """Tests for settings validation."""
    
    def test_port_is_integer(self):
        """Test PORT is converted to integer."""
        with patch.dict(os.environ, {"PORT": "8080"}):
            s = Settings()
            assert isinstance(s.PORT, int)
    
    def test_batch_size_is_integer(self):
        """Test BATCH_SIZE is converted to integer."""
        s = Settings()
        assert isinstance(s.BATCH_SIZE, int)
    
    def test_debug_is_boolean(self):
        """Test DEBUG is converted to boolean."""
        s = Settings()
        assert isinstance(s.DEBUG, bool)
    
    def test_use_gpu_is_boolean(self):
        """Test USE_GPU is converted to boolean."""
        s = Settings()
        assert isinstance(s.USE_GPU, bool)
    
    def test_allowed_extensions_is_set(self):
        """Test ALLOWED_EXTENSIONS is a set."""
        s = Settings()
        assert isinstance(s.ALLOWED_EXTENSIONS, set)


class TestCorsSettings:
    """Tests for CORS configuration."""
    
    def test_default_cors_origins(self):
        """Test default CORS origins."""
        s = Settings()
        assert "localhost:3000" in s.CORS_ORIGINS
        assert "localhost:5173" in s.CORS_ORIGINS
    
    def test_cors_origins_from_env(self):
        """Test CORS origins from environment."""
        with patch.dict(os.environ, {"CORS_ORIGINS": "http://example.com,http://test.com"}):
            s = Settings()
            assert "example.com" in s.CORS_ORIGINS
            assert "test.com" in s.CORS_ORIGINS


class TestGetSettings:
    """Tests for cached settings retrieval."""
    
    def test_get_settings_returns_settings(self):
        """Test get_settings returns Settings instance."""
        result = get_settings()
        assert isinstance(result, Settings)
    
    def test_settings_singleton(self):
        """Test that settings module provides singleton."""
        from app.config import settings as s1
        from app.config import settings as s2
        assert s1 is s2
    
    def test_global_settings_accessible(self):
        """Test global settings instance is accessible."""
        assert settings is not None
        assert isinstance(settings, Settings)


class TestSettingsModelConfig:
    """Tests for Pydantic model configuration."""
    
    def test_env_file_encoding(self):
        """Test env file encoding is UTF-8."""
        s = Settings()
        assert s.model_config.get("env_file_encoding") == "utf-8"
    
    def test_env_file_name(self):
        """Test env file name is .env."""
        s = Settings()
        assert s.model_config.get("env_file") == ".env"


class TestModelSettings:
    """Tests for model-related settings."""
    
    def test_supported_model_names(self):
        """Test that model name is a valid option."""
        s = Settings()
        valid_models = ["resnet50", "resnet101", "efficientnet_b0"]
        assert s.MODEL_NAME in valid_models or s.MODEL_NAME == "resnet50"
    
    def test_model_weights_path_default(self):
        """Test default model weights path is empty."""
        s = Settings()
        assert s.MODEL_WEIGHTS_PATH == ""


class TestUploadSettings:
    """Tests for upload-related settings."""
    
    def test_max_file_size_reasonable(self):
        """Test max file size is within reasonable range."""
        s = Settings()
        # Should be between 1MB and 100MB
        assert 1024 * 1024 <= s.MAX_FILE_SIZE <= 100 * 1024 * 1024
    
    def test_max_batch_size_reasonable(self):
        """Test max batch size is within reasonable range."""
        s = Settings()
        # Should be between 1 and 100
        assert 1 <= s.MAX_BATCH_SIZE <= 100
    
    def test_all_image_extensions_allowed(self):
        """Test all common image extensions are allowed."""
        s = Settings()
        common_extensions = [".jpg", ".jpeg", ".png"]
        for ext in common_extensions:
            assert ext in s.ALLOWED_EXTENSIONS, f"Missing extension: {ext}"


class TestLoggingSettings:
    """Tests for logging-related settings."""
    
    def test_log_level_valid(self):
        """Test log level is a valid Python log level."""
        s = Settings()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert s.LOG_LEVEL.upper() in valid_levels
    
    def test_log_format_contains_placeholders(self):
        """Test log format contains expected placeholders."""
        s = Settings()
        assert "%(asctime)s" in s.LOG_FORMAT
        assert "%(levelname)s" in s.LOG_FORMAT
        assert "%(message)s" in s.LOG_FORMAT
