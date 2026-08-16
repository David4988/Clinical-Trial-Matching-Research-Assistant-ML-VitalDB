"""Tests for LLM configuration."""

import os
from unittest import mock
import pytest

from vitaldb_audit.llm.config import get_llm_config, ConfigError

def test_get_llm_config_gemini_default():
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True):
        config = get_llm_config()
        assert config["provider"] == "gemini"
        assert config["model"] == "gemini-3.6-flash"
        assert config["api_key"] == "test_key"

def test_get_llm_config_gemini_missing_key():
    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ConfigError, match="GEMINI_API_KEY environment variable is required"):
            get_llm_config()

def test_get_llm_config_openai():
    with mock.patch.dict(os.environ, {"XAI_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}, clear=True):
        config = get_llm_config()
        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4o-mini"
        assert config["api_key"] == "sk-test"

def test_get_llm_config_anthropic():
    with mock.patch.dict(os.environ, {"XAI_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "ant-test"}, clear=True):
        config = get_llm_config()
        assert config["provider"] == "anthropic"
        assert config["model"] == "claude-3-5-haiku-20241022"
        assert config["api_key"] == "ant-test"

def test_get_llm_config_unknown_provider():
    with mock.patch.dict(os.environ, {"XAI_PROVIDER": "unknown"}, clear=True):
        with pytest.raises(ConfigError, match="Unknown XAI_PROVIDER"):
            get_llm_config()
