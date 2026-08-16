"""Configuration for LLM XAI providers."""

import os


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""
    pass


def get_llm_config() -> dict:
    """Load LLM configuration from environment variables.

    Returns:
        dict: A dictionary containing 'provider', 'model', and 'api_key'.

    Raises:
        ConfigError: If the API key is missing or the provider is unknown.
    """
    provider = os.getenv("XAI_PROVIDER", "gemini").lower()
    
    # Defaults
    if provider == "gemini":
        model = os.getenv("XAI_MODEL", "gemini-3.6-flash")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigError("GEMINI_API_KEY environment variable is required for Gemini provider.")
    elif provider == "openai":
        model = os.getenv("XAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
    elif provider == "anthropic":
        model = os.getenv("XAI_MODEL", "claude-3-5-haiku-20241022")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider.")
    else:
        raise ConfigError(f"Unknown XAI_PROVIDER: {provider}")

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
    }
