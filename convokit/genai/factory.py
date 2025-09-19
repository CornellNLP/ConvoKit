GPTClient = None
GeminiClient = None
LocalClient = None

try:
    from .gpt_client import GPTClient
except ImportError:
    pass

try:
    from .gemini_client import GeminiClient
except ImportError:
    pass

try:
    from .local_client import LocalClient
except ImportError:
    pass


def get_llm_client(provider: str, config_manager, **kwargs):
    """Factory function as a unified interface to create LLM client instances.

    Creates and returns the appropriate LLM client based on the provider name.
    The client is initialized with the config manager and any additional parameters.

    :param provider: Name of the LLM provider ("gpt", "gemini", "local")
    :param config_manager: Configuration manager instance to pass to the client
    :param **kwargs: Additional parameters to pass to the client constructor
    :return: Initialized LLM client instance
    :raises ValueError: If the provider is not supported or dependencies are missing
    """
    if provider.lower() == "gpt":
        if GPTClient is None:
            raise ValueError("GPT client not available. Please install the 'openai' package.")
        return GPTClient(config_manager=config_manager, **kwargs)
    elif provider.lower() == "gemini":
        if GeminiClient is None:
            raise ValueError(
                "Gemini client not available. Please install the 'google-genai' package."
            )
        return GeminiClient(config_manager=config_manager, **kwargs)
    elif provider.lower() == "local":
        if LocalClient is None:
            raise ValueError(
                "Local client not available. Please install required dependencies for local model support."
            )
        return LocalClient(config_manager=config_manager, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
