from .base import LLMClient, LLMResponse
from .genai_config import GenAIConfigManager
import time


class LocalClient(LLMClient):
    """Template client for local LLM models. This is not a implemented client.

    This is a template implementation for local LLM clients. It provides a mock
    implementation that should be replaced with actual local model loading and inference.
    Currently returns mock responses for testing purposes.

    :param model_path: Path to the local model files (e.g., llama.cpp or GGUF model)
    :param config_manager: GenAIConfigManager instance (optional, will create one if not provided)
    """

    def __init__(self, model_path: str = "./", config_manager: GenAIConfigManager = None):
        if config_manager is None:
            config_manager = GenAIConfigManager()

        self.config_manager = config_manager
        self.model_path = model_path  # e.g., load a llama.cpp or GGUF-backed model

    def generate(self, messages, **kwargs) -> LLMResponse:
        """Generate text using the local model.

        Currently returns a mock response. This method should be implemented to
        actually load and run the local model for text generation.

        :param messages: Input messages for generation
        :param **kwargs: Additional generation parameters
        :return: LLMResponse object containing the generated text and metadata
        """
        start = time.time()
        prompt = " ".join(m["content"] for m in messages)
        response = f"[Mock local model output for: {prompt}]"
        latency = time.time() - start
        return LLMResponse(text=response, tokens=-1, latency=latency, raw={"prompt": prompt})
