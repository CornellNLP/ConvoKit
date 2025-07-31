from abc import ABC, abstractmethod
import time


class LLMResponse:
    """Encapsulates the response from an LLM client.

    Contains the generated text, token usage information, latency, and raw response data.

    :param text: The generated text response from the LLM
    :param tokens: Number of tokens used in the generation (may be -1 if not available from the LLM output)
    :param latency: Time taken for generation in seconds
    :param raw: Raw response object from the LLM
    """

    def __init__(self, text: str, tokens: int, latency: float, raw: dict):
        self.text = text
        self.tokens = tokens
        self.latency = latency
        self.raw = raw

    def __repr__(self):
        return f"LLMResponse(text={self.text[:30]}..., tokens={self.tokens}, latency={self.latency:.2f}s)"


class LLMClient(ABC):
    """Abstract base class for LLM clients. Used as a template for all LLM clients.

    Provides a common interface for different LLM providers (GPT, Gemini, local models, etc.).
    All LLM clients should inherit from this class and implement the required methods.
    """

    @abstractmethod
    def generate(self, messages, **kwargs) -> LLMResponse:
        """Generate text from the LLM.

        :param messages: Input messages/prompt for the LLM. Can be a string or list of message dicts
        :param **kwargs: Additional parameters for generation (temperature, max_tokens, etc.)
        :return: LLMResponse object containing the generated text and metadata
        """
        pass

    def stream(self, messages, callback, **kwargs):
        """Stream text generation from the LLM. Notice that this is not supported yet.

        :param messages: Input messages/prompt for the LLM
        :param callback: Function to call with each generated token/chunk
        :param **kwargs: Additional parameters for generation
        :raises NotImplementedError: If streaming is not supported by this client
        """
        raise NotImplementedError("Streaming not supported yet.")
