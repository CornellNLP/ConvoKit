from openai import OpenAI, OpenAIError, RateLimitError, Timeout
from .base import LLMClient, LLMResponse
import time


class GPTClient(LLMClient):
    """Client for interacting with OpenAI GPT models.

    Provides an interface to generate text using OpenAI's GPT models through their API.
    Handles authentication, request formatting, and error retry logic.

    :param api_key: OpenAI API key for authentication
    :param model: Name of the GPT model to use (default: "gpt-4o-mini")
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self, prompt, output_max_tokens=512, temperature=0.0, times_retried=0
    ) -> LLMResponse:
        """Generate text using the GPT model.

        Sends a prompt to the GPT model and returns the generated response. Handles
        different input formats (string or message list) and includes retry logic for
        API errors.

        :param prompt: Input prompt for generation. Can be a string or list of message dicts
        :param output_max_tokens: Maximum number of tokens to generate (default: 512)
        :param temperature: Sampling temperature for generation (default: 0.0)
        :param times_retried: Number of retry attempts made so far (for internal use)
        :return: LLMResponse object containing the generated text and metadata
        :raises Exception: If output error and retry attempts are exhausted
        """
        start = time.time()
        retry_after = 10

        # Check prompt type to determine how to format messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
                messages = prompt
            else:
                raise ValueError(
                    "Invalid message format: each message must be a dict with 'role' and 'content'"
                )
        else:
            raise TypeError("Prompt must be either a string or a list of message dicts")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=output_max_tokens,
                temperature=temperature,
            )
        except (OpenAIError, RateLimitError, Timeout) as e:
            if times_retried >= 3:
                raise Exception("Retry failed after multiple attempts.") from e
            print(f"{type(e).__name__}: {e}. Retrying in {retry_after}s...")
            time.sleep(retry_after)
            return self.generate(prompt, output_max_tokens, temperature, times_retried + 1)

        elapsed = time.time() - start
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else -1
        return LLMResponse(text=content, tokens=tokens_used, latency=elapsed, raw=response)
