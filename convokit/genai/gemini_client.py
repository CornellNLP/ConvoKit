import os
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from .base import LLMClient, LLMResponse
import time


class GeminiClient(LLMClient):
    """Client for interacting with Google Gemini models.

    Provides an interface to generate text using Google's Gemini models through their API.
    Supports both Vertex AI and direct API access, with configurable project and location settings.

    :param api_key: Google API key for authentication
    :param model: Name of the Gemini model to use (default: "gemini-2.0-flash-001")
    :param google_cloud_project: Google Cloud project ID (optional)
    :param google_cloud_location: Google Cloud location (optional)
    :param use_vertex_ai: Whether to use Vertex AI (default: True)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-001",
        google_cloud_project: str = None,
        google_cloud_location: str = None,
        use_vertex_ai: bool = True,
    ):
        if not use_vertex_ai:
            os.environ["GOOGLE_API_KEY"] = api_key
        if use_vertex_ai:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        if google_cloud_location:
            os.environ["GOOGLE_CLOUD_PROJECT"] = google_cloud_location
        if google_cloud_project:
            os.environ["GOOGLE_CLOUD_LOCATION"] = google_cloud_project
        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
        self.model = model

    def generate(self, prompt, temperature=0.0, times_retried=0) -> LLMResponse:
        """Generate text using the Gemini model.

        Sends a prompt to the Gemini model and returns the generated response. The function includes
        retry logic for API errors and handles different input formats.

        :param prompt: Input prompt for generation
        :param temperature: Sampling temperature for generation (default: 0.0)
        :param times_retried: Number of retry attempts made so far (for internal use)
        :return: LLMResponse object containing the generated text and metadata
        :raises Exception: If retry attempts are exhausted
        """
        start = time.time()
        retry_after = 10

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=GenerateContentConfig(temperature=temperature),
            )
        except Exception as e:
            if times_retried >= 3:
                raise Exception("Retry failed after multiple attempts.") from e
            print(f"Gemini Exception: {e}. Retrying in {retry_after}s...")
            time.sleep(retry_after)
            return self.generate(prompt, temperature, times_retried + 1)

        elapsed = time.time() - start
        text = response.text
        # Gemini does not currently provide token usage reliably
        return LLMResponse(text=text, tokens=-1, latency=elapsed, raw=response)
