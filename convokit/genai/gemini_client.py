import os
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from .base import LLMClient, LLMResponse
from .genai_config import GenAIConfigManager
import time


class GeminiClient(LLMClient):
    """Client for interacting with Google Gemini models via Vertex AI.

    This client is configured to use Vertex AI and requires Google Cloud project and location
    to be set. Configuration can be provided via the GenAI config system or environment variables.

    :param model: Name of the Gemini model to use (default: "gemini-2.0-flash-001")
    :param config_manager: GenAIConfigManager instance (optional, will create one if not provided)
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-001",
        config_manager: GenAIConfigManager = None,
    ):
        if config_manager is None:
            config_manager = GenAIConfigManager()

        self.config_manager = config_manager

        # Get required Vertex AI configuration
        google_cloud_project = config_manager.get_google_cloud_project()
        google_cloud_location = config_manager.get_google_cloud_location()

        # Validate required fields
        if not google_cloud_project:
            raise ValueError(
                "Google Cloud project is required for Vertex AI. "
                "Set it using config_manager.set_google_cloud_config(project, location) "
                "or via GOOGLE_CLOUD_PROJECT environment variable."
            )

        if not google_cloud_location:
            raise ValueError(
                "Google Cloud location is required for Vertex AI. "
                "Set it using config_manager.set_google_cloud_config(project, location) "
                "or via GOOGLE_CLOUD_LOCATION environment variable."
            )

        # Set up Vertex AI environment
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = google_cloud_project
        os.environ["GOOGLE_CLOUD_LOCATION"] = google_cloud_location

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
