from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from .base import LLMClient, LLMResponse
from .genai_config import GenAIConfigManager
import time


class GeminiClient(LLMClient):
    """Client for interacting with Google Gemini models.

    This client uses the Gemini Developer API with an API key by default. If no API key
    is configured, or if ``use_vertex_ai`` is set, it falls back to Vertex AI with Google
    Cloud project and location configuration.

    :param model: Name of the Gemini model to use
    :param config_manager: GenAIConfigManager instance (optional, will create one if not provided)
    :param use_vertex_ai: Force Vertex AI authentication instead of API key authentication
    """

    def __init__(
        self,
        model: str,
        config_manager: GenAIConfigManager = None,
        use_vertex_ai: bool = False,
    ):
        if config_manager is None:
            config_manager = GenAIConfigManager()

        self.config_manager = config_manager
        self.model = model

        api_key = config_manager.get_api_key("gemini")
        if api_key and not use_vertex_ai:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = self._create_vertex_ai_client(config_manager)

    @staticmethod
    def _create_vertex_ai_client(config_manager: GenAIConfigManager):
        """Create a Gemini client authenticated through Vertex AI."""
        google_cloud_project = config_manager.get_google_cloud_project()
        google_cloud_location = config_manager.get_google_cloud_location()

        if not google_cloud_project:
            raise ValueError(
                "Gemini API key or Google Cloud project is required. "
                "Set an API key using config_manager.set_api_key('gemini', 'your-key'), "
                "or via GEMINI_API_KEY. For Vertex AI, set project and location using "
                "config_manager.set_google_cloud_config(project, location) or via "
                "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION."
            )

        if not google_cloud_location:
            raise ValueError(
                "Google Cloud location is required for Vertex AI. "
                "Set it using config_manager.set_google_cloud_config(project, location) "
                "or via GOOGLE_CLOUD_LOCATION environment variable."
            )

        return genai.Client(
            vertexai=True,
            project=google_cloud_project,
            location=google_cloud_location,
            http_options=HttpOptions(api_version="v1"),
        )

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
