import os
import ast
import re

try:
    from convokit.genai import get_llm_client

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class SCDWriter:
    """A writer class to generate Summary of Conversation Dynamics (SCD) and
    Sequence of Patterns (SoP) using GenAI models.

    SCDWriter provides functionality to generate Summary of Conversation Dynamics (SCD)
    from conversation transcripts and extract Sequence of Patterns (SoP) from SCD using LLM models.
    We provide a default prompt for the SCD and SoP generation, but users can also provide their
    own prompts for the SCD and SoP generation on their own data.

    :param model_provider: The GenAI provider to use (e.g., "gpt", "gemini")
    :param config: The GenAIConfigManager instance to use
    :param model: Optional specific model name
    :param custom_summary_prompt: Custom text for the summary prompt template
    :param custom_bulletpoint_prompt: Custom text for the bulletpoint prompt template
    :param custom_prompt_dir: Directory to save custom prompts (if not provided, overwrites defaults in ./prompts)
    """

    SUMMARY_PROMPT_TEMPLATE = None
    BULLETPOINT_PROMPT_TEMPLATE = None

    @classmethod
    def _load_prompts(cls):
        """Lazy load prompts into class variables.

        Loads the SCD and SoP prompt templates from the prompts directory if not already loaded.
        """
        if cls.SUMMARY_PROMPT_TEMPLATE is None or cls.BULLETPOINT_PROMPT_TEMPLATE is None:
            base_path = os.path.dirname(__file__)
            with open(
                os.path.join(base_path, "prompts/scd_prompt.txt"), "r", encoding="utf-8"
            ) as f:
                cls.SUMMARY_PROMPT_TEMPLATE = f.read()
            with open(
                os.path.join(base_path, "prompts/sop_prompt.txt"), "r", encoding="utf-8"
            ) as f:
                cls.BULLETPOINT_PROMPT_TEMPLATE = f.read()

    def __init__(
        self,
        model_provider: str,
        config,
        model: str = None,
        custom_scd_prompt: str = None,
        custom_sop_prompt: str = None,
        custom_prompt_dir: str = None,
    ):
        """Initialize the SCD processor with a specified model provider and optional model name.

        If no model is specified, defaults to our selected default model.

        :param model_provider: The GenAI provider to use (e.g., "gpt", "gemini")
        :param config: The GenAIConfigManager instance to use
        :param model: Optional specific model name
        :param custom_scd_prompt: Custom text for the summary prompt template
        :param custom_sop_prompt: Custom text for the bulletpoint prompt template
        :param custom_prompt_dir: Directory to save custom prompts (if not provided, overwrites defaults in ./prompts)
        :raises ImportError: If genai dependencies are not available
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "GenAI dependencies not available. Please install required packages: pip install openai google-genai"
            )

        self.model_provider = model_provider
        self.config = config
        self.model = model
        self.custom_prompt_dir = custom_prompt_dir

        # Load default prompts first
        self._load_prompts()

        # Override with custom prompts if provided
        if custom_scd_prompt is not None:
            self.SUMMARY_PROMPT_TEMPLATE = custom_scd_prompt
            if custom_prompt_dir:
                self._save_custom_prompt("scd_prompt.txt", custom_scd_prompt)
            else:
                self._save_custom_prompt_to_default("scd_prompt.txt", custom_scd_prompt)

        if custom_sop_prompt is not None:
            self.BULLETPOINT_PROMPT_TEMPLATE = custom_sop_prompt
            if custom_prompt_dir:
                self._save_custom_prompt("sop_prompt.txt", custom_sop_prompt)
            else:
                self._save_custom_prompt_to_default("sop_prompt.txt", custom_sop_prompt)

        self.client = get_llm_client(model_provider, config, model=model)

    def _save_custom_prompt(self, filename: str, prompt_content: str):
        """Save custom prompt to the specified directory.

        :param filename: Name of the file to save
        :param prompt_content: Content of the prompt to save
        """
        if self.custom_prompt_dir:
            os.makedirs(self.custom_prompt_dir, exist_ok=True)
            filepath = os.path.join(self.custom_prompt_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(prompt_content)

    def _save_custom_prompt_to_default(self, filename: str, prompt_content: str):
        """Save custom prompt to the default prompts directory.

        :param filename: Name of the file to save
        :param prompt_content: Content of the prompt to save
        """
        base_path = os.path.dirname(__file__)
        filepath = os.path.join(base_path, "prompts", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt_content)

    def set_custom_scd_prompt(self, prompt_text: str, save_to_file: bool = True):
        """Set a custom SCD prompt template.

        :param prompt_text: The custom prompt text
        :param save_to_file: Whether to save the prompt to file in custom_prompt_dir or default prompts directory
        """
        self.SUMMARY_PROMPT_TEMPLATE = prompt_text
        if save_to_file:
            if self.custom_prompt_dir:
                self._save_custom_prompt("scd_prompt.txt", prompt_text)
            else:
                self._save_custom_prompt_to_default("scd_prompt.txt", prompt_text)

    def set_custom_sop_prompt(self, prompt_text: str, save_to_file: bool = True):
        """Set a custom SoP prompt template.

        :param prompt_text: The custom prompt text
        :param save_to_file: Whether to save the prompt to file in custom_prompt_dir or default prompts directory
        """
        self.BULLETPOINT_PROMPT_TEMPLATE = prompt_text
        if save_to_file:
            if self.custom_prompt_dir:
                self._save_custom_prompt("sop_prompt.txt", prompt_text)
            else:
                self._save_custom_prompt_to_default("sop_prompt.txt", prompt_text)

    def load_custom_prompts_from_directory(self, prompt_dir: str):
        """Load custom prompts from a specified directory.

        :param prompt_dir: Directory containing custom prompt files
        """
        scd_path = os.path.join(prompt_dir, "scd_prompt.txt")
        sop_path = os.path.join(prompt_dir, "sop_prompt.txt")

        if os.path.exists(scd_path):
            with open(scd_path, "r", encoding="utf-8") as f:
                self.SUMMARY_PROMPT_TEMPLATE = f.read()

        if os.path.exists(sop_path):
            with open(sop_path, "r", encoding="utf-8") as f:
                self.BULLETPOINT_PROMPT_TEMPLATE = f.read()

    def _convert_to_single_quote_dict_string(self, response: str) -> str:
        """Clean and normalize model output for safe ast.literal_eval parsing.

        Handles common formatting issues in LLM responses to ensure they can be
        safely parsed as Python dictionaries.

        :param response: Raw response text from LLM
        :return: Cleaned string suitable for ast.literal_eval
        """
        response = response.strip()

        # Normalize possessives and contractions
        response = re.sub(r"'s\b", "s", response)
        response = re.sub(r"'t\b", "t", response)
        response = re.sub(r"'ve\b", "ve", response)

        # Ensure entries are comma-separated
        response = re.sub(r'(")(\s*\'\d+\'\s*:)', r"\1,\2", response)

        # Replace double quotes around values with single quotes
        response = re.sub(
            r'(?<=:\s)"(.*?)"',
            lambda m: "'" + m.group(1).replace("'", "\\'") + "'",
            response,
        )

        return response

    def get_scd_summary(self, transcript: str) -> str:
        """Generate the SCD summary from a transcript.

        Uses the LLM to generate a Summary of Conversation Dynamics (SCD) from
        a conversation transcript.

        :param transcript: Conversation transcript to summarize
        :return: SCD summary text
        """
        query = self.SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
        response = self.client.generate(query)
        return response.text

    def get_sop_from_summary(self, summary: str) -> dict:
        """Generate SoP bulletpoints from an SCD summary.

        Uses the LLM to generate Sequence of Patterns (SoP) bulletpoints from
        an SCD summary.

        :param summary: SCD summary text to convert to SoP
        :return: Dictionary containing SoP patterns
        :raises Exception: If the response cannot be parsed as a dictionary
        """
        query = self.BULLETPOINT_PROMPT_TEMPLATE + summary
        response = self.client.generate(query)

        parsed_text = response.text.strip()

        try:
            return ast.literal_eval(parsed_text)
        except Exception:
            # Fallback: clean and retry
            cleaned_text = self._convert_to_single_quote_dict_string(parsed_text)
            try:
                return ast.literal_eval(cleaned_text)
            except Exception:
                print("output (cleaned): " + cleaned_text)
                raise Exception("The response was not formatted properly for literal_eval parsing")

    def get_scd_and_sop(self, transcript: str) -> tuple[str, dict]:
        """Generate both the SCD summary and the SoP patterns in one call.

        Convenience method that generates both SCD summary and SoP patterns
        from a transcript in a single operation.

        :param transcript: Conversation transcript to process
        :return: Tuple of (SCD summary, SoP patterns)
        """
        scd = self.get_scd_summary(transcript)
        sop = self.get_sop_from_summary(scd)
        return scd, sop
