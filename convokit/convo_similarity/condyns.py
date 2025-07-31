import ast
import numpy as np
import os
import re

try:
    from convokit.genai import get_llm_client

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class ConDynS:
    """A class to compute ConDynS score between conversations.

    ConDynS computes similarity scores between conversations by analyzing their
    Summary of Conversation Dynamics (SCD) patterns, which are extracted from the SCD
    as the Sequence of Patterns (SoP), and comparing them with conversation transcripts.
    The method uses bidirectional similarity computation to capture the full dynamics
    of both conversations.

    :param model_provider: The LLM provider to use (e.g., "gpt", "gemini")
    :param config: The GenAIConfigManager instance to use
    :param model: Optional specific model name
    :param custom_condyns_prompt: Custom text for the condyns prompt template
    :param custom_prompt_dir: Directory to save custom prompts (if not provided, overwrites default prompts in ./prompts)
    """

    CONDYNS_PROMPT_TEMPLATE = None

    @classmethod
    def _load_prompts(cls):
        """Lazy load prompts into class variables."""
        if cls.CONDYNS_PROMPT_TEMPLATE is None:
            base_path = os.path.dirname(__file__)
            with open(
                os.path.join(base_path, "prompts/condyns_prompt.txt"), "r", encoding="utf-8"
            ) as f:
                cls.CONDYNS_PROMPT_TEMPLATE = f.read()

    def __init__(
        self,
        model_provider: str,
        config,
        model: str = None,
        custom_condyns_prompt: str = None,
        custom_prompt_dir: str = None,
    ):
        """Initialize the ConDynS score calculator with a specified model provider and optional model name.

        If no model is specified, defaults to our selected default model.

        :param model_provider: The LLM provider to use (e.g., "gpt", "gemini")
        :param config: The GenAIConfigManager instance to use
        :param model: Optional specific model name
        :param custom_condyns_prompt: Custom text for the condyns prompt template
        :param custom_prompt_dir: Directory to save custom prompts (if not provided, overwrites defaults in ./prompts)
        :raises ImportError: If genai dependencies are not available
        """
        if not GENAI_AVAILABLE:
            raise ImportError("GenAI dependencies not available. Please install required packages.")

        self.model_provider = model_provider
        self.config = config
        self.model = model
        self.custom_prompt_dir = custom_prompt_dir

        # Load default prompts first
        self._load_prompts()

        # Override with custom prompts if provided
        if custom_condyns_prompt is not None:
            self.CONDYNS_PROMPT_TEMPLATE = custom_condyns_prompt
            if custom_prompt_dir:
                self._save_custom_prompt("condyns_prompt.txt", custom_condyns_prompt)
            else:
                self._save_custom_prompt_to_default("condyns_prompt.txt", custom_condyns_prompt)

        if model is not None:
            self.client = get_llm_client(model_provider, config, model=model)
        else:
            self.client = get_llm_client(model_provider, config)

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

    def set_custom_condyns_prompt(self, prompt_text: str, save_to_file: bool = True):
        """Set a custom condyns prompt template.

        :param prompt_text: The custom prompt text
        :param save_to_file: Whether to save the prompt to file in custom_prompt_dir or default prompts directory
        """
        self.CONDYNS_PROMPT_TEMPLATE = prompt_text
        if save_to_file:
            if self.custom_prompt_dir:
                self._save_custom_prompt("condyns_prompt.txt", prompt_text)
            else:
                self._save_custom_prompt_to_default("condyns_prompt.txt", prompt_text)

    def load_custom_prompts_from_directory(self, prompt_dir: str):
        """Load custom prompts from a specified directory.

        :param prompt_dir: Directory containing custom prompt files
        """
        condyns_path = os.path.join(prompt_dir, "condyns_prompt.txt")

        if os.path.exists(condyns_path):
            with open(condyns_path, "r", encoding="utf-8") as f:
                self.CONDYNS_PROMPT_TEMPLATE = f.read()

    def _clean_model_output_to_dict(self, text: str) -> dict:
        """Clean and parse model output into a dictionary.

        Extracts dictionary content from model responses and handles common
        formatting issues for safe parsing.

        :param text: Raw model output text
        :return: Parsed dictionary from the model output
        :raises ValueError: If no valid dictionary boundaries are found
        """
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No valid dictionary boundaries found.")

        dict_str = text[start : end + 1]
        dict_str = re.sub(r"'s\b", "s", dict_str)
        dict_str = re.sub(r"'t\b", "t", dict_str)
        dict_str = re.sub(r"'ve\b", "ve", dict_str)
        return ast.literal_eval(dict_str)

    def get_condyns_score(self, transcript1, transcript2, sop1, sop2):
        """Compute ConDynS score between two conversations.

        Computes ConDynS with the bidirectional similarity between two conversations using their
        transcripts and SoPs, then returns the mean score.

        :param transcript1: First conversation transcript
        :param transcript2: Second conversation transcript
        :param sop1: SoP for first conversation
        :param sop2: SoP for second conversation
        :return: ConDynS score
        """
        condyns_score = self.compute_bidirectional_similarity(transcript1, transcript2, sop1, sop2)
        return np.mean(self.compute_score_from_results(condyns_score))

    def compute_unidirectional_similarity(self, sop1, transcript2):
        """Compute unidirectional similarity between SoPs and a transcript.

        Analyzes how well the SoPs from one conversation match the dynamics
        observed in another conversation's transcript.

        :param sop1: Dictionary of SoPs with ordered keys ('0', '1', etc.) from the first conversation
        :param transcript2: Conversation transcript to analyze from the second conversation
        :return: Dictionary with analysis and scores for each event in sop1
        """
        # Format the prompt with the events and transcript
        full_prompt = self.CONDYNS_PROMPT_TEMPLATE.format(events=sop1, transcript=transcript2)

        response = self.client.generate(full_prompt)
        try:
            response_dict = self._clean_model_output_to_dict(response.text)
        except (SyntaxError, ValueError) as e:
            print(response.text)
            print("Error parsing output:", e)
            raise Exception("error parsing")
        return response_dict

    def compute_bidirectional_similarity(self, transcript1, transcript2, sop1, sop2):
        """Compute bidirectional similarity between two conversations.

        Computes similarity in both directions: SoP1 vs Transcript2 and SoP2 vs Transcript1
        to capture the full dynamics of both conversations.

        :param transcript1: First conversation transcript
        :param transcript2: Second conversation transcript
        :param sop1: SoP for first conversation
        :param sop2: SoP for second conversation
        :return: List of [response_dict1, response_dict2] where each dict contains
            analysis and scores for each event
        """
        response_dict1 = self.compute_unidirectional_similarity(sop1, transcript2)
        response_dict2 = self.compute_unidirectional_similarity(sop2, transcript1)
        return [response_dict1, response_dict2]

    def measure_score(self, data):
        """Calculate the mean score from a similarity result dictionary.

        :param data: Dictionary containing similarity analysis results
        :return: Mean score across all events
        """
        sum_score = []
        for item in data.values():
            sum_score.append(item["score"])
        return np.mean(sum_score)

    def compute_score_from_results(self, results):
        """Compute scores from bidirectional similarity results.

        :param results: List of bidirectional similarity results
        :return: List of mean scores for each direction
        """
        scores = []
        for result in results:
            scores.append(self.measure_score(result))
        return scores
