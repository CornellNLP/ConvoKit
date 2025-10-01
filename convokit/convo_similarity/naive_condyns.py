import ast
import numpy as np
import os
import re

try:
    from convokit.genai import get_llm_client

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class NaiveConDynS:
    """A class to compute naive ConDynS score between two Sequences of Patterns.

    NaiveConDynS computes similarity scores between conversations by directly
    comparing their Sequences of Patterns (SoP) without
    using conversation transcripts. This provides a simpler approach to measuring
    conversation dynamics similarity.

    :param model_provider: The LLM provider to use (e.g., "gpt", "gemini")
    :param config: The GenAIConfigManager instance to use
    :param model: Optional specific model name
    :param custom_naive_condyns_prompt: Custom text for the naive condyns prompt template
    :param custom_prompt_dir: Directory to save custom prompts (if not provided, overwrites defaults in ./prompts)
    """

    NAIVE_CONDYNS_PROMPT_TEMPLATE = None

    @classmethod
    def _load_prompts(cls):
        """Lazy load prompts into class variables.

        Loads the NaiveConDynS prompt template from the prompts directory if not already loaded.
        """
        if cls.NAIVE_CONDYNS_PROMPT_TEMPLATE is None:
            base_path = os.path.dirname(__file__)
            with open(
                os.path.join(base_path, "prompts/naive_condyns_prompt.txt"), "r", encoding="utf-8"
            ) as f:
                cls.NAIVE_CONDYNS_PROMPT_TEMPLATE = f.read()

    def __init__(
        self,
        model_provider: str,
        config,
        model: str = None,
        custom_naive_condyns_prompt: str = None,
        custom_prompt_dir: str = None,
    ):
        """Initialize the NaiveConDynS score computer with a specified model provider and optional model name.

        If no model is specified, defaults to our selected default model.

        :param model_provider: The LLM provider to use (e.g., "gpt", "gemini")
        :param config: The GenAIConfigManager instance to use
        :param model: Optional specific model name
        :param custom_naive_condyns_prompt: Custom text for the naive condyns prompt template
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
        if custom_naive_condyns_prompt is not None:
            self.NAIVE_CONDYNS_PROMPT_TEMPLATE = custom_naive_condyns_prompt
            if custom_prompt_dir:
                self._save_custom_prompt("naive_condyns_prompt.txt", custom_naive_condyns_prompt)
            else:
                self._save_custom_prompt_to_default(
                    "naive_condyns_prompt.txt", custom_naive_condyns_prompt
                )

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

    def set_custom_naive_condyns_prompt(self, prompt_text: str, save_to_file: bool = True):
        """Set a custom naive condyns prompt template.

        :param prompt_text: The custom prompt text
        :param save_to_file: Whether to save the prompt to file in custom_prompt_dir or default prompts directory
        """
        self.NAIVE_CONDYNS_PROMPT_TEMPLATE = prompt_text
        if save_to_file:
            if self.custom_prompt_dir:
                self._save_custom_prompt("naive_condyns_prompt.txt", prompt_text)
            else:
                self._save_custom_prompt_to_default("naive_condyns_prompt.txt", prompt_text)

    def load_custom_prompts_from_directory(self, prompt_dir: str):
        """Load custom prompts from a specified directory.

        :param prompt_dir: Directory containing custom prompt files
        """
        naive_condyns_path = os.path.join(prompt_dir, "naive_condyns_prompt.txt")

        if os.path.exists(naive_condyns_path):
            with open(naive_condyns_path, "r", encoding="utf-8") as f:
                self.NAIVE_CONDYNS_PROMPT_TEMPLATE = f.read()

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

    def compute_unidirectional_naive_condyns(self, sop1, sop2):
        """Compute unidirectional naive conditional dynamics similarity between two Sequences of Patterns.

        Compares the SoPs from one conversation against another to measure how well
        the dynamics of one conversation match those of another.

        :param sop1: Dictionary of patterns from the first conversation with ordered keys ('0', '1', etc.)
        :param sop2: Dictionary of patterns from the second conversation with ordered keys ('0', '1', etc.)
        :return: Dictionary with analysis and scores for each pattern in sop1
        """
        # Format the prompt with the two sequences of patterns
        full_prompt = self.NAIVE_CONDYNS_PROMPT_TEMPLATE.format(sop1=sop1, sop2=sop2)

        response = self.client.generate(full_prompt)
        try:
            response_dict = self._clean_model_output_to_dict(response.text)
        except (SyntaxError, ValueError) as e:
            print(response.text)
            print("Error parsing output:", e)
            raise Exception("error parsing")
        return response_dict

    def compute_bidirectional_naive_condyns(self, sop1, sop2):
        """Compute bidirectional naive conditional dynamics similarity between two Sequences of Patterns.

        Computes similarity in both directions: sop1 vs sop2 and sop2 vs sop1
        to capture the full dynamics of both conversations.

        :param sop1: Dictionary of patterns from the first conversation with ordered keys ('0', '1', etc.)
        :param sop2: Dictionary of patterns from the second conversation with ordered keys ('0', '1', etc.)
        :return: List of [response_dict1, response_dict2] where each dict contains
            analysis and scores for each pattern
        """
        response_dict1 = self.compute_unidirectional_naive_condyns(sop1, sop2)
        response_dict2 = self.compute_unidirectional_naive_condyns(sop2, sop1)
        return [response_dict1, response_dict2]

    def measure_score(self, data):
        """Calculate the mean score from a similarity result dictionary.

        :param data: Dictionary containing similarity analysis results
        :return: Mean score across all patterns
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

    def compare_conversations(self, corpus, convo_id1: str, convo_id2: str, sop_meta_name: str):
        """Compare two conversations using NaiveConDynS and store the result in both conversations' metadata.

        This method retrieves two conversations from the corpus, extracts their SoP data
        from metadata, computes the NaiveConDynS score between them, and stores the result in both
        conversations' metadata with the key format "condyns_{convo_id1}_{convo_id2}".

        Note: NaiveConDynS only uses SoP data for comparison, not conversation transcripts.

        :param corpus: The ConvoKit Corpus containing the conversations
        :param convo_id1: ID of the first conversation
        :param convo_id2: ID of the second conversation  
        :param sop_meta_name: Name of the metadata field containing SoP data
        :return: The computed NaiveConDynS score
        :raises KeyError: If conversations don't exist or required metadata is missing
        :raises ValueError: If SoP data is malformed
        """
        # Get conversations from corpus
        try:
            convo1 = corpus.get_conversation(convo_id1)
            convo2 = corpus.get_conversation(convo_id2)
        except KeyError as e:
            raise KeyError(f"Conversation not found in corpus: {e}")

        # Extract SoP data from metadata
        try:
            sop1 = convo1.meta[sop_meta_name]
            sop2 = convo2.meta[sop_meta_name]
        except KeyError as e:
            raise KeyError(f"SoP metadata '{sop_meta_name}' not found in conversation: {e}")

        # Validate that SoP data is properly formatted (should be dict with ordered keys)
        if not isinstance(sop1, dict) or not isinstance(sop2, dict):
            raise ValueError("SoP data must be dictionaries")

        # Compute bidirectional NaiveConDynS similarity
        results = self.compute_bidirectional_naive_condyns(sop1, sop2)
        
        # Compute the mean score from bidirectional results
        naive_condyns_score = np.mean(self.compute_score_from_results(results))

        # Store the score in both conversations' metadata
        score_key1 = f"condyns_{convo_id1}_{convo_id2}"
        score_key2 = f"condyns_{convo_id2}_{convo_id1}"
        
        convo1.meta[score_key1] = naive_condyns_score
        convo2.meta[score_key2] = naive_condyns_score

        return naive_condyns_score
