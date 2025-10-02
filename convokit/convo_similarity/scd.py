import os
import ast
import re
from typing import Callable, Optional, Union, Any, List
from convokit.transformer import Transformer
from convokit.model import Corpus, Conversation

try:
    from convokit.genai import GenAITransformer
    from convokit.genai.genai_config import GenAIConfigManager

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class SCD(Transformer):
    """
    A ConvoKit Transformer that generates Summary of Conversation Dynamics (SCD) and
    Sequence of Patterns (SoP) for conversations in a corpus through a LLM.

    This transformer takes a corpus and generates SCD and/or SoP for selected conversations,
    storing the results as metadata on the conversations.

    Prompt Templates:
    - SCD prompt: Uses {formatted_output} placeholder for the conversation transcript
    - SoP prompt: Uses {formatted_output} placeholder for the SCD summary

    :param model_provider: The LLM provider to use (e.g., "gpt", "gemini")
    :param config: The GenAIConfigManager instance to use for LLM configuration
    :param model: Optional specific model name
    :param custom_scd_prompt: Custom text for the SCD prompt template. Should include {formatted_output}
        placeholder for the conversation transcript.
    :param custom_sop_prompt: Custom text for the SoP prompt template. Should include {formatted_output}
        placeholder for the SCD summary.
    :param custom_prompt_dir: Directory to save custom prompts
    :param generate_scd: Whether to generate SCD summaries (default: True)
    :param generate_sop: Whether to generate SoP patterns (default: True)
    :param scd_metadata_name: Name for the SCD metadata field (default: "machine_scd")
    :param sop_metadata_name: Name for the SoP metadata field (default: "machine_sop")
    :param conversation_formatter: Optional function to format conversations for processing.
        Should take a Conversation object and return a string. If None, uses default formatting.
    :param llm_kwargs: Additional keyword arguments to pass to the LLM client
    """

    # Class variables for lazy loading of prompts
    SUMMARY_PROMPT_TEMPLATE = None
    BULLETPOINT_PROMPT_TEMPLATE = None

    @classmethod
    def _load_prompts(cls):
        """Lazy load prompts into class variables."""
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
        generate_scd: bool = True,
        generate_sop: bool = True,
        scd_metadata_name: str = "machine_scd",
        sop_metadata_name: str = "machine_sop",
        conversation_formatter: Optional[Callable[[Conversation], str]] = None,
        llm_kwargs: Optional[dict] = None,
    ):
        if not GENAI_AVAILABLE:
            raise ImportError("GenAI dependencies not available. Please install required packages.")

        self.model_provider = model_provider
        self.config = config
        self.model = model
        self.custom_prompt_dir = custom_prompt_dir
        self.generate_scd = generate_scd
        self.generate_sop = generate_sop
        self.scd_metadata_name = scd_metadata_name
        self.sop_metadata_name = sop_metadata_name
        self.conversation_formatter = conversation_formatter
        self.llm_kwargs = llm_kwargs or {}

        # Load default prompts
        self._load_prompts()

        # Set up prompts (use custom if provided)
        self.scd_prompt = custom_scd_prompt or self.SUMMARY_PROMPT_TEMPLATE
        self.sop_prompt = custom_sop_prompt or self.BULLETPOINT_PROMPT_TEMPLATE

        # Save custom prompts if provided
        if custom_scd_prompt is not None:
            self._save_custom_prompt("scd_prompt.txt", custom_scd_prompt)
        if custom_sop_prompt is not None:
            self._save_custom_prompt("sop_prompt.txt", custom_sop_prompt)

    def _save_custom_prompt(self, filename: str, prompt_content: str):
        """Save custom prompt to the specified directory."""
        if self.custom_prompt_dir:
            os.makedirs(self.custom_prompt_dir, exist_ok=True)
            filepath = os.path.join(self.custom_prompt_dir, filename)
        else:
            base_path = os.path.dirname(__file__)
            filepath = os.path.join(base_path, "prompts", filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt_content)

    def _default_conversation_formatter(self, conversation: Conversation) -> str:
        """
        Default conversation formatter that creates a transcript from conversation utterances.

        :param conversation: The conversation to format
        :return: Formatted transcript string
        """
        utterances = conversation.get_chronological_utterance_list()
        transcript_parts = []

        for utt in utterances:
            speaker_name = f"Speaker_{utt.speaker.id}"
            transcript_parts.append(f"{speaker_name}: {utt.text}")

        return "\n".join(transcript_parts)

    def set_custom_scd_prompt(self, prompt_text: str, save_to_file: bool = True):
        """Set a custom SCD prompt template."""
        self.scd_prompt = prompt_text
        if save_to_file:
            self._save_custom_prompt("scd_prompt.txt", prompt_text)

    def set_custom_sop_prompt(self, prompt_text: str, save_to_file: bool = True):
        """Set a custom SoP prompt template."""
        self.sop_prompt = prompt_text
        if save_to_file:
            self._save_custom_prompt("sop_prompt.txt", prompt_text)

    def load_custom_prompts_from_directory(self, prompt_dir: str):
        """Load custom prompts from a specified directory."""
        scd_path = os.path.join(prompt_dir, "scd_prompt.txt")
        sop_path = os.path.join(prompt_dir, "sop_prompt.txt")

        if os.path.exists(scd_path):
            with open(scd_path, "r", encoding="utf-8") as f:
                self.scd_prompt = f.read()

        if os.path.exists(sop_path):
            with open(sop_path, "r", encoding="utf-8") as f:
                self.sop_prompt = f.read()

    def transform(
        self, corpus: Corpus, selector: Callable[[Conversation], bool] = lambda x: True
    ) -> Corpus:
        """
        Transform the corpus by generating SCD and/or SoP for selected conversations.

        :param corpus: The target corpus
        :param selector: A function that takes a Conversation object and returns True/False
            to determine which conversations to process. By default, processes all conversations.
        :return: The modified corpus with SCD/SoP metadata added to conversations
        """
        if self.generate_scd:
            formatter = self.conversation_formatter or self._default_conversation_formatter
            scd_transformer = GenAITransformer(
                provider=self.model_provider,
                model=self.model,
                object_level="conversation",
                prompt=self.scd_prompt,
                formatter=formatter,
                metadata_name=self.scd_metadata_name,
                selector=selector,
                config_manager=self.config,
                llm_kwargs=self.llm_kwargs,
            )
            scd_transformer.transform(corpus)

        if self.generate_sop:
            # Formatter that gets the SCD from conversation metadata
            def scd_formatter(conversation):
                if self.scd_metadata_name not in conversation.meta:
                    raise ValueError(f"SCD not found for conversation {conversation.id}")
                return conversation.meta.get(self.scd_metadata_name, "")

            sop_transformer = GenAITransformer(
                provider=self.model_provider,
                model=self.model,
                object_level="conversation",
                prompt=self.sop_prompt,
                formatter=scd_formatter,
                metadata_name=self.sop_metadata_name,
                selector=selector,
                config_manager=self.config,
                llm_kwargs=self.llm_kwargs,
            )
            sop_transformer.transform(corpus)

        return corpus
