from typing import Callable, Optional, Union, Any, List
from convokit.transformer import Transformer
from convokit.model import Corpus, Conversation
from .summary import SCDWriter


class SCD(Transformer):
    """
    A ConvoKit Transformer that generates Summary of Conversation Dynamics (SCD) and
    Sequence of Patterns (SoP) for conversations in a corpus through a LLM.

    This transformer takes a corpus and generates SCD and/or SoP for selected conversations,
    storing the results as metadata on the conversations.

    :param model_provider: The LLM provider to use (e.g., "gpt", "gemini")
    :param config: The LLMConfigManager instance to use for LLM configuration
    :param model: Optional specific model name
    :param custom_scd_prompt: Custom text for the SCD prompt template
    :param custom_sop_prompt: Custom text for the SoP prompt template
    :param custom_prompt_dir: Directory to save custom prompts
    :param generate_scd: Whether to generate SCD summaries (default: True)
    :param generate_sop: Whether to generate SoP patterns (default: True)
    :param scd_metadata_name: Name for the SCD metadata field (default: "machine_scd")
    :param sop_metadata_name: Name for the SoP metadata field (default: "machine_sop")
    :param conversation_formatter: Optional function to format conversations for processing.
        Should take a Conversation object and return a string. If None, uses default formatting.
    """

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
    ):
        self.model_provider = model_provider
        self.config = config
        self.model = model
        self.custom_scd_prompt = custom_scd_prompt
        self.custom_sop_prompt = custom_sop_prompt
        self.custom_prompt_dir = custom_prompt_dir
        self.generate_scd = generate_scd
        self.generate_sop = generate_sop
        self.scd_metadata_name = scd_metadata_name
        self.sop_metadata_name = sop_metadata_name
        self.conversation_formatter = conversation_formatter

        # Initialize the SCDWriter
        self.scd_writer = SCDWriter(
            model_provider=model_provider,
            config=config,
            model=model,
            custom_scd_prompt=custom_scd_prompt,
            custom_sop_prompt=custom_sop_prompt,
            custom_prompt_dir=custom_prompt_dir,
        )

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
        # Get the conversation formatter
        formatter = self.conversation_formatter or self._default_conversation_formatter

        # Process selected conversations
        for conversation in corpus.iter_conversations(selector):
            # Format the conversation
            transcript = formatter(conversation)

            # Generate SCD and/or SoP
            if self.generate_scd and self.generate_sop:
                scd, sop = self.scd_writer.get_scd_and_sop(transcript)
                conversation.add_meta(self.scd_metadata_name, scd)
                conversation.add_meta(self.sop_metadata_name, sop)
            elif self.generate_scd:
                scd = self.scd_writer.get_scd_summary(transcript)
                conversation.add_meta(self.scd_metadata_name, scd)
            elif self.generate_sop:
                # For SoP, we need to generate SCD first, then extract SoP
                scd = self.scd_writer.get_scd_summary(transcript)
                sop = self.scd_writer.get_sop_from_summary(scd)
                conversation.add_meta(self.sop_metadata_name, sop)

        return corpus
