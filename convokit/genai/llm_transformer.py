from typing import Optional, Union, Callable, Dict, Any
from convokit import Transformer, Corpus, Conversation, Speaker, Utterance
from .factory import get_llm_client
from .genai_config import GenAIConfigManager


class LLM(Transformer):
    """
    A ConvoKit Transformer that uses LLM clients to process prompts and record outputs as metadata.

    This transformer can apply LLM prompts to different levels of the corpus (conversation, speaker, utterance, corpus)
    and store the LLM responses as metadata attributes.

    :param provider: LLM provider name ("gpt", "gemini", "local")
    :param model: LLM model name
    :param prompt_template: Template string for the prompt.
    :param output_field: Name of the metadata field to store the LLM response
    :param level: Object level at which to apply the transformer ("conversation", "speaker", "utterance", "corpus")
    :param config_manager: GenAIConfigManager instance for LLM API key management
    :param context_func: Optional function to extract context for the prompt. If None, uses default context extraction.
    :param llm_kwargs: Additional keyword arguments to pass to the LLM client
    :param input_filter: Optional function to filter which objects to process
    """

    def __init__(
        self,
        provider: str,
        model: str,
        prompt_template: str,
        output_field: str,
        level: str = "utterance",
        config_manager: Optional[GenAIConfigManager] = None,
        context_func: Optional[Callable] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        input_filter: Optional[Callable] = None,
    ):
        self.provider = provider
        self.model = model
        self.prompt_template = prompt_template
        self.output_field = output_field
        self.level = level
        self.config_manager = config_manager or GenAIConfigManager()
        self.context_func = context_func
        self.llm_kwargs = llm_kwargs or {}
        self.input_filter = input_filter

        if model is not None:
            self.llm_kwargs["model"] = model

        # Validate level
        if level not in ["conversation", "speaker", "utterance", "corpus"]:
            raise ValueError(
                f"Invalid level: {level}. Must be one of: conversation, speaker, utterance, corpus"
            )

        # Initialize LLM client
        self.llm_client = get_llm_client(provider, self.config_manager, **self.llm_kwargs)

    def _format_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format the prompt template with context variables.

        :param context: Dictionary of context variables
        :return: Formatted prompt string
        """
        try:
            return self.prompt_template.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing context variable in prompt template: {e}")

    def _should_process(self, obj) -> bool:
        """
        Check if the object should be processed based on the input filter.

        :param obj: Object to check
        :return: True if object should be processed
        """
        if self.input_filter is None:
            return True
        return self.input_filter(obj)

    def transform(self, corpus: Corpus, context_func: Optional[Callable] = None) -> Corpus:
        """
        Apply the LLM transformer to the corpus.

        :param corpus: The corpus to transform
        :return: The transformed corpus with LLM responses added as metadata
        """
        if self.level == "utterance":
            for utterance in corpus.iter_utterances():
                if self._should_process(utterance):
                    context = context_func(corpus, utterance)
                    prompt = self._format_prompt(context)

                    try:
                        response = self.llm_client.generate(prompt)
                        utterance.add_meta(self.output_field, response.text)
                    except Exception as e:
                        print(f"Error processing utterance {utterance.id}: {e}")
                        utterance.add_meta(self.output_field, None)

        elif self.level == "conversation":
            for conversation in corpus.iter_conversations():
                if self._should_process(conversation):
                    context = context_func(corpus, conversation)
                    prompt = self._format_prompt(context)

                    try:
                        response = self.llm_client.generate(prompt)
                        conversation.add_meta(self.output_field, response.text)
                    except Exception as e:
                        print(f"Error processing conversation {conversation.id}: {e}")
                        conversation.add_meta(self.output_field, None)

        elif self.level == "speaker":
            for speaker in corpus.iter_speakers():
                if self._should_process(speaker):
                    context = context_func(corpus, speaker)
                    prompt = self._format_prompt(context)

                    try:
                        response = self.llm_client.generate(prompt)
                        speaker.add_meta(self.output_field, response.text)
                    except Exception as e:
                        print(f"Error processing speaker {speaker.id}: {e}")
                        speaker.add_meta(self.output_field, None)

        elif self.level == "corpus":
            context = context_func(corpus, corpus)
            prompt = self._format_prompt(context)

            try:
                response = self.llm_client.generate(prompt)
                corpus.add_meta(self.output_field, response.text)
            except Exception as e:
                print(f"Error processing corpus: {e}")
                corpus.add_meta(self.output_field, None)

        return corpus
