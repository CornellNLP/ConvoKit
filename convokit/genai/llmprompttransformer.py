from typing import Optional, Union, Callable, Dict, Any
from convokit import Transformer, Corpus, Conversation, Speaker, Utterance
from .factory import get_llm_client
from .genai_config import GenAIConfigManager


class LLMPromptTransformer(Transformer):
    """
    A ConvoKit Transformer that uses GenAI clients to process objects and store outputs as metadata.

    This transformer applies LLM prompts to different levels of the corpus (conversation, speaker, utterance, corpus)
    using a formatter function to prepare the object data for the prompt, and stores the LLM responses as metadata.

    :param provider: LLM provider name ("gpt", "gemini", "local", etc.)
    :param model: LLM model name
    :param object_level: Object level at which to apply the transformer ("conversation", "speaker", "utterance", "corpus")
    :param prompt: Template string for the prompt. Must contain '{formatted_object}' as a placeholder where the formatted object data will be inserted
    :param formatter: Function that takes an object and returns a string representation that will replace the '{formatted_object}' placeholder in the prompt
    :param metadata_name: Name of the metadata field to store the LLM response
    :param selector: Optional function to filter which objects to process. Defaults to processing all objects
    :param config_manager: GenAIConfigManager instance for LLM API key management
    :param llm_kwargs: Additional keyword arguments to pass to the LLM client
    """

    def __init__(
        self,
        provider: str,
        model: str,
        object_level: str,
        prompt: str,
        formatter: Callable[[Union[Corpus, Conversation, Speaker, Utterance]], str],
        metadata_name: str,
        selector: Optional[
            Callable[[Union[Corpus, Conversation, Speaker, Utterance]], bool]
        ] = None,
        config_manager: Optional[GenAIConfigManager] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.provider = provider
        self.model = model
        self.object_level = object_level
        self.prompt = prompt
        self.formatter = formatter
        self.metadata_name = metadata_name
        self.selector = selector or (lambda obj: True)
        self.config_manager = config_manager or GenAIConfigManager()
        self.llm_kwargs = llm_kwargs or {}

        if model is not None:
            self.llm_kwargs["model"] = model

        if object_level not in ["conversation", "speaker", "utterance", "corpus"]:
            raise ValueError(
                f"Invalid object_level: {object_level}. Must be one of: conversation, speaker, utterance, corpus"
            )

        if "{formatted_object}" not in prompt:
            raise ValueError(
                "Prompt must contain '{formatted_object}' placeholder for the formatted object data"
            )

        self.llm_client = get_llm_client(provider, self.config_manager, **self.llm_kwargs)

    def _format_prompt(self, obj: Union[Corpus, Conversation, Speaker, Utterance]) -> str:
        """
        Format the prompt with the object data using the formatter function.

        :param obj: Object to format
        :return: Formatted prompt string
        """
        try:
            formatted_object = self.formatter(obj)
            return self.prompt.format(formatted_object=formatted_object)
        except Exception as e:
            raise ValueError(f"Error formatting object for prompt: {e}")

    def _process_object(self, obj: Union[Corpus, Conversation, Speaker, Utterance]) -> None:
        """
        Process a single object with the LLM and store the result in metadata.

        :param obj: Object to process
        """
        try:
            formatted_prompt = self._format_prompt(obj)
            response = self.llm_client.generate(formatted_prompt)
            obj.add_meta(self.metadata_name, response.text)
        except Exception as e:
            print(f"Error processing {self.object_level} {obj.id}: {e}")
            obj.add_meta(self.metadata_name, None)

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Apply the GenAI transformer to the corpus.

        :param corpus: The corpus to transform
        :return: The transformed corpus with LLM responses added as metadata
        """
        if self.object_level == "utterance":
            for utterance in corpus.iter_utterances():
                if self.selector(utterance):
                    self._process_object(utterance)
                else:
                    utterance.add_meta(self.metadata_name, None)

        elif self.object_level == "conversation":
            for conversation in corpus.iter_conversations():
                if self.selector(conversation):
                    self._process_object(conversation)
                else:
                    conversation.add_meta(self.metadata_name, None)

        elif self.object_level == "speaker":
            for speaker in corpus.iter_speakers():
                if self.selector(speaker):
                    self._process_object(speaker)
                else:
                    speaker.add_meta(self.metadata_name, None)

        elif self.object_level == "corpus":
            if self.selector(corpus):
                self._process_object(corpus)
            else:
                corpus.add_meta(self.metadata_name, None)

        return corpus
