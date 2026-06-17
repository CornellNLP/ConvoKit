from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union, Callable, Dict, Any, Iterable
from convokit import Transformer, Corpus, Conversation, Speaker, Utterance
from tqdm.auto import tqdm
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
    :param num_workers: Number of worker threads to use for LLM calls. Defaults to 1 (sequential)
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
        num_workers: int = 1,
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
        self.num_workers = num_workers

        if model is not None:
            self.llm_kwargs["model"] = model

        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")

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

    def _transform_object(self, obj: Union[Corpus, Conversation, Speaker, Utterance]) -> None:
        """
        Apply selector logic and process a single object.

        :param obj: Object to transform
        """
        if self.selector(obj):
            self._process_object(obj)
        else:
            obj.add_meta(self.metadata_name, None)

    def _transform_objects(
        self,
        objects: Iterable[Union[Conversation, Speaker, Utterance]],
        desc: str,
    ) -> None:
        """
        Transform a collection of objects, optionally in parallel.

        :param objects: Objects to transform
        :param desc: Progress bar description
        """
        if self.num_workers == 1:
            for obj in tqdm(objects, desc=desc):
                self._transform_object(obj)
            return

        objects = list(objects)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._transform_object, obj) for obj in objects]
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                future.result()

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Apply the GenAI transformer to the corpus.

        :param corpus: The corpus to transform
        :return: The transformed corpus with LLM responses added as metadata
        """
        if self.object_level == "utterance":
            self._transform_objects(corpus.iter_utterances(), "Applying LLM prompt to utterances")

        elif self.object_level == "conversation":
            self._transform_objects(
                corpus.iter_conversations(), "Applying LLM prompt to conversations"
            )

        elif self.object_level == "speaker":
            self._transform_objects(corpus.iter_speakers(), "Applying LLM prompt to speakers")

        elif self.object_level == "corpus":
            if self.selector(corpus):
                self._process_object(corpus)
            else:
                corpus.add_meta(self.metadata_name, None)

        return corpus

    def transform_single(
        self, obj: Union[str, Corpus, Conversation, Speaker, Utterance]
    ) -> Union[Corpus, Conversation, Speaker, Utterance]:
        """
        Transform a single object (utterance, conversation, speaker, or corpus) with the LLM prompt.
        This method allows users to easily test their prompt on a single unit without processing an entire corpus.

        :param obj: The object to transform. Can be:
            - A string (will be converted to an Utterance with a default speaker)
            - An Utterance, Conversation, or Speaker object
        :return: The transformed object with LLM response stored in metadata
        """
        # Handle string input by converting to Utterance
        if isinstance(obj, str):
            if self.object_level != "utterance":
                raise ValueError(
                    f"Cannot convert string to {self.object_level}. String input is only supported for utterance-level transformation."
                )
            obj = Utterance(text=obj, speaker=Speaker(id="speaker"))

        # Validate object type matches the transformer's object_level
        if self.object_level == "utterance" and not isinstance(obj, Utterance):
            raise ValueError(
                f"Expected Utterance object for utterance-level transformation, got {type(obj).__name__}"
            )
        elif self.object_level == "conversation" and not isinstance(obj, Conversation):
            raise ValueError(
                f"Expected Conversation object for conversation-level transformation, got {type(obj).__name__}"
            )
        elif self.object_level == "speaker" and not isinstance(obj, Speaker):
            raise ValueError(
                f"Expected Speaker object for speaker-level transformation, got {type(obj).__name__}"
            )

        # Check if object passes the selector
        if not self.selector(obj):
            obj.add_meta(self.metadata_name, None)
            return obj

        # Process the object
        self._process_object(obj)
        return obj
