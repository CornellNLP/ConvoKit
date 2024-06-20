import warnings
from collections import defaultdict
from itertools import chain
from typing import Callable, List, Tuple, Dict, Any, Optional, Union, Set

import numpy as np
from IPython import get_ipython
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from convokit import Transformer
from convokit.model import Corpus, Utterance, CorpusComponent
from convokit.util import random_sampler
from .convokit_lm import ConvoKitLanguageModel

try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell" or shell == "TerminalInteractiveShell":
        from tqdm.notebook import tqdm
except (NameError, ModuleNotFoundError, ImportError):
    pass


class Surprise(Transformer):
    """Measures the amount of "surprise" between target and context utterance(s).

    This transformer computes how surprising a target utterance or group of utterances is, when
    compared to some context. The amount of "surprise" is measured by comparing the deviation
    of the target distribution from the context distribution (e.g., cross-entropy, perplexity).
    Furthermore, to mitigate the effects of text length on language model evaluation, the surprise
    transformer uses several random fixed length samples from target and context text.

    :param model_key_selector: A function that specifies how utterances are to be mapped to models.
        The function takes in an utterance and returns the key to use in mapping the utterance to a
        corresponding model.
    :param tokenizer: A function that returns a list of tokens in a given string, defaults to
        `nltk.word_tokenize`.
    :param surprise_attr_name: The name for the metadata attribute to add to the objects, defaults
        to "surprise".
    :param target_sample_size: The number of tokens to sample from each target (test text); when
        specified as `None`, then the entire target will be used, defaults to 100.
    :param context_sample_size: The number of tokens to sample from each context (training text);
        when specified as `None`, then the entire context will be used, defaults to 100.
    :param n_samples: The number of samples to take for each target-context pair, defaults to 50.
    :param sampling_fn: A function to generate samples of tokens, defaults to a random sampler.
    :param n_jobs: The number of concurrent threads to be used for routines that are parallelized
        with `joblib`, defaults to 1.
    """

    def __init__(
        self,
        model_key_selector: Callable[[Utterance], str],
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        surprise_attr_name: str = "surprise",
        target_sample_size: int = 100,
        context_sample_size: int = 100,
        n_samples: int = 50,
        sampling_fn: Callable[
            [List[Union[np.ndarray, List[str]]], int, int], np.ndarray
        ] = random_sampler,
        n_jobs: int = 1,
    ):
        self._model_key_selector = model_key_selector
        self._tokenizer = tokenizer
        self._surprise_attr_name = surprise_attr_name
        self._target_sample_size = target_sample_size
        self._context_sample_size = context_sample_size
        self._n_samples = n_samples
        self._sampling_fn = sampling_fn
        self._n_jobs = n_jobs
        self._model_groups = None

    def fit(
        self,
        corpus: Corpus,
        text_func: Callable[[Utterance], List[str]] = None,
        selector: Callable[[Utterance], bool] = lambda utt: True,
    ) -> Transformer:
        """Populate models for each group of utterances in a corpus.

        For each group of utterances in the corpus, a specific model is populated. The group that an
        utterance belongs to is determined by the `model_key_selector` parameter in the constructor.
        Furthermore, based on the `tokenizer` specified in the constructor, the text corresponding
        to the model key is tokenized.

        :param corpus: The corpus to create models from.
        :param text_func: The function used to define how the text a model is trained on should be
            selected. Takes an utterance as input and returns a list of strings to train the model
            corresponding to that utterance on. The model corresponding to the utterance is
            determined by the `model_key_selector` parameter specified in the constructor. For each
            utterance corresponding to the same model key, this function should return the same
            result.
            Defaults to `None`; when the value is `None`, a model will be trained on the text from
            all the utterances that belong to its group.
        :param selector: A function to specify which utterances in the corpus to train models for.
            Defaults to choosing all utterances, `lambda utt: True`.
        :return: An instance of the Surprise transformer with the corresponding models populated.
        """
        self._model_groups = defaultdict(list)

        for utt in tqdm(corpus.iter_utterances(selector=selector), desc="fit"):
            key = self._model_key_selector(utt)
            if text_func is not None:
                if key not in self._model_groups:
                    self._model_groups[key] = text_func(utt)
            else:
                self._model_groups[key].append(utt.text)

        for key in tqdm(self._model_groups, desc="fit"):
            if text_func is None:
                self._model_groups[key] = [" ".join(self._model_groups[key])]
            # Using `map()` with `lambda` is (microscopically) costlier than a list comprehension.
            # Reference: https://stackoverflow.com/a/1247490/6907625.
            self._model_groups[key] = [
                self._tokenizer(utt_text) for utt_text in self._model_groups[key]
            ]

        return self

    def _compute_surprise(
        self,
        target: List[str],
        context: List[List[str]],
        lm_evaluation_fn: Callable[
            [Union[List[str], np.ndarray], Union[List[str], np.ndarray], Optional[Any]],
            np.ndarray,
        ],
        **kwargs: Optional[Any],
    ) -> np.ndarray:
        """Compute the amount of "surprise" between target and context utterance(s).

        This method computes how surprising a target text is, when compared to some context. The
        amount of "surprise" is measured by comparing the deviation of the target distribution from
        the context distribution (e.g., cross-entropy, perplexity). Furthermore, to mitigate the
        effects of text length on language model evaluation, several random samples of fixed sizes
        are taken from the target and context.

        :param target: A list of tokens in the target.
        :param context: A list of lists of tokens in each group of the context.
        :param lm_evaluation_fn: The language model evaluation function. If using an instance of
            :py:class:`~convokit.LanguageModel`, the :py:meth:`~convokit.LanguageModel.evaluate`
            function is to be used here. To see examples of :py:class:`~convokit.LanguageModel`,
            see: :py:class:`~convokit.ConvoKitLanguageModel` and :py:class:`~convokit.Kenlm`. This
            function takes in a list of target samples and corresponding context samples, and
            returns the amount of surprise using some underlying language model evaluation metric.
        :param kwargs: Additional keyword arguments to be passed to the language model evaluation
            function:

            * When using :py:class:`~convokit.LanguageModel`, the following keywords are relevant:

                * `eval_type`: The language model evaluation metric, defaults to "cross_entropy".
                * The following arguments, if specified, overwrite the existing class values:

                    * `n_jobs`: The number of concurrent threads to be used for routines that are
                      parallelized with `joblib`, defaults to 1.
                    * `model_type`: The name of :py:class:`~convokit.LanguageModel`, defaults to
                      "language_model".

            * When using :py:class:`~convokit.ConvoKitLanguageModel`, the following keywords are
              relevant:

                * `eval_type`: The language model evaluation metric, defaults to "cross_entropy".
                * The following arguments, if specified, overwrite the existing class values:

                    * `smooth`: Indicator of using Laplace smoothing in the computation of surprise
                      scores, defaults to `True`.

                * The following arguments, inherited from :py:class:`~convokit.LanguageModel`, if
                  specified, overwrite the existing class values:

                    * `n_jobs`: The number of concurrent threads to be used for routines that are
                      parallelized with `joblib`, defaults to 1.
                    * `model_type`: The name of :py:class:`~convokit.LanguageModel`, defaults to
                      "convokit_lm".

            * When using :py:class:`~convokit.Kenlm`, the following keywords are relevant:

                * `eval_type`: The language model evaluation metric, defaults to "cross_entropy".
                * The following arguments, if specified, overwrite the existing class values:

                    * `ngram_order`: The order of n-gram language model.
                    * `trained_model_filepath`: The filepath to a pre-trained language model that is
                      to be persistently used.
                    * `is_persistent`: Indicator of model persistence, i.e., the model generated
                      in the first pass or that loaded from `trained_model_filepath` is used in all
                      evaluations. When `trained_model_filepath` is specified, persistence is
                      automatically implied.
                    * `kenlm_path`: The folder path to the folder of KenLM library.
                    * `models_dir`: The folder path to store the (trained) binary KenLM models.
                    * `model_filename`: The filename used in storing the KenLM model artefacts.

                * The following arguments, inherited from :py:class:`~convokit.LanguageModel`, if
                  specified, overwrite the existing class values:

                    * `n_jobs`: The number of concurrent threads to be used for routines that are
                      parallelized with `joblib`, defaults to 1.
                    * `model_type`: The name of :py:class:`~convokit.LanguageModel`, defaults to
                      "kenlm".
        :return: The surprise score output by the language model evaluation function.
        """
        target_tokens = np.array(target)
        context_tokens = [np.array(text) for text in context]
        target_samples = self._sampling_fn(
            [target_tokens], self._target_sample_size, self._n_samples
        )
        context_samples = self._sampling_fn(
            context_tokens, self._context_sample_size, self._n_samples
        )

        if target_samples is None or context_samples is None:
            return np.nan
        return lm_evaluation_fn(target_samples, context_samples, **kwargs)

    def _transform(
        self,
        corpus: Corpus,
        obj_type: str,
        group_and_models: Callable[[Utterance], Tuple[str, List[str]]] = None,
        target_text_func: Callable[[Utterance], List[str]] = None,
        selector: Callable[[CorpusComponent], bool] = lambda _: True,
        group_model_attr_key: Callable[[str, str], str] = None,
        **kwargs: Optional[Any],
    ) -> Corpus:
        """Annotates `obj_type` components in a corpus with surprise scores.

        The transform function adds surprise score metadata to the `obj_type` components in the
        given corpus.

        :param corpus: The corpus to compute surprise for.
        :param obj_type: The type of corpus components to annotate. Should be one of "utterance",
            "speaker", "conversation", or "corpus".
        :param group_and_models: A function that defines how an utterance should be grouped to form
            a target text and what models (contexts) the group should be compared to in calculating
            surprise scores. Takes in an utterance and returns a tuple containing the name of the
            group the utterance belongs to and a list of models to calculate how surprising that
            group is against. Objects will be annotated with a metadata field `surprise_attr_name`
            (specified in the constructor) that maps a key corresponding to the `group_name` and
            `model_key` to the surprise score for the utterances in the group when compared to the
            model. The key used is defined by the `group_model_attr_key` parameter.
            Defaults to `None`; if `group_and_models` is `None`, `model_key_selector` specified in
            the constructor will be used to select the group that an utterance belongs to. The
            surprise score will be calculated for each group of utterances compared to the model in
            `self.models` corresponding to the group.
        :param target_text_func: A function to define what the target text corresponding to an
            utterance should be; takes in an utterance and returns a list of string tokens.
            Defaults to `None`.
        :param selector: A function to specify which objects in the corpus to train models for,
            defaults to choosing all `obj_type` objects, `lambda _: True`.
        :param group_model_attr_key: A function that defines what key is to be used for a given
            `group_name` and `model_key`, defaults to `None`. If `group_model_attr_key` is `None`,
            the default key used will be "GROUP_group_name_MODEL_model_key" unless `group_name` and
            `model_key` are equal, in which case just "model_key" will be used as the key.
        :param kwargs: Additional keyword arguments to be passed for surprise computations (see
            the documentation for :py:meth:`~Surprise._compute_surprise()` for these arguments), and
            in creating the language model (if needed):

            * `language_model`: An instance of :py:class:`~convokit.LanguageModel` to be used in
              computing the surprise scores, defaults to :py:class:`~convokit.ConvoKitLanguageModel`
              and the arguments to the :py:class:`~convokit.ConvoKitLanguageModel` can be specified
              here as:

                * `smooth`: Indicator of using Laplace smoothing in the computation of surprise
                  scores, defaults to `True`.
                * `n_jobs`: The number of concurrent threads to be used for routines that are
                  parallelized with `joblib`, defaults to 1.
                * `model_type`: The name of :py:class:`~convokit.LanguageModel`, defaults to
                  "convokit_lm".
        :return: A modified version of the input corpus with the surprise scores.
        """

        def _update_groups_models(
            utt_: Utterance,
            utt_groups_: Dict[str, List[List[str]]],
            group_models_: Dict[str, Set[str]],
        ):
            """Updates the utterance groups and models based on `groups_and_models`.

            :param utt_: The utterance whose groups and models are to be populated (updated).
            :param utt_groups_: Update utterance groups based on `groups_and_models` parameter. The
                dictionary is modified in place.
            :param group_models_: Update utterance models based on `groups_and_models` parameter.
                The dictionary is modified in place.
            """
            group_name, models = (
                group_and_models(utt_)
                if group_and_models
                else (self._model_key_selector(utt_), None)
            )
            models = {group_name} if not models else models
            if target_text_func:
                if group_name not in utt_groups_:
                    utt_groups_[group_name] = [target_text_func(utt_)]
            else:
                utt_groups_[group_name].append(self._tokenizer(utt_.text))
            group_models_[group_name].update(models)

        def _format_attr_key(
            group_name: str, model_key: str, format_fn: Callable[[str, str], str] = None
        ) -> str:
            """Formats the surprise score attribute key, given model name and key.

            :param group_name: The group name to be included in the surprise score attribute key.
            :param model_key: The model key to be included in the surprise score attribute key.
            :param format_fn: A function that takes in the `group_name` and `model_key` and outputs
                the formatted attribute key, defaults to `None`. When `group_model_attr_key` is
                `None`, the default key used will be "GROUP_group_name_MODEL_model_key" unless
                `group_name` and `model_key` are equal, in which case just "model_key" will be used
                as the key.
            :return: The formatted surprise score attribute key.
            """
            if format_fn:
                return format_fn(group_name, model_key)
            if group_name == model_key:
                return model_key
            return f"GROUP_{group_name}__MODEL_{model_key}"

        def __surprise_score_helper(
            group_name: str,
            utt_group: List[List[str]],
            group_models_: Dict[str, Set[str]],
            surprise_scores_: Dict[str, np.ndarray],
            lm_evaluation_fn: Callable[
                [
                    Union[List[str], np.ndarray],
                    Union[List[str], np.ndarray],
                    Optional[Any],
                ],
                np.ndarray,
            ],
        ):
            """A helper function to aid in the computation of surprise scores.

            :param group_name: The group name corresponding to the group model to be used.
            :param utt_group: The utterance group from those populated using `groups_and_models`.
            :param group_models_: The group models that were populated using `groups_and_models`.
            :param surprise_scores_: The surprise score (dictionary value) that is to be updated for
                the corresponding utterance group and model. The dictionary is modified in place.
            :param lm_evaluation_fn: The language model evaluation function. If using an instance
                of :py:class:`~convokit.LanguageModel`, :py:meth:`~convokit.LanguageModel.evaluate`
                function is to be used here. To see examples of :py:class:`~convokit.LanguageModel`,
                see: :py:class:`~convokit.ConvoKitLanguageModel` and :py:class:`~convokit.Kenlm`.
                The function takes in a list of target samples and corresponding context samples,
                and returns the amount of surprise using some underlying model evaluation metric.
            """
            for model_key in group_models_[group_name]:
                assert model_key in self._model_groups, "invalid model key"
                surprise_key = _format_attr_key(group_name, model_key, group_model_attr_key)
                context = self._model_groups[model_key]
                target = list(chain(*utt_group))
                surprise_scores_[surprise_key] = self._compute_surprise(
                    target, context, lm_evaluation_fn, **kwargs
                )

        def _update_surprise_scores(
            utt_groups_: Dict[str, List[List[str]]],
            group_models_: Dict[str, Set[str]],
            surprise_scores_: Dict[str, np.ndarray],
            lm_evaluation_fn: Callable[
                [
                    Union[List[str], np.ndarray],
                    Union[List[str], np.ndarray],
                    Optional[Any],
                ],
                np.ndarray,
            ],
        ):
            """Populate (update) the surprise score for utterance groups and models.

            :param utt_groups_: The utterance groups that were populated using `groups_and_models`.
            :param group_models_: The group models that were populated using `groups_and_models`.
            :param surprise_scores_: The surprise scores (dictionary values) that are to be updated
                for the corresponding utterance groups and models. The surprise scores dictionary is
                modified in place.
            :param lm_evaluation_fn: The language model evaluation function. If using an instance
                of :py:class:`~convokit.LanguageModel`, the `evaluate` function is to be used here.
                To see the subclass implementations of :py:class:`~convokit.LanguageModel`, see:
                :py:class:`~convokit.ConvoKitLanguageModel` and :py:class:`~convokit.Kenlm`. The
                function takes in a list of target samples and corresponding context samples, and
                returns the amount of surprise using some underlying model evaluation metric.
            """
            if self._n_jobs == 1:
                for group_name in tqdm(utt_groups_, leave=False, desc="surprise", delay=2):
                    __surprise_score_helper(
                        group_name,
                        utt_groups_[group_name],
                        group_models_,
                        surprise_scores_,
                        lm_evaluation_fn,
                    )
            else:
                Parallel(n_jobs=self._n_jobs, backend="threading")(
                    delayed(__surprise_score_helper)(
                        group_name,
                        utt_groups_[group_name],
                        group_models_,
                        surprise_scores_,
                        lm_evaluation_fn,
                    )
                    for group_name in tqdm(utt_groups_, leave=False, desc="surprise", delay=2)
                )

        if "n_jobs" in kwargs and kwargs["n_jobs"] != self._n_jobs:
            warnings.warn(
                f"specified n_jobs={kwargs['n_jobs']}; however, the surprise transformer was "
                f"initialized with {self._n_jobs}, so defaulting to {self._n_jobs} jobs."
            )
            kwargs["n_jobs"] = self._n_jobs
        language_model = (
            kwargs["language_model"]
            if "language_model" in kwargs
            else ConvoKitLanguageModel(**kwargs)
        )

        if obj_type == "corpus":
            surprise_scores = defaultdict()
            utt_groups, group_models = defaultdict(list), defaultdict(set)
            for utt in tqdm(corpus.iter_utterances(), desc="transform"):
                _update_groups_models(utt, utt_groups, group_models)
            _update_surprise_scores(
                utt_groups, group_models, surprise_scores, language_model.evaluate
            )
            corpus.add_meta(self._surprise_attr_name, surprise_scores)
        elif obj_type == "utterance":
            for utt in tqdm(corpus.iter_utterances(selector=selector), desc="transform"):
                surprise_scores = defaultdict()
                utt_groups, group_models = defaultdict(list), defaultdict(set)
                _update_groups_models(utt, utt_groups, group_models)
                _update_surprise_scores(
                    utt_groups, group_models, surprise_scores, language_model.evaluate
                )
                utt.add_meta(self._surprise_attr_name, surprise_scores)
        else:
            for obj in tqdm(corpus.iter_objs(obj_type, selector=selector), desc="transform"):
                surprise_scores = defaultdict()
                utt_groups, group_models = defaultdict(list), defaultdict(set)
                for utt in obj.iter_utterances():
                    _update_groups_models(utt, utt_groups, group_models)
                _update_surprise_scores(
                    utt_groups, group_models, surprise_scores, language_model.evaluate
                )
                obj.add_meta(self._surprise_attr_name, surprise_scores)
        return corpus

    def transform(self, corpus: Corpus, **kwargs) -> Corpus:
        """A wrapper over :py:meth:`~convokit.Surprise._transform` of the Surprise transformer.

        Note: Since the transformer's :py:meth:`~convokit.Surprise.fit` method populates the model
        groups, the :py:meth:`~convokit.Surprise.transform` function is to be called after calling
        :py:meth:`~convokit.Surprise.fit`.

        :param corpus: The corpus to transform.
        :param kwargs: Any keyword arguments to be passed to :py:meth:`~convokit.Surprise.transform`
            function of the Surprise transformer (e.g., `eval_type`). Refer to the documentation of
            :py:meth:`~convokit.Surprise._transform()` for specific keyword arguments.
        :return: A modified version of the input corpus with the surprise scores.
        """
        return self._transform(corpus=corpus, **kwargs)
