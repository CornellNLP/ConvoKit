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
from .convokit_lm import ConvoKitLanguageModel
from .utils import random_sampler

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell' or shell == 'TerminalInteractiveShell':
        from tqdm.notebook import tqdm
except NameError:
    pass


class Surprise(Transformer):
    """

    :param model_key_selector:
    :param tokenizer:
    :param surprise_attr_name:
    :param target_sample_size:
    :param context_sample_size:
    :param n_samples:
    :param sampling_fn:
    :param n_jobs:
    """

    def __init__(self, model_key_selector: Callable[[Utterance], str],
                 tokenizer: Callable[[str], List[str]] = word_tokenize, surprise_attr_name: str = 'surprise',
                 target_sample_size: int = 100, context_sample_size: int = 100, n_samples: int = 50,
                 sampling_fn: Callable[[List[Union[np.ndarray, List[str]]], int, int], np.ndarray] = random_sampler,
                 n_jobs: int = 1):
        self._model_key_selector = model_key_selector
        self._tokenizer = tokenizer
        self._surprise_attr_name = surprise_attr_name
        self._target_sample_size = target_sample_size
        self._context_sample_size = context_sample_size
        self._n_samples = n_samples
        self._sampling_fn = sampling_fn
        self._n_jobs = n_jobs
        self._model_groups = None

    def fit(self, corpus: Corpus, text_func: Callable[[Utterance], List[str]] = None,
            selector: Callable[[Utterance], bool] = lambda utt: True) -> Transformer:
        """

        :param corpus:
        :param text_func:
        :param selector:
        :return:
        """
        self._model_groups = defaultdict(list)

        for utt in tqdm(corpus.iter_utterances(selector=selector), desc='fit'):
            key = self._model_key_selector(utt)
            if text_func is not None:
                if key not in self._model_groups:
                    self._model_groups[key] = text_func(utt)
            else:
                self._model_groups[key].append(utt.text)

        for key in tqdm(self._model_groups, desc='fit'):
            if text_func is None:
                self._model_groups[key] = [' '.join(self._model_groups[key])]
            # Using `map()` with a `lambda` function is (microscopically) costlier than a list comprehension.
            # Reference: https://stackoverflow.com/a/1247490/6907625.
            self._model_groups[key] = [self._tokenizer(utt_text) for utt_text in self._model_groups[key]]

        return self

    def _compute_surprise(self, target: List[str], context: List[List[str]],
                          lm_evaluation_fn: Callable[[Union[List[str], np.ndarray], Union[List[str], np.ndarray],
                                                      Optional[Any]], np.ndarray],
                          **kwargs: Optional[Any]) -> np.ndarray:
        """

        :param target:
        :param context:
        :param lm_evaluation_fn:
        :param kwargs:
        :return:
        """
        target_tokens = np.array(target)
        context_tokens = [np.array(text) for text in context]
        target_samples = self._sampling_fn([target_tokens], self._target_sample_size, self._n_samples)
        context_samples = self._sampling_fn(context_tokens, self._context_sample_size, self._n_samples)

        if target_samples is None or context_samples is None:
            return np.nan
        return lm_evaluation_fn(target_samples, context_samples, **kwargs)

    def _transform(self, corpus: Corpus, obj_type: str,
                   group_and_models: Callable[[Utterance], Tuple[str, List[str]]] = None,
                   target_text_func: Callable[[Utterance], List[str]] = None,
                   selector: Callable[[CorpusComponent], bool] = lambda _: True,
                   group_model_attr_key: Callable[[str, str], str] = None, **kwargs: Optional[Any]) -> Corpus:
        """

        :param corpus:
        :param obj_type:
        :param group_and_models:
        :param target_text_func:
        :param selector:
        :param group_model_attr_key:
        :param kwargs:
        :return:
        """

        def _update_groups_models(utt_: Utterance, utt_groups_: Dict[str, List[List[str]]],
                                  group_models_: Dict[str, Set[str]]):
            """

            :param utt_:
            :param utt_groups_:
            :param group_models_:
            :return:
            """
            group_name, models = group_and_models(utt_) if group_and_models else (self._model_key_selector(utt_), None)
            models = {group_name} if not models else models
            if target_text_func:
                if group_name not in utt_groups_:
                    utt_groups_[group_name] = [target_text_func(utt_)]
            else:
                utt_groups_[group_name].append(self._tokenizer(utt_.text))
            group_models_[group_name].update(models)

        def _format_attr_key(group_name: str, model_key: str, format_fn: Callable[[str, str], str] = None) -> str:
            """

            :param group_name:
            :param model_key:
            :param format_fn:
            :return:
            """
            if format_fn:
                return format_fn(group_name, model_key)
            if group_name == model_key:
                return model_key
            return f'GROUP_{group_name}__MODEL_{model_key}'

        def __surprise_score_helper(group_name: str, utt_group: List[List[str]], group_models_: Dict[str, Set[str]],
                                    surprise_scores_: Dict[str, np.ndarray],
                                    lm_evaluation_fn: Callable[
                                        [Union[List[str], np.ndarray], Union[List[str], np.ndarray],
                                         Optional[Any]], np.ndarray]):
            """

            :param group_name:
            :param utt_group:
            :param group_models_:
            :param surprise_scores_:
            :param lm_evaluation_fn:
            :return:
            """
            for model_key in group_models_[group_name]:
                assert model_key in self._model_groups, 'invalid model key'
                surprise_key = _format_attr_key(group_name, model_key, group_model_attr_key)
                context = self._model_groups[model_key]
                target = list(chain(*utt_group))
                surprise_scores_[surprise_key] = self._compute_surprise(target, context, lm_evaluation_fn, **kwargs)

        def _update_surprise_scores(utt_groups_: Dict[str, List[List[str]]], group_models_: Dict[str, Set[str]],
                                    surprise_scores_: Dict[str, np.ndarray],
                                    lm_evaluation_fn: Callable[
                                        [Union[List[str], np.ndarray], Union[List[str], np.ndarray],
                                         Optional[Any]], np.ndarray]):
            """

            :param utt_groups_:
            :param group_models_:
            :param surprise_scores_:
            :param lm_evaluation_fn:
            :return:
            """
            if self._n_jobs == 1:
                for group_name in tqdm(utt_groups_, leave=False, desc='surprise', delay=2):
                    __surprise_score_helper(group_name, utt_groups_[group_name], group_models_, surprise_scores_,
                                            lm_evaluation_fn)
            else:
                Parallel(n_jobs=self._n_jobs, backend='threading')(
                    delayed(__surprise_score_helper)(group_name, utt_groups_[group_name], group_models_,
                                                     surprise_scores_, lm_evaluation_fn) for group_name in
                    tqdm(utt_groups_, leave=False, desc='surprise', delay=2))

        language_model = kwargs['language_model'] if 'language_model' in kwargs else ConvoKitLanguageModel(
            n_jobs=self._n_jobs, **kwargs)

        if obj_type == 'corpus':
            surprise_scores = {}
            utt_groups, group_models = defaultdict(list), defaultdict(set)
            for utt in tqdm(corpus.iter_utterances(), desc='transform'):
                _update_groups_models(utt, utt_groups, group_models)
            _update_surprise_scores(utt_groups, group_models, surprise_scores, language_model.evaluate)
            corpus.add_meta(self._surprise_attr_name, surprise_scores)
        elif obj_type == 'utterance':
            for utt in tqdm(corpus.iter_utterances(selector=selector), desc='transform'):
                surprise_scores = {}
                utt_groups, group_models = defaultdict(list), defaultdict(set)
                _update_groups_models(utt, utt_groups, group_models)
                _update_surprise_scores(utt_groups, group_models, surprise_scores, language_model.evaluate)
                utt.add_meta(self._surprise_attr_name, surprise_scores)
        else:
            for obj in tqdm(corpus.iter_objs(obj_type, selector=selector), desc='transform'):
                surprise_scores = {}
                utt_groups, group_models = defaultdict(list), defaultdict(set)
                for utt in obj.iter_utterances():
                    _update_groups_models(utt, utt_groups, group_models)
                _update_surprise_scores(utt_groups, group_models, surprise_scores, language_model.evaluate)
                obj.add_meta(self._surprise_attr_name, surprise_scores)
        return corpus

    def transform(self, corpus: Corpus, **kwargs) -> Corpus:
        """

        :param corpus:
        :param kwargs:
        :return:
        """
        return self._transform(corpus=corpus, **kwargs)
