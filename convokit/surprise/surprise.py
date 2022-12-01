import multiprocessing
from collections import defaultdict
from itertools import chain

import numpy as np
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from convokit import Transformer
from convokit.model import Corpus
from .cross_entropy import CrossEntropy
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

    def __init__(self, model_key_selector, tokenizer=word_tokenize, surprise_attr_name='surprise',
                 target_sample_size=100, context_sample_size=100, n_samples=50, sampling_fn=random_sampler,
                 n_jobs=None):
        self._model_key_selector = model_key_selector
        self._tokenizer = tokenizer
        self._surprise_attr_name = surprise_attr_name
        self._target_sample_size = target_sample_size
        self._context_sample_size = context_sample_size
        self._n_samples = n_samples
        self._sampling_fn = sampling_fn
        self._n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
        self._model_groups = None

    def fit(self, corpus: Corpus, text_func=None, selector=lambda utt: True):
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

    def _compute_surprise(self, target, context, perplexity_fn, **kwargs):
        """

        :param target:
        :param context:
        :param perplexity_fn:
        :param kwargs:
        :return:
        """
        target_tokens = np.array(target)
        context_tokens = [np.array(text) for text in context]
        target_samples = self._sampling_fn([target_tokens], self._target_sample_size, self._n_samples)
        context_samples = self._sampling_fn(context_tokens, self._context_sample_size, self._n_samples)

        if target_samples is None or context_samples is None:
            return np.nan
        return perplexity_fn(target_samples, context_samples, **kwargs)

    def _transform(self, corpus, obj_type, group_and_models=None, target_text_func=None, selector=lambda _: True,
                   group_model_attr_key=None, **kwargs):
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

        def _update_groups_models(utt_, utt_groups_, group_models_):
            """

            :param utt_:
            :param utt_groups_:
            :param group_models_:
            :return:
            """
            group_name, models = group_and_models(utt_) if group_and_models else self._model_key_selector(utt_), None
            models = {group_name} if models is None else models
            if target_text_func:
                if group_name not in utt_groups_:
                    utt_groups_[group_name] = [target_text_func(utt_)]
            else:
                utt_groups_[group_name].append(self._tokenizer(utt_.text))
            group_models_[group_name].update(models)

        def _format_attr_key(group_name, model_key, format_fn=None):
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

        def __surprise_score_helper(group_name, utt_group, group_models_, surprise_scores_, perplexity_fn):
            """

            :param group_name:
            :param utt_group:
            :param group_models_:
            :param surprise_scores_:
            :param perplexity_fn:
            :return:
            """
            for model_key in group_models_[group_name]:
                assert model_key in self._model_groups, 'invalid model key'
                surprise_key = _format_attr_key(group_name, model_key, group_model_attr_key)
                context = self._model_groups[model_key]
                target = list(chain(*utt_group))
                surprise_scores_[surprise_key] = self._compute_surprise(target, context, perplexity_fn, **kwargs)

        def _get_surprise_scores(utt_groups_, group_models_, surprise_scores_, perplexity_fn):
            """

            :param utt_groups_:
            :param group_models_:
            :param surprise_scores_:
            :param perplexity_fn:
            :return:
            """
            Parallel(n_jobs=self._n_jobs, backend='threading')(
                delayed(__surprise_score_helper)(group_name, utt_groups_[group_name], group_models_, surprise_scores_,
                                                 perplexity_fn) for group_name in
                tqdm(utt_groups_, leave=False, desc='surprise'))

        surprise_scores = {}
        perplexity = kwargs['perplexity'] if 'perplexity' in kwargs else CrossEntropy(**kwargs)

        if obj_type == 'corpus':
            utt_groups, group_models = defaultdict(list), defaultdict(set)
            for utt in tqdm(corpus.iter_utterances(), desc='transform'):
                _update_groups_models(utt, utt_groups, group_models)
            _get_surprise_scores(utt_groups, group_models, surprise_scores, perplexity.perplexity_fn)
            corpus.add_meta(self._surprise_attr_name, surprise_scores)
        elif obj_type == 'utterance':
            for utt in tqdm(corpus.iter_utterances(selector=selector), desc='transform'):
                utt_groups, group_models = defaultdict(list), defaultdict(set)
                _update_groups_models(utt, utt_groups, group_models)
                _get_surprise_scores(utt_groups, group_models, surprise_scores, perplexity.perplexity_fn)
                utt.add_meta(self._surprise_attr_name, surprise_scores)
        else:
            for obj in tqdm(corpus.iter_objs(obj_type, selector=selector), desc='transform'):
                utt_groups, group_models = defaultdict(list), defaultdict(set)
                for utt in obj.iter_utterances():
                    _update_groups_models(utt, utt_groups, group_models)
                _get_surprise_scores(utt_groups, group_models, surprise_scores, perplexity.perplexity_fn)
                obj.add_meta(self._surprise_attr_name, surprise_scores)
        return corpus

    def transform(self, corpus: Corpus, **kwargs) -> Corpus:
        """

        :param corpus:
        :param kwargs:
        :return:
        """
        return self._transform(corpus=corpus, **kwargs)
