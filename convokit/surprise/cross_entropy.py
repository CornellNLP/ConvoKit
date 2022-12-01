import multiprocessing
from collections import Counter

import numpy as np
from joblib import Parallel, delayed

from .perplexity import Perplexity


class CrossEntropy(Perplexity):
    """

    :param perplexity_type:
    :param kwargs:
    """

    def __init__(self, perplexity_type='convokit_cross_entropy', **kwargs):
        super().__init__(perplexity_type, **kwargs)

        self._smooth = kwargs['smooth'] if 'smooth' in kwargs else True
        self._n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else multiprocessing.cpu_count()

    @staticmethod
    def __cross_entropy(target, context, smooth):
        """

        :param target:
        :param context:
        :param smooth:
        :return:
        """
        n_target, n_context = len(target), len(context)
        if min(n_target, n_context) == 0:
            return np.nan

        context_counts = Counter(context)
        smooth_v = len(context_counts) + 1 if smooth else 0
        smooth_k = 1 if smooth else 0
        value = 0 if smooth else 1

        return sum(-np.log((context_counts.get(token, value) + smooth_k) / (n_context + smooth_v)) for token in
                   target) / n_target

    def perplexity_fn(self, target_samples, context_samples, **kwargs):
        """

        :param target_samples:
        :param context_samples:
        :param kwargs:
        :return:
        """
        self.overwrite_args(kwargs.keys(), kwargs)

        model_scores = Parallel(n_jobs=self._n_jobs, backend='threading')(
            delayed(self.__cross_entropy)(target_sample, context_sample, smooth=self._smooth) for
            target_sample, context_sample in zip(target_samples, context_samples))
        return np.nanmean(model_scores)
