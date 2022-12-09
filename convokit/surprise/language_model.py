from abc import ABC
from typing import Optional, Any, List, Dict, Union

import numpy as np
from joblib import Parallel, delayed


class LanguageModel(ABC):
    """

    :param model_type:
    :param kwargs:
    """

    def __init__(self, model_type: str = 'language_model', **kwargs: Optional[Any]):
        self._model_type = model_type
        self._n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else 1

        self.__dict__.update((f'_{arg}', value) for arg, value in kwargs.items())

    @property
    def type(self) -> str:
        """

        :return:
        """
        return self._model_type

    @property
    def config(self) -> Dict[str, Any]:
        """

        :return:
        """
        private_var_prefix = f'_{self.__class__.__name__}'
        return {arg[1:]: value for arg, value in self.__dict__.items() if not arg.startswith(private_var_prefix)}

    def _overwrite_args(self, args_to_overwrite: List[str], kwargs: Dict[str, Any]):
        """

        :param args_to_overwrite:
        :param kwargs:
        :return:
        """
        for arg in args_to_overwrite:
            self.__dict__[f'_{arg}'] = kwargs[arg] if arg in kwargs else self.__dict__[f'_{arg}']

    def cross_entropy(self, target: Union[List[str], np.ndarray], context: Union[List[str], np.ndarray]) -> float:
        """

        :param target:
        :param context:
        :return:
        """
        raise RuntimeError('cross entropy is not implemented')

    def evaluate(self, target_samples: Union[List[List[str]], np.ndarray],
                 context_samples: Union[List[List[str]], np.ndarray], eval_type: str = 'cross_entropy',
                 **kwargs: Optional[Any]) -> np.ndarray:
        """

        :param target_samples:
        :param context_samples:
        :param eval_type:
        :param kwargs:
        :return:
        """
        self._overwrite_args(list(kwargs.keys()), kwargs)
        eval_fn = getattr(self, eval_type)

        if self._n_jobs == 1:
            model_scores = [eval_fn(target_sample, context_sample) for target_sample, context_sample in
                            zip(target_samples, context_samples)]
        else:
            model_scores = Parallel(n_jobs=self._n_jobs, backend='threading')(
                delayed(eval_fn)(target_sample, context_sample) for target_sample, context_sample in
                zip(target_samples, context_samples))
        return np.nanmean(model_scores)
