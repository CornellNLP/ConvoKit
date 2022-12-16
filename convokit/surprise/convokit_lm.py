from collections import Counter
from typing import Optional, Any, Union, List

import numpy as np

from .language_model import LanguageModel


class ConvoKitLanguageModel(LanguageModel):
    """

    :param model_type:
    :param kwargs:
    """

    def __init__(self, model_type: str = "convokit_lm", **kwargs: Optional[Any]):
        super().__init__(model_type, **kwargs)

        self._smooth = kwargs["smooth"] if "smooth" in kwargs else True

    def cross_entropy(
        self,
        target: Union[List[str], np.ndarray],
        context: Union[List[str], np.ndarray],
    ) -> float:
        """

        :param target:
        :param context:
        :return:
        """
        n_target, n_context = len(target), len(context)
        if min(n_target, n_context) == 0:
            return np.nan

        context_counts = Counter(context)
        smooth_v = len(context_counts) + 1 if self._smooth else 0
        smooth_k = 1 if self._smooth else 0
        value = 0 if self._smooth else 1

        return (
            sum(
                -np.log((context_counts.get(token, value) + smooth_k) / (n_context + smooth_v))
                for token in target
            )
            / n_target
        )
