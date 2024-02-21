from collections import Counter
from typing import Optional, Any, Union, List

import numpy as np

from .language_model import LanguageModel


class ConvoKitLanguageModel(LanguageModel):
    """A simple language model to compute the deviation of target from context.

    This language model implements cross-entropy and perplexity language model evaluation functions,
    to be used in evaluating the average deviation of target from the specified context.

    :param model_type: The name (identifier) of the :py:class:`~convokit.ConvoKitLanguageModel`,
        defaults to "convokit_lm". Note that the `model_type` can be accessed using the `type`
        property (e.g., `lm.type`).
    :param kwargs: Any additional keyword arguments needed in the language model evaluations. This
        language model currently uses the following keyword arguments:

        * `smooth`: Indicator of using Laplace smoothing in the computation of cross-entropy scores,
          defaults to `True`.
        * `n_jobs`: The number of concurrent threads to be used for routines that are parallelized
          with `joblib`, defaults to 1.

        The language model configuration can be retrieved using the `config` property of the model
        class object (e.g., `lm.config`).
    """

    def __init__(self, model_type: str = "convokit_lm", **kwargs: Optional[Any]):
        super().__init__(model_type, **kwargs)

        self._smooth = kwargs["smooth"] if "smooth" in kwargs else True

    def cross_entropy(
        self,
        target: Union[List[str], np.ndarray],
        context: Union[List[str], np.ndarray],
    ) -> float:
        r"""Implements the base class method to compute the cross-entropy.

        Calculates :math:`H(P, Q) = -\sum_{x \in X}P(x) \times \ln(Q(x))`. Note that we use the
        natural logarithm; however, any base and corresponding exponent can be employed. For
        instance, KenLM uses base-10 (see :py:class:`~convokit.Kenlm` for reference).

        The smoothing boolean argument, `smooth`, is accessed from the setting in the language model
        constructor (defaults to `True` when unspecified).

        :param target: A list of tokens that make up the target text (P).
        :param context: A list of tokens that make up the context text (Q).
        :return: The cross-entropy score computed as :math:`H(P, Q)`.
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

    def perplexity(
        self, target: Union[List[str], np.ndarray], context: Union[List[str], np.ndarray]
    ) -> float:
        r"""Implements the base class method to compute perplexity.

        Calculates :math:`\text{PPL}(P, Q) = \exp(-\sum_{x \in X}P(x) \times \ln(Q(x)))`. Note that
        we use the natural logarithm; however, any base and corresponding exponent can be employed.
        For instance, KenLM uses base-10 (see :py:class:`~convokit.Kenlm` for reference).

        For convenience, the perplexity score is computed as the exponentiation of the cross-entropy
        calculated using the `cross_entropy()` method.

        :param target: A list of tokens that make up the target text (P).
        :param context: A list of tokens that make up the context text (Q).
        :return: The perplexity score computed as :math:`\text{PPL}(P, Q)`.
        """
        return np.exp(self.cross_entropy(target, context))
