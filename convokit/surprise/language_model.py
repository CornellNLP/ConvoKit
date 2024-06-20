from abc import ABC
from typing import Optional, Any, List, Dict, Union, final

import numpy as np
from joblib import Parallel, delayed


class LanguageModel(ABC):
    """The abstract base class for all language models.

    The language model base class defines the :py:meth:`~convokit.LanguageModel.evaluate` method,
    which performs language model evaluation using the `eval_type` specified as an argument to the
    :py:meth:`~convokit.LanguageModel.evaluate` method. Note that this method must be defined and
    implemented in the subclass (e.g., if the `eval_type` is set to "cross_entropy", the subclass
    must implement :py:meth:`~convokit.LanguageModel.cross_entropy` method). The implemented method
    should take in a list of target tokens and a list of context tokens, and output the language
    model evaluation score.

    Since most language models employs cross-entropy and perplexity evaluations, this base class
    includes unimplemented designs of :py:meth:`~convokit.LanguageModel.cross_entropy` and
    :py:meth:`~convokit.LanguageModel.perplexity` functions, which may be implemented (as needed) in
    the subclasses. See the subclass implementations: :py:class:`~convokit.ConvoKitLanguageModel`
    and :py:class:`~convokit.Kenlm` classes, which extend this base class.

    The :py:meth:`~convokit.LanguageModel.evaluate` method defined in this class is called on a set
    of context samples and a set of target samples, and evaluates the target-context distribution
    deviations using the `eval_type` language model evaluation function.

    Note: The subclasses cannot override the :py:meth:`~convokit.LanguageModel.evaluate` method.

    :param model_type: The name (identifier) of :py:class:`~convokit.LanguageModel`, defaults to
        "language_model". Note that the `model_type` can be accessed using the `type` property
        (e.g., `lm.type`).
    :param kwargs: Any additional keyword arguments needed in the language model evaluations. For
        instance, the cross-entropy computes might require smoothing parameter; hence, a `smooth`
        parameter can be passed as an additional keyword argument.
        Another keyword argument is `n_jobs`, used to specify the number of concurrent threads to be
        used for routines that are parallelized with `joblib`, defaults to 1.
        The language model configuration can be retrieved using the `config` property of the model
        class object (e.g., `lm.config`).
    """

    def __init__(self, model_type: str = "language_model", **kwargs: Optional[Any]):
        self._model_type = model_type
        self._n_jobs = kwargs["n_jobs"] if "n_jobs" in kwargs else 1

        self.__dict__.update((f"_{arg}", value) for arg, value in kwargs.items())

    @property
    def type(self) -> str:
        """The `model_type` property of the language model.

        :return: The `model_type` specified in the class constructor, defaults to "language_model".
        """
        return self._model_type

    @property
    def config(self) -> Dict[str, Any]:
        """The configuration (all the class parameters) of the language model.

        :return: The configuration (all the class parameters specified in the class constructor and
            elsewhere) of the language model.
        """
        private_var_prefix = f"_{self.__class__.__name__}"
        return {
            arg[1:]: value
            for arg, value in self.__dict__.items()
            if not arg.startswith(private_var_prefix)
        }

    def _overwrite_args(self, args_to_overwrite: List[str], kwargs: Dict[str, Any]):
        """Overwrites the class variables with the values specified in `kwargs`.

        :param args_to_overwrite: The list of arguments (class variable names) whose values are to
            be overwritten using the values in the `kwargs`.
        :param kwargs: The keyword arguments with updates to the values of the class variables.
        """
        for arg in args_to_overwrite:
            self.__dict__[f"_{arg}"] = kwargs[arg] if arg in kwargs else self.__dict__[f"_{arg}"]

    def cross_entropy(
        self, target: Union[List[str], np.ndarray], context: Union[List[str], np.ndarray]
    ) -> float:
        r"""An unimplemented base class method to compute the cross-entropy.

        The cross-entropy between a list of target tokens and a list of context tokens is to be
        computed by the implementation in the subclass. Note that any variables to be used in this
        method (e.g., smoothing value) must be accessed from the class scope.

        Calculates :math:`H(P, Q) = -\sum_{x \in X}P(x) \times \ln(Q(x))`.

        Note that we use the natural logarithm; however, any base and corresponding exponent can be
        employed. For instance, KenLM uses base-10 (see :py:class:`~convokit.Kenlm` for reference).

        :param target: A list of tokens that make up the target text (P).
        :param context: A list of tokens that make up the context text (Q).
        :raises: Raises a `RuntimeError` if called without implementing it in the subclass.
        """
        raise RuntimeError("cross entropy is not implemented")

    def perplexity(
        self, target: Union[List[str], np.ndarray], context: Union[List[str], np.ndarray]
    ) -> float:
        r"""An unimplemented base class method to compute perplexity.

        The perplexity between a list of target tokens and a list of context tokens is to be
        computed by the implementation in the subclass. Note that any variables to be used in this
        method (e.g., smoothing value) must be accessed from the class scope.

        Calculates :math:`\text{PPL}(P, Q) = \exp(-\sum_{x \in X}P(x) \times \ln(Q(x)))`.

        Note that we use the natural logarithm; however, any base and corresponding exponent can be
        employed. For instance, KenLM uses base-10 (see :py:class:`~convokit.Kenlm` for reference).

        :param target: A list of tokens that make up the target text (P).
        :param context: A list of tokens that make up the context text (Q).
        :raises: Raises a `RuntimeError` if called without implementing it in the subclass.
        """
        raise RuntimeError("perplexity is not implemented")

    @final
    def evaluate(
        self,
        target_samples: Union[List[List[str]], np.ndarray],
        context_samples: Union[List[List[str]], np.ndarray],
        eval_type: str = "cross_entropy",
        **kwargs: Optional[Any],
    ) -> np.ndarray:
        """Computes the average deviation between target and context distributions.

        For a given list of (fixed size) target sample lists and (fixed size) context sample lists,
        the :py:meth:`~convokit.LanguageModel.evaluate` method computes the deviation between each
        target and corresponding context pair, using `eval_type` language model evaluation metric.
        Note that the subclass implementing this abstract base class must define and implement the
        `eval_type` evaluation method. The final score output by this method is an average of all
        the individual scores.

        Also note that, if specified as keyword arguments, any class variable values are overwritten
        from within this method.

        :param target_samples: A list of target sample lists to be used to evaluate against the
            corresponding context sample lists.
        :param context_samples: A list of context sample lists that are to be used in evaluating the
            corresponding target sample lists.
        :param eval_type: The language model evaluation function (as `str`), used in evaluating the
            language model trained using the context text, evaluated using the target text. Defaults
            to "cross_entropy", i.e., calls the :py:meth:`~convokit.LanguageModel.cross_entropy`
            method.
        :param kwargs: Any additional keyword arguments needed in the language model evaluations. If
            any class variables are passed using `kwargs`, the corresponding class variable values
            are overwritten using the new values.
        :return: The average score that measures the average deviation of target text from context.
        """
        self._overwrite_args(list(kwargs.keys()), kwargs)
        eval_fn = getattr(self, eval_type)

        if self._n_jobs == 1:
            model_scores = [
                eval_fn(target_sample, context_sample)
                for target_sample, context_sample in zip(target_samples, context_samples)
            ]
        else:
            model_scores = Parallel(n_jobs=self._n_jobs, backend="threading")(
                delayed(eval_fn)(target_sample, context_sample)
                for target_sample, context_sample in zip(target_samples, context_samples)
            )
        return np.nanmean(model_scores)
