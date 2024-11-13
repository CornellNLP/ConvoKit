from abc import ABC, abstractmethod
from typing import Callable


class ForecasterModel(ABC):
    """
    An abstract class defining an interface that Forecaster can call into to invoke a conversational forecasting algorithm.
    The “contract” between Forecaster and ForecasterModel means that ForecasterModel can expect to receive conversational data
    in a consistent format, defined above.
    """

    def __init__(self):
        self._labeler = None

    @property
    def labeler(self):
        return self._labeler

    @labeler.setter
    def labeler(self, value: Callable):
        self._labeler = value

    @abstractmethod
    def fit(self, contexts, val_contexts=None):
        """
        Train this conversational forecasting model on the given data

        :param contexts: an iterator over context tuples
        :param val_contexts: an optional second iterator over context tuples to be used as a separate held-out validation set. Concrete ForecasterModel implementations may choose to ignore this, or conversely even enforce its presence.
        """
        pass

    @abstractmethod
    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        """
        Apply this trained conversational forecasting model to the given data, and return its forecasts
        in the form of a DataFrame indexed by (current) utterance ID

        :param contexts: an iterator over context tuples

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name. Subclass implementations of ForecasterModel MUST adhere to this return value specification!
        """
        pass
