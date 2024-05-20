from abc import ABC, abstractmethod
from typing import Callable

class ForecasterModel(ABC):
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
