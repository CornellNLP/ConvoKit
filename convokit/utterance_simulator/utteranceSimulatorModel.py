from abc import ABC, abstractmethod
from typing import Callable


class UtteranceSimulatorModel(ABC):
    """
    Abstract class representing a model to simulate the next utterance in
    the conversation. Different models can be supported by inheriting from
    this base class.
    """

    def __init__(self):
        self._name = None

    @property
    def name(self):
        """
        Name of the simulator model.
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of the simulator model.

        :param name: Name of model
        """
        self._name = name

    @abstractmethod
    def fit(self, contexts, val_contexts):
        """
        Finetune this utterance simulator model on the given `contexts` and
        validates on `val_contexts`.

        :param contexts: Iterator over context tuples for training
        :param val_contexts: Iterator over context tuples for validation
        """
        pass

    @abstractmethod
    def transform(self, contexts, simulated_reply_attribute_name):
        """
        Apply this model to the given data, and return its simulated responses
        in the form of a DataFrame indexed by (current) utterance ID

        :param contexts: Iterator over context tuples

        :return: a Pandas DataFrame, with one row for each context, indexed by
            the ID of that context's current utterance. Contains list of
            simulated replies in the `simulated_reply_attribute_name` column.
        """
        pass
