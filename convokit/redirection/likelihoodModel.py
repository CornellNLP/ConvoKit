from abc import ABC, abstractmethod
from typing import Callable


class LikelihoodModel(ABC):
    """
    Abstract class representing a model to compute utterance likelihoods
    based on provided context. Different models (Gemma, Llama, Mistral, etc.)
    can be supported by inheriting from this base class.
    """

    def __init__(self):
        self._name = None

    @property
    def name(self):
        """
        Name of the likelihood model.
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Sets the name of the likelihood model.

        :param name: Name of model
        """
        self._name = name

    @abstractmethod
    def fit(self, train_data, val_data):
        """
        Fine-tunes the likelihood model on the provided `train_data` and
        validates on `val_data`.

        :param train_data: Data to fine-tune model
        :param val_data: Data to validate model
        """
        pass

    @abstractmethod
    def transform(self, test_data):
        """
        Computes the utterance likelihoods for the provided `test_data`.

        :param test_data: Data to compute likelihoods over

        :return: Likelihoods of the `test_data`
        """
        pass
