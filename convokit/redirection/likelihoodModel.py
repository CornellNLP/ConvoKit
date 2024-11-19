from abc import ABC, abstractmethod
from typing import Callable


class LikelihoodModel(ABC):
    def __init__(self):
        self._name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @abstractmethod
    def fit(self, train_data, val_data):
        pass

    @abstractmethod
    def transform(self, test_data):
        pass
