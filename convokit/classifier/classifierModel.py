from abc import ABC, abstractmethod
from typing import Callable


class ClassifierModel(ABC):
    """
    An abstract class defining an interface that Classifier can call into to invoke a conversational classification algorithm.
    The “contract” between Classifier and ClassifierModel means that ClassifierModel can expect to receive conversational data
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
        A method the user would use to fit the model.

        :param contexts: an iterator over context objects
        :param val_contexts: optional second iterator which would produce validation data for the model.
        """
        pass

    @abstractmethod
    def transform(self, contexts, classification_attribute_name, classification_prob_attribute_name):
        """
        Function underlying the higher-level `transform` method in the Classifier class which operates
        at a context level (again, Utterance, Conversation, or Speaker, etc.) to annotate.

        :param contexts: iterator over context objects, which may or not be narrowed down by the selector argument in the Classifier wrapper

        :return: a pandas DataFrame containing two added columns: one with raw probabilities named according to classification_prob_attribute_name, and one with discretized (binary) classification. Indexed by the ID of that context’s current utterance. 
        """
        pass
