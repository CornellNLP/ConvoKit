from convokit import Transformer
from .likelihoodModel import LikelihoodModel
from .contextSelector import default_previous_context_selector, default_future_context_selector
import torch
import random
from .preprocessing import format_conversations, get_chunk_dataset
import numpy as np


class UtteranceLikelihood(Transformer):
    """
    ConvoKit Transformer to compute utterance likelihoods derived from
    `likelihood_model`. The contexts used to compute the likelihoods can be
    defined using `previous_context_selector` and `future_context_selector`,
    which are by default the immediate previous and current contexts from
    different speaker roles.

    :param likelihood_model: Likelihood model to compute utterance likelihoods
    :param previous_context_selector: Computes previous contexts
    :param default_future_context_selector: Computes future contexts
    :param likelihood_attribute_name: Name of meta-data attribute to
    save likelihoods
    """

    def __init__(
        self,
        likelihood_model,
        previous_context_selector=None,
        future_context_selector=None,
        likelihood_attribute_name="utterance_likelihood",
    ):
        self.likelihood_model = likelihood_model
        self.tokenizer = self.likelihood_model.tokenizer
        self.previous_context_selector = previous_context_selector
        self.future_context_selector = future_context_selector
        self.likelihood_attribute_name = likelihood_attribute_name

    def fit(self, corpus, train_selector=lambda convo: True, val_selector=lambda convo: True):
        """
        Fits the UtteranceLikelihood transformer to the corpus by generating the training
        and validation data and fine-tuning the likelihood model.

        :param corpus: Corpus to fit transformer
        :param train_selector: Selector for train conversations
        :param val_selector: Selector for val conversations
        """
        train_convos = [convo for convo in corpus.iter_conversations() if train_selector(convo)]
        val_convos = [convo for convo in corpus.iter_conversations() if val_selector(convo)]
        train_convos_formatted = format_conversations(train_convos)
        val_convos_formatted = format_conversations(val_convos)
        train_data = get_chunk_dataset(
            self.tokenizer, train_convos_formatted, max_tokens=self.likelihood_model.max_length
        )
        val_data = get_chunk_dataset(
            self.tokenizer, val_convos_formatted, max_tokens=self.likelihood_model.max_length
        )
        self.likelihood_model.fit(train_data=train_data, val_data=val_data)
        return self

    def transform(self, corpus, selector=lambda convo: True, verbosity=5):
        """
        Populates the corpus test data with utterance likelihoods, by first
        computing previous and future contexts.

        :param corpus: Corpus to transform
        :param selector: Selector for test data
        :param verbosity: Verbosity for update messages

        :return: Corpus where test data is labeled with utterance likelihoods
        """
        test_convos = [convo for convo in corpus.iter_conversations() if selector(convo)]
        previous_contexts = []
        future_contexts = []
        print("Computing contexts")
        for i, convo in enumerate(test_convos):
            if i % verbosity == 0 and i > 0:
                print(i, "/", len(test_convos))
            if self.previous_context_selector is None and self.future_context_selector is None:
                contexts, _ = default_previous_context_selector(convo)
                previous = {utt_id: pair[0] for utt_id, pair in contexts.items()}
                future = {utt_id: pair[1] for utt_id, pair in contexts.items()}
            else:
                previous = self.previous_context_selector(convo)
                future = self.future_context_selector(convo)

            previous_contexts.append(previous)
            future_contexts.append(future)

        print("Computing utterance likelihoods")
        test_data = (previous_contexts, future_contexts)
        likelihoods = self.likelihood_model.transform(test_data, verbosity=verbosity)

        print("Labeling utterance likelihoods")
        for i, convo in enumerate(test_convos):
            if i % verbosity == 0 and i > 0:
                print(i, "/", len(test_convos))
            likelihoods = likelihoods[i]
            for utt in convo.iter_utterances():
                if utt.id in likelihoods:
                    utt.meta[self.likelihood_attribute_name] = likelihoods[utt.id]

        return corpus

    def fit_transform(
        self,
        train_selector=lambda convo: True,
        val_selector=lambda convo: True,
        test_selector=lambda convo: True,
        verbosity=10,
    ):
        """
        Fit and transform the model.

        :param corpus: Corpus to transform
        :param train_selector: Selector for train data
        :param val_selector: Selector for val data
        :param test_selector: Selector for test data
        :param verbosity: Verbosity for update messages

        :return: Corpus where test data is labeled with utterance likelihoods
        """
        self.fit(corpus, train_selector=train_selector, val_selector=val_selector)
        return self.transform(corpus, selector=test_selector, verbosity=verbosity)

    def summarize(self, corpus, top_sample_size=10, bottom_sample_size=10):
        """
        Summarizes UtteranceLikelihood transformer using utterances with
        high and low probabilities.

        :param corpus: Corpus to analyze
        :param top_sample_size: Number of utterances to print for high probabilities
        :param bottom_sample_size: Number of utterances to print for low probabilities
        """
        utts = [utt for utt in corpus.iter_utterances() if self.utterance in utt.meta]
        sorted_utts = sorted(utts, key=lambda utt: utt.meta[self.likelihood_attribute_name])
        top_sample_size = min(top_sample_size, len(sorted_utts))
        bottom_sample_size = min(bottom_sample_size, len(sorted_utts))
        print("[high]" + self.likelihood_attribute_name)
        for i in range(-1, -1 - top_sample_size, -1):
            utt = sorted_utts[i]
            print(utt.speaker.id, ":", utt.text, "\n")

        print()

        print("[low]" + self.likelihood_attribute_name)
        for i in range(bottom_sample_size):
            utt = sorted_utts[i]
            print(utt.speaker.id, ":", utt.text, "\n")

        return self
