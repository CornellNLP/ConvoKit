from convokit import Transformer
from .likelihoodModel import LikelihoodModel
from .contextSelector import default_previous_context_selector, default_future_context_selector
import torch
import random
from .preprocessing import format_conversations, get_chunk_dataset


class RedirectionModel(Transformer):
    def __init__(
        self,
        likelihood_model,
        previous_context_selector=default_previous_context_selector,
        future_context_selector=default_future_context_selector,
        redirection_attribute_name="redirection",
    ):
        self.likelihood_model = likelihood_model
        self.previous_context_selector = previous_context_selector
        self.future_context_selector = future_context_selector
        self.redirection_attribute_name = redirection_attribute_name

    def fit(self, corpus, train_selector=lambda convo: True, val_selector=lambda convo: True):
        train_convos = [convo for convo in corpus.iter_conversations() if train_selector(convo)]
        val_convos = [convo for convo in corpus.iter_conversations() if val_selector(convo)]
        train_convos_formatted = format_conversations(train_convos)
        val_convos_formatted = format_conversations(val_convos)
        train_data = get_chunk_dataset(
            train_convos_formatted, max_tokens=self.likelihood_model.max_length
        )
        val_data = get_chunk_dataset(
            train_convos_formatted, max_tokens=self.likelihood_model.max_length
        )
        self.likelihood_model.fit(train_data=train_data, val_data=val_data)
        return self

    def transform(self, corpus, selector=lambda convo: True, verbosity=5):
        test_convos = [convo for convo in corpus.iter_conversations() if selector(convo)]
        actual_contexts = []
        reference_contexts = []
        future_contexts = []
        print("Computing contexts")
        for i, convo in enumerate(len(test_convos)):
            if i % verbosity == 0 and i > 0:
                print(i, "/", len(test_convos))
            actual, reference = self.previous_context_selector(convo)
            future = self.future_context_selector(convo)
            actual_contexts.append(actual)
            reference_contexts.append(reference)
            future_contexts.append(future)

        print("Computing actual likelihoods")
        test_data = (actual_contexts, future_contexts)
        actual_likelihoods = self.likelihood_model.transform(test_data, verbosity=verbosity)

        print("Computing reference likelihoods")
        test_data = (reference_contexts, future_contexts)
        reference_likelihoods = self.likelihood_model.transform(test_data, verbosity=verbosity)

        print("Computing redirection scores")
        for i, convo in enumerate(len(test_convos)):
            if i % verbosity == 0 and i > 0:
                print(i, "/", len(test_convos))
            for utt in convo.iter_utterances():
                if utt.id in actual_likelihoods and utt.id in reference_likelihoods:
                    actual_prob = actual_likelihoods[utt.id]
                    reference_prob = reference_likelihoods[utt_id]
                    redirection = (
                        actual_prob
                        - torch.log(1 - torch.exp(actual_prob))
                        - (reference_prob + torch.log(1 - torch.exp(reference_prob)))
                    )
                    utt.meta[self.redirection_attribute_name] = redirection

        return corpus

    def fit_transform(
        self,
        train_selector=lambda convo: True,
        val_selector=lambda convo: True,
        test_selector=lambda convo: True,
        verbosity=10,
    ):
        self.fit(corpus, train_selector=train_selector, val_selector=val_selector)
        return self.transform(corpus, selector=test_selector, verbosity=verbosity)

    def summarize(self, top_sample_size=10, bottom_sample_size=10):
        utts = [
            utt for utt in corpus.iter_utterances() if self.redirection_attribute_name in utt.meta
        ]
        sorted_utts = sorted(utts, key=lambda utt: utt.meta[self.redirection_attribute_name])
        top_sample_size = min(top_sample_size, len(sorted_utts))
        bottom_sample_size = min(bottom_sample_size, len(sorted_utts))
        print("[high]" + self.redirection_attribute_name)
        for i in range(-1, -1 - top_sample_size, -1):
            utt = sorted_utts[i]
            print(utt.speaker.id, ":", utt.text, "\n")

        print()

        print("[low]" + self.redirection_attribute_name)
        for i in range(bottom_sample_size):
            utt = sorted_utts[i]
            print(utt.speaker.id, ":", utt.text, "\n")

        return self