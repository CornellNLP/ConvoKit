import random
import unittest

import numpy as np

from convokit.surprise import Surprise, ConvoKitLanguageModel
from convokit.tests.test_utils import small_burr_conv_corpus


class TestSurprise(unittest.TestCase):
    def _init(self, corpus) -> None:
        self._corpus = corpus

    def test_fit_model_groups(self):
        surprise = Surprise(
            model_key_selector=lambda utt: "_".join([utt.speaker.id, utt.conversation_id])
        )
        surprise = surprise.fit(self._corpus)
        expected_model_groups = {
            "hamilton_0": [["Pardon", "me", "."]],
            "hamilton_1": [["Who", "'s", "asking", "?"]],
            "hamilton_2": [["Are", "you", "Aaron", "Burr", ",", "sir", "?"]],
            "burr_0": [["Are", "you", "Aaron", "Burr", ",", "sir", "?"]],
            "burr_1": [["That", "depends", ".", "Pardon", "me", "."]],
            "burr_2": [["That", "depends", "."]],
        }
        self.assertEqual(surprise._model_groups, expected_model_groups)

    def test_fit_model_groups_text_func_selector(self):
        surprise = Surprise(
            model_key_selector=lambda utt: "_".join([utt.speaker.id, utt.conversation_id])
        )
        surprise = surprise.fit(
            self._corpus,
            text_func=lambda utt: [
                " ".join(
                    [
                        speaker_utt.text
                        for speaker_utt in utt.speaker.iter_utterances()
                        if speaker_utt.conversation_id != utt.conversation_id
                    ]
                )
            ],
            selector=lambda utt: utt.conversation_id == "0",
        )
        expected_model_groups = {
            "hamilton_0": [
                ["Who", "'s", "asking", "?", "Are", "you", "Aaron", "Burr", ",", "sir", "?"]
            ],
            "burr_0": [["That", "depends", ".", "Pardon", "me", ".", "That", "depends", "."]],
        }
        self.assertEqual(surprise._model_groups, expected_model_groups)

    def test_transform_large_context_target_size(self):
        surprise = Surprise(model_key_selector=lambda utt: "corpus")
        surprise = surprise.fit(
            self._corpus,
            text_func=lambda utt: [
                " ".join([corpus_utt.text for corpus_utt in self._corpus.iter_utterances()])
            ],
        )
        transformed_corpus = surprise.transform(self._corpus, obj_type="utterance")

        utts = transformed_corpus.get_utterances_dataframe()["meta.surprise"]
        surprise_scores = np.array([score["corpus"] for score in utts])
        self.assertTrue(np.isnan(surprise_scores).all())

    def test_transform_multiple_jobs(self):
        surprise = Surprise(model_key_selector=lambda utt: "corpus", n_jobs=2)
        surprise = surprise.fit(
            self._corpus,
            text_func=lambda utt: [
                " ".join([corpus_utt.text for corpus_utt in self._corpus.iter_utterances()])
            ],
        )
        transformed_corpus = surprise.transform(self._corpus, obj_type="utterance", n_jobs=2)

        utts = transformed_corpus.get_utterances_dataframe()["meta.surprise"]
        surprise_scores = np.array([score["corpus"] for score in utts])
        self.assertTrue(np.isnan(surprise_scores).all())

    def test_transform_convokit_language_model(self):
        random.Random(42)
        surprise = Surprise(
            model_key_selector=lambda utt: "corpus", target_sample_size=3, context_sample_size=3
        )
        surprise = surprise.fit(
            self._corpus,
            text_func=lambda utt: [
                " ".join([corpus_utt.text for corpus_utt in self._corpus.iter_utterances()])
            ],
        )
        language_model = ConvoKitLanguageModel(smooth=False)
        transformed_corpus = surprise.transform(
            self._corpus, obj_type="utterance", language_model=language_model
        )

        utts = transformed_corpus.get_utterances_dataframe()["meta.surprise"]
        surprise_scores = np.round(np.array([score["corpus"] for score in utts]), 1)
        expected_scores = np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
        self.assertTrue(np.allclose(surprise_scores, expected_scores, atol=1e-01))

    def test_transform_language_model_parameters(self):
        random.Random(42)
        surprise = Surprise(
            model_key_selector=lambda utt: "corpus", target_sample_size=3, context_sample_size=3
        )
        surprise = surprise.fit(
            self._corpus,
            text_func=lambda utt: [
                " ".join([corpus_utt.text for corpus_utt in self._corpus.iter_utterances()])
            ],
        )
        transformed_corpus = surprise.transform(self._corpus, obj_type="utterance", smooth=False)

        utts = transformed_corpus.get_utterances_dataframe()["meta.surprise"]
        surprise_scores = np.round(np.array([score["corpus"] for score in utts]), 1)
        expected_scores = np.array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1])
        self.assertTrue(np.allclose(surprise_scores, expected_scores, atol=1e-01))

    def test_transform(self):
        random.Random(42)
        surprise = Surprise(
            model_key_selector=lambda utt: "corpus", target_sample_size=3, context_sample_size=3
        )
        surprise = surprise.fit(
            self._corpus,
            text_func=lambda utt: [
                " ".join([corpus_utt.text for corpus_utt in self._corpus.iter_utterances()])
            ],
        )
        transformed_corpus = surprise.transform(self._corpus, obj_type="utterance")

        utts = transformed_corpus.get_utterances_dataframe()["meta.surprise"]
        surprise_scores = np.round(np.array([score["corpus"] for score in utts]), 1)
        expected_scores = np.array([1.8, 1.7, 1.7, 1.8, 1.7, 1.8, 1.8])
        self.assertTrue(np.allclose(surprise_scores, expected_scores, atol=1e-01))


class TestWithMemory(TestSurprise):
    def setUp(self) -> None:
        self._small_burr_corpus = small_burr_conv_corpus()
        super()._init(self._small_burr_corpus)

    def test_fit_model_groups(self):
        super().test_fit_model_groups()

    def test_fit_model_groups_text_func_selector(self):
        super().test_fit_model_groups_text_func_selector()

    def test_transform_large_context_target_size(self):
        super().test_transform_large_context_target_size()

    def test_transform_multiple_jobs(self):
        super().test_transform_multiple_jobs()

    def test_transform_convokit_language_model(self):
        super().test_transform_convokit_language_model()

    def test_transform_language_model_parameters(self):
        super().test_transform_language_model_parameters()

    def test_transform(self):
        super().test_transform()


class TestWithDb(TestSurprise):
    def setUp(self) -> None:
        self._small_burr_corpus = small_burr_conv_corpus()
        super()._init(self._small_burr_corpus)

    def test_fit_model_groups(self):
        super().test_fit_model_groups()

    def test_fit_model_groups_text_func_selector(self):
        super().test_fit_model_groups_text_func_selector()

    def test_transform_large_context_target_size(self):
        super().test_transform_large_context_target_size()

    def test_transform_multiple_jobs(self):
        super().test_transform_multiple_jobs()

    def test_transform_convokit_language_model(self):
        super().test_transform_convokit_language_model()

    def test_transform_language_model_parameters(self):
        super().test_transform_language_model_parameters()

    def test_transform(self):
        super().test_transform()
