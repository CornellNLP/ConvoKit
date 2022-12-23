import unittest

import nltk.lm as nltk_lm
from nltk.util import ngrams, everygrams

from convokit.surprise import language_model


class TestLm(language_model.LanguageModel):
    def __init__(self):
        super().__init__("test_language_model")

    @staticmethod
    def eval_func(target, context):
        return abs(len(context) - len(target))


class TestNltkLm(language_model.LanguageModel):
    def __init__(self, ngram_order=2):
        super().__init__("test_nltk_language_model")
        self._ngram_order = ngram_order

    def eval_func(self, target, context):
        kneser_ney_lm = nltk_lm.models.KneserNeyInterpolated(
            order=self._ngram_order, vocabulary=nltk_lm.Vocabulary(target + context)
        )
        kneser_ney_lm.fit([everygrams(context, max_len=self._ngram_order)])
        return kneser_ney_lm.entropy(ngrams(target, n=self._ngram_order))


class TestLanguageModel(unittest.TestCase):
    def _init(self, target_samples, context_samples):
        self._target_samples = target_samples
        self._context_samples = context_samples

    def test_model_type(self):
        test_lm = language_model.LanguageModel(model_type="test_language_model")
        self.assertEqual(test_lm.type, "test_language_model")

    def test_model_config(self):
        test_lm = language_model.LanguageModel(model_type="test_language_model", smooth=True)
        expected_config = {"model_type": "test_language_model", "n_jobs": 1, "smooth": True}
        self.assertEqual(test_lm.config, expected_config)

    def test_overwrite_args(self):
        test_lm = language_model.LanguageModel(model_type="test_language_model", smooth=True)
        try:
            test_lm.evaluate(self._target_samples, self._context_samples, smooth=False)
        except RuntimeError:
            pass
        expected_config = {"model_type": "test_language_model", "n_jobs": 1, "smooth": False}
        self.assertEqual(test_lm.config, expected_config)

    def test_evaluate_cross_entropy_runtime_error(self):
        test_lm = language_model.LanguageModel(model_type="test_language_model")
        with self.assertRaises(RuntimeError):
            test_lm.evaluate(self._target_samples, self._context_samples, "cross_entropy")

    def test_evaluate_perplexity_runtime_error(self):
        test_lm = language_model.LanguageModel(model_type="test_language_model")
        with self.assertRaises(RuntimeError):
            test_lm.evaluate(self._target_samples, self._context_samples, "perplexity")

    def test_evaluate_unimplemented_attribute_error(self):
        test_lm = language_model.LanguageModel(model_type="test_language_model")
        with self.assertRaises(AttributeError):
            test_lm.evaluate(self._target_samples, self._context_samples, "unimplemented")

    def test_evaluate(self):
        test_lm = TestLm()
        score = test_lm.evaluate(self._target_samples, self._context_samples, "eval_func")
        self.assertEqual(score, 0.5)

    def test_evaluate_nltk(self):
        test_lm = TestNltkLm()
        score = test_lm.evaluate(self._target_samples, self._context_samples, "eval_func")
        self.assertEqual(round(float(score), 2), 1.25)


class TestWithMemory(TestLanguageModel):
    def setUp(self) -> None:
        self._target_samples = [["this", "is", "test"], ["is", "test"]]
        self._context_samples = [["this", "is", "a", "test"], ["this", "test"]]
        super()._init(self._target_samples, self._context_samples)

    def test_model_type(self):
        super().test_model_type()

    def test_model_config(self):
        super().test_model_config()

    def test_overwrite_args(self):
        super().test_overwrite_args()

    def test_evaluate_cross_entropy_runtime_error(self):
        super().test_evaluate_cross_entropy_runtime_error()

    def test_evaluate_perplexity_runtime_error(self):
        super().test_evaluate_perplexity_runtime_error()

    def test_evaluate_unimplemented_attribute_error(self):
        super().test_evaluate_unimplemented_attribute_error()

    def test_evaluate(self):
        super().test_evaluate()

    def test_evaluate_nltk(self):
        super().test_evaluate_nltk()


class TestWithDb(TestLanguageModel):
    def setUp(self) -> None:
        self._target_samples = [["this", "is", "test"], ["is", "test"]]
        self._context_samples = [["this", "is", "a", "test"], ["this", "test"]]
        super()._init(self._target_samples, self._context_samples)

    def test_model_type(self):
        super().test_model_type()

    def test_model_config(self):
        super().test_model_config()

    def test_overwrite_args(self):
        super().test_overwrite_args()

    def test_evaluate_cross_entropy_runtime_error(self):
        super().test_evaluate_cross_entropy_runtime_error()

    def test_evaluate_perplexity_runtime_error(self):
        super().test_evaluate_perplexity_runtime_error()

    def test_evaluate_unimplemented_attribute_error(self):
        super().test_evaluate_unimplemented_attribute_error()

    def test_evaluate(self):
        super().test_evaluate()

    def test_evaluate_nltk(self):
        super().test_evaluate_nltk()
