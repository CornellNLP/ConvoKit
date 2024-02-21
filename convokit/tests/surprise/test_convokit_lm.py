import unittest

from convokit import ConvoKitLanguageModel


class TestConvoKitLanguageModel(unittest.TestCase):
    def _init(self, target_samples, context_samples):
        self._target_samples = target_samples
        self._context_samples = context_samples

    def test_cross_entropy_smooth(self):
        convokit_lm = ConvoKitLanguageModel(model_type="test_convokit_lm", smooth=True)
        score = convokit_lm.evaluate(
            self._target_samples, self._context_samples, eval_type="cross_entropy"
        )
        self.assertEqual(round(float(score), 2), 1.38)

    def test_cross_entropy_no_smooth(self):
        convokit_lm = ConvoKitLanguageModel(model_type="test_convokit_lm", smooth=False)
        score = convokit_lm.evaluate(
            self._target_samples, self._context_samples, eval_type="cross_entropy"
        )
        self.assertEqual(round(float(score), 2), 1.04)

    def test_perplexity_smooth(self):
        convokit_lm = ConvoKitLanguageModel(model_type="test_convokit_lm", smooth=True)
        score = convokit_lm.evaluate(
            self._target_samples, self._context_samples, eval_type="perplexity"
        )
        self.assertEqual(round(float(score), 2), 4.02)

    def test_perplexity_no_smooth(self):
        convokit_lm = ConvoKitLanguageModel(model_type="test_convokit_lm", smooth=False)
        score = convokit_lm.evaluate(
            self._target_samples, self._context_samples, eval_type="perplexity"
        )
        self.assertEqual(round(float(score), 2), 3.00)


class TestWithMemory(TestConvoKitLanguageModel):
    def setUp(self) -> None:
        self._target_samples = [["this", "is", "test"], ["is", "test"]]
        self._context_samples = [["this", "is", "a", "test"], ["this", "test"]]
        super()._init(self._target_samples, self._context_samples)

    def test_cross_entropy_smooth(self):
        super().test_cross_entropy_smooth()

    def test_cross_entropy_no_smooth(self):
        super().test_cross_entropy_no_smooth()

    def test_perplexity_smooth(self):
        super().test_perplexity_smooth()

    def test_perplexity_no_smooth(self):
        super().test_perplexity_no_smooth()


class TestWithDb(TestConvoKitLanguageModel):
    def setUp(self) -> None:
        self._target_samples = [["this", "is", "test"], ["is", "test"]]
        self._context_samples = [["this", "is", "a", "test"], ["this", "test"]]
        super()._init(self._target_samples, self._context_samples)

    def test_cross_entropy_smooth(self):
        super().test_cross_entropy_smooth()

    def test_cross_entropy_no_smooth(self):
        super().test_cross_entropy_no_smooth()

    def test_perplexity_smooth(self):
        super().test_perplexity_smooth()

    def test_perplexity_no_smooth(self):
        super().test_perplexity_no_smooth()
