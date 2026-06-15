from typing import Callable, Tuple

from .decisionPolicy import DecisionPolicy


class ThresholdDecisionPolicy(DecisionPolicy):
    """
    A simple decision policy that predicts 1 when score > threshold.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        forecast_prob_attribute_name: str = "forecast_prob",
        reuse_cached_forecast_probs: bool = True,
    ):
        super().__init__(
            forecast_prob_attribute_name=forecast_prob_attribute_name,
            reuse_cached_forecast_probs=reuse_cached_forecast_probs,
        )
        self.threshold = float(threshold)

    def decide(self, context, score_fn: Callable) -> Tuple[float, int]:
        score = self._score(context, score_fn)
        return score, int(score > self.threshold)

    def fit(self, contexts, val_contexts=None, score_fn: Callable = None):
        if val_contexts is None or score_fn is None or self.labeler is None:
            print(
                "either no validation contexts/score function/labeler were provided, returning current threshold"
            )
            return {"best_threshold": self.threshold}

        val_contexts = list(val_contexts)
        if len(val_contexts) == 0:
            print("no validation contexts were provided, returning current threshold")
            return {"best_threshold": self.threshold}

        fit_result = self._fit_with_model_checkpoint_selection(val_contexts, score_fn=score_fn)
        if isinstance(fit_result, dict):
            if "best_threshold" in fit_result:
                self.threshold = float(fit_result["best_threshold"])
            return fit_result

        fit_result = self._fit_threshold_for_loaded_model(val_contexts, score_fn=score_fn)
        if "best_threshold" in fit_result:
            self.threshold = float(fit_result["best_threshold"])
        return fit_result
