import numpy as np
from typing import Callable, List, Optional, Dict, Any, Tuple
from .decisionPolicy import DecisionPolicy

class RandomDeferralDecisionPolicy(DecisionPolicy):
    """
    Decision policy that defers intervention by looking ahead at simulated next utterances.

    :param simulator: utterance simulator model (must have a ``transform(contexts)`` method
        returning a DataFrame indexed by utterance id). if the simulator exposes
        ``get_num_simulations()``, ``num_simulations`` is capped to that value.
    :param threshold: probability threshold above which a context is flagged.
    :param tau: minimum number of simulated branches that must exceed the threshold
        before an intervention is issued.
    :param num_simulations: how many simulated branches to use per context (capped to
        simulator's ``get_num_simulations()`` if available).
    :param store_simulations: if True, simulated reply strings are cached during decide()
        and written to corpus utterance metadata by post_transform().
    :param simulated_reply_attribute_name: metadata field name used when storing simulations
        on corpus utterances (only relevant when store_simulations=True).
    """

    def __init__(
        self,
        simulator,
        threshold,
        deferral_probability: float = 0.1515,
        reuse_cached_forecast_probs: bool = True,
        forecast_prob_attribute_name: str = "forecast_prob",
    ):
        # forward the cache flag to the base class so its _score helper honors it.
        # without this, reuse_cached_probabilities on this subclass had no effect.
        super().__init__(
            forecast_prob_attribute_name=forecast_prob_attribute_name,
            reuse_cached_forecast_probs=reuse_cached_forecast_probs,
        )
        self.simulator = simulator
        self.threshold = float(threshold)
        self.deferral_probability = float(deferral_probability)

    def _decision_score(self, context, score_fn: Callable):
        # use base _score so a cached forecast_prob on the utterance meta is reused
        # instead of re-invoking the belief estimator.
        return self._score(context, score_fn)

    def decide(self, context, score_fn: Callable) -> Tuple[float, int, Optional[Dict[str, Any]]]:
        decision_score = self._score(context, score_fn)

        p = float(np.random.rand())
   

        # return an empty metadata dict (not None) so downstream code that iterates
        # utt_metadata.items() in TransformerDecoderModel.transform doesn't crash.
        return (decision_score,
        1 if decision_score > self.threshold and p > self.deferral_probability else 0,
        {}
        )

    def fit(self, contexts, val_contexts=None, score_fn: Callable = None):
        if val_contexts is None or score_fn is None or self.labeler is None:
            print("either no validation contexts/score function/labeler were provided, returning current threshold")
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