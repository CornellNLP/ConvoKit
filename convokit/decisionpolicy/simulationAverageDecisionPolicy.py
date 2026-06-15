from typing import Callable, Optional, Dict, Any, Tuple

from convokit.decisionpolicy import DeferralDecisionPolicy


class SimulationAverageDecisionPolicy(DeferralDecisionPolicy):
    """
    decision policy that intervenes if the mean of the simulated next-utterance
    scores is at or above the threshold.

    this subclass inherits all simulation fetching, per-utterance metadata
    caching, sim-score caching, and threshold fitting from
    DeferralDecisionPolicy. the only differences are:
      * no ``tau`` parameter (unused; forwarded as 0 to super)
      * ``decide`` predicts based on mean(simulation_scores) >= threshold
    """

    def __init__(
        self,
        simulator,
        threshold,
        num_simulations: int = 10,
        store_simulations: bool = False,
        simulated_reply_attribute_name: str = "sim_replies",
        sim_replies_forecast_probs_attribute_name: str = "sim_replies_forecast_probs",
        reuse_cached_simulations: bool = True,
    ):
        # tau is irrelevant for the mean-based decision rule, so we pin it to 0
        # upstream rather than expose it to callers of this subclass.
        super().__init__(
            simulator=simulator,
            threshold=threshold,
            tau=0,
            num_simulations=num_simulations,
            store_simulations=store_simulations,
            simulated_reply_attribute_name=simulated_reply_attribute_name,
            sim_replies_forecast_probs_attribute_name=sim_replies_forecast_probs_attribute_name,
            reuse_cached_simulations=reuse_cached_simulations,
        )

    def decide(self, context, score_fn: Callable) -> Tuple[float, int, Optional[Dict[str, Any]]]:
        decision_score, simulations, simulation_scores = self._decision_score(context, score_fn)
        # empty simulation_scores would zero-divide. this happens when the
        # simulator returns no completions for a context (e.g. end-of-conversation
        # contexts that slip through the selector). treat as no intervention so
        # a single degenerate context doesn't abort the whole transform run.
        if len(simulation_scores) == 0:
            average_simulation_score = 0.0
            pred = 0
        else:
            average_simulation_score = sum(simulation_scores) / len(simulation_scores)
            pred = 1 if average_simulation_score >= self.threshold else 0
        return (
            decision_score,
            pred,
            {
                self.simulated_reply_attribute_name: simulations,
                self.sim_replies_forecast_probs_attribute_name: simulation_scores,
            },
        )
