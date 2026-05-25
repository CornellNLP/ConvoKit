from typing import Callable, Optional, Dict, Any, Tuple

from convokit.decisionpolicy import DeferralDecisionPolicy


class SimulationMajorityDecisionPolicy(DeferralDecisionPolicy):
    """
    decision policy that intervenes if at least ``tau`` of the simulated next
    utterances score above the threshold, ignoring the current utterance score.

    this subclass inherits all simulation fetching, per-utterance metadata
    caching, sim-score caching, and threshold fitting from
    DeferralDecisionPolicy. the only difference is in ``decide``: the gate
    ``decision_score > threshold`` is dropped so that only the simulated-branch
    vote count drives the prediction.
    """

    def decide(self, context, score_fn: Callable) -> Tuple[float, int, Optional[Dict[str, Any]]]:
        decision_score, simulations, simulation_scores = self._decision_score(context, score_fn)
        num_simulations_above_threshold = sum(
            1 for score in simulation_scores if score > self.threshold
        )
        return (
            decision_score,
            1 if num_simulations_above_threshold >= self.tau else 0,
            {
                self.simulated_reply_attribute_name: simulations,
                self.sim_replies_forecast_probs_attribute_name: simulation_scores,
            },
        )
