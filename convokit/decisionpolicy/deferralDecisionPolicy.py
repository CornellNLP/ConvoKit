from typing import Callable, List, Optional, Dict, Any, Tuple

from .decisionPolicy import DecisionPolicy


class _synthetic_speaker:
    def __init__(self, speaker_id: str):
        self.id = speaker_id


class _synthetic_utterance:
    def __init__(self, text: str, utterance_id: str, speaker_id: str):
        self.text = text
        self.id = utterance_id
        self.speaker_ = _synthetic_speaker(speaker_id)
        self.meta = {}

    def get_conversation(self):
        return None


class DeferralDecisionPolicy(DecisionPolicy):
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
    :param reuse_cached_simulations: if True (default), simulations already present on the
        current utterance's metadata under ``simulated_reply_attribute_name`` are reused
        instead of re-invoking the simulator. similarly, cached simulation scores under
        ``sim_replies_forecast_probs_attribute_name`` are reused when they align with the
        reused simulations, skipping re-scoring. set to False to force regeneration.
    """

    def __init__(
        self,
        simulator,
        threshold,
        tau: int = 5,
        num_simulations: int = 10,
        store_simulations: bool = False,
        simulated_reply_attribute_name: str = "sim_replies",
        sim_replies_forecast_probs_attribute_name: str = "sim_replies_forecast_probs",
        reuse_cached_simulations: bool = True,
        forecast_prob_attribute_name: str = "forecast_prob",
        reuse_cached_forecast_probs: bool = True,
    ):
        super().__init__(
            forecast_prob_attribute_name=forecast_prob_attribute_name,
            reuse_cached_forecast_probs=reuse_cached_forecast_probs,
        )
        self.simulator = simulator
        self.threshold = float(threshold)
        self.tau = int(tau)
        n = int(num_simulations)
        if simulator is not None and hasattr(simulator, "get_num_simulations"):
            n = min(n, int(simulator.get_num_simulations()))
        self.num_simulations = n
        self.store_simulations = store_simulations
        self.simulated_reply_attribute_name = simulated_reply_attribute_name
        self.sim_replies_forecast_probs_attribute_name = sim_replies_forecast_probs_attribute_name
        self.reuse_cached_simulations = bool(reuse_cached_simulations)
        self._sim_cache: dict = {}
        self._sim_score_cache: dict = {}

    def _get_utt_meta(self, context):
        # unified accessor so both real Utterance and _synthetic_utterance work; returns {} if absent.
        return getattr(context.current_utterance, "meta", {}) or {}

    def _get_cached_simulations(self, context) -> Optional[List[str]]:
        # returns cached simulation strings for this utterance if available on its metadata, else None.
        if not self.reuse_cached_simulations:
            return None
        meta = self._get_utt_meta(context)
        cached = meta.get(self.simulated_reply_attribute_name)
        if cached is None:
            return None
        cached_list = list(cached)
        if len(cached_list) == 0:
            return None
        return cached_list[: self.num_simulations]

    def _get_cached_simulation_scores(
        self, context, num_expected: int
    ) -> Optional[List[float]]:
        # returns cached per-simulation scores aligned with reused simulations, else None.
        if not self.reuse_cached_simulations or num_expected == 0:
            return None
        meta = self._get_utt_meta(context)
        cached = meta.get(self.sim_replies_forecast_probs_attribute_name)
        if cached is None:
            return None
        cached_list = list(cached)
        # if the cached scores are shorter than the reused simulations, fall back to re-scoring
        # rather than silently mixing cached and fresh scores.
        if len(cached_list) < num_expected:
            return None
        return [float(x) for x in cached_list[:num_expected]]

    def get_simulations(self, context, simulator=None) -> List[str]:
        # fast path: reuse pre-computed simulations from utterance metadata when present.
        cached = self._get_cached_simulations(context)
        if cached is not None:
            return cached
        sim = simulator if simulator is not None else self.simulator
        if sim is None or not hasattr(sim, "transform"):
            return []
        sims = sim.transform(iter([context]))
        utt_id = context.current_utterance.id
        if utt_id not in sims.index or sims.shape[1] == 0:
            return []
        col_name = sims.columns[0]
        return list(sims.loc[utt_id][col_name])[: self.num_simulations]

    def _build_simulated_context(self, context, simulation_text: str, simulation_idx: int):
        current_utt = context.current_utterance
        synthetic_utt = _synthetic_utterance(
            text=simulation_text,
            utterance_id=f"{current_utt.id}__sim_{simulation_idx}",
            speaker_id="",
        )
        new_context_utts = list(context.context) + [synthetic_utt]
        context_cls = context.__class__
        return context_cls(
            context=new_context_utts,
            current_utterance=synthetic_utt,
            future_context=None,
            conversation_id=context.conversation_id,
        )

    def _decision_score(self, context, score_fn: Callable):
        current_score = self._score(context, score_fn)
        simulations = self.get_simulations(context)
        # the get_simulations method actively checks if cached simulations exist

        # fast path: if cached per-simulation scores align with the reused simulations,
        # skip re-scoring the simulated contexts entirely.
        cached_scores = self._get_cached_simulation_scores(context, len(simulations))
        if cached_scores is not None:
            simulation_scores = cached_scores
        else:
            simulation_scores = []
            for idx, sim_text in enumerate(simulations):
                sim_context = self._build_simulated_context(context, sim_text, idx)
                # synthetic utterances have empty meta so _score falls through to score_fn.
                simulation_scores.append(self._score(sim_context, score_fn))
        if self.store_simulations and simulations:
            utt_id = context.current_utterance.id
            self._sim_cache[utt_id] = simulations
            self._sim_score_cache[utt_id] = simulation_scores
        return current_score, simulations, simulation_scores

    def decide(self, context, score_fn: Callable) -> Tuple[float, int, Optional[Dict[str, Any]]]:
        decision_score, simulations, simulation_scores = self._decision_score(context, score_fn)
        num_simulations_above_threshold = sum(1 for score in simulation_scores if score > self.threshold)
        num_simulations = len(simulations)

        num_calm = num_simulations - num_simulations_above_threshold
        defer = (num_calm > self.tau)
        return (
            decision_score,
            1 if decision_score > self.threshold and not defer else 0,
            {
                self.simulated_reply_attribute_name: simulations,
                self.sim_replies_forecast_probs_attribute_name: simulation_scores,
            },
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
