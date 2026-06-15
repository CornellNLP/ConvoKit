from abc import ABC, abstractmethod
from itertools import tee
from typing import Callable

import json
import os
import shutil

from convokit.decisionpolicy import ThresholdDecisionPolicy


class ForecasterModel(ABC):
    """
    An abstract class defining an interface that Forecaster can call into to invoke a conversational forecasting algorithm.
    The “contract” between Forecaster and ForecasterModel means that ForecasterModel can expect to receive conversational data
    in a consistent format, defined above.
    """

    def __init__(self, decision_policy=None, **kwargs):
        self._labeler = None
        self._forecast_prob_attribute_name = "forecast_prob"
        self._decision_policy = decision_policy or ThresholdDecisionPolicy()

    @property
    def labeler(self):
        return self._labeler

    @labeler.setter
    def labeler(self, value: Callable):
        self._labeler = value
        if self._decision_policy is not None:
            self._decision_policy.labeler = value

    @property
    def forecast_prob_attribute_name(self) -> str:
        return self._forecast_prob_attribute_name

    @forecast_prob_attribute_name.setter
    def forecast_prob_attribute_name(self, value: str):
        # keeps the decision policy's cache key aligned with the forecaster's
        # meta field so policies can reuse previously written forecast probs.
        self._forecast_prob_attribute_name = value
        if self._decision_policy is not None:
            self._decision_policy.forecast_prob_attribute_name = value

    @property
    def decision_policy(self):
        return self._decision_policy

    @decision_policy.setter
    def decision_policy(self, value):
        self._decision_policy = value
        if self._decision_policy is not None:
            self._decision_policy.labeler = self._labeler
            self._decision_policy.forecast_prob_attribute_name = self._forecast_prob_attribute_name

    @abstractmethod
    def fit(self, contexts, val_contexts=None):
        """
        Train this conversational forecasting model on the given data by fitting
        both the belief estimator and the decision policy.

        :param contexts: an iterator over context tuples
        :param val_contexts: an optional second iterator over context tuples to be used as a separate held-out validation set. Concrete ForecasterModel implementations may choose to ignore this, or conversely even enforce its presence.
        """
        pass

    @abstractmethod
    def fit_belief_estimator(self, contexts, val_contexts=None):
        """
        Fit only the belief estimator component that produces continuous scores.
        """
        pass

    def fit_decision_policy(self, contexts, val_contexts=None, score_fn: Callable = None):
        """
        Fit only the decision policy component.
        """
        if self.decision_policy is not None:
            if score_fn is None:
                score_fn = self.score
            fit_result = self.decision_policy.fit(
                contexts=contexts, val_contexts=val_contexts, score_fn=score_fn
            )
            self._json_dump_fit_result(fit_result)
            return fit_result
        return None

    def _json_dump_fit_result(self, fit_result):
        if not isinstance(fit_result, dict):
            return

        output_dir = getattr(getattr(self, "config", None), "output_dir", None)
        if output_dir is None:
            return

        config_file = os.path.join(output_dir, "dev_config.json")
        existing_config = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as infile:
                    existing_config = json.load(infile)
            except (json.JSONDecodeError, OSError):
                existing_config = {}

        if "best_checkpoint" in fit_result:
            existing_config["best_checkpoint"] = fit_result["best_checkpoint"]
        if "best_threshold" in fit_result:
            existing_config["best_threshold"] = float(fit_result["best_threshold"])
        if "best_val_accuracy" in fit_result:
            existing_config["best_val_accuracy"] = float(fit_result["best_val_accuracy"])

        with open(config_file, "w") as outfile:
            json.dump(existing_config, outfile, indent=4)

    def get_checkpoints(self):
        return []

    def load_checkpoint(self, checkpoint_name):
        raise NotImplementedError("checkpoint loading is not implemented for this model")

    def finalize_best_checkpoint_selection(
        self, best_checkpoint, best_config, val_contexts=None, score_fn: Callable = None
    ):
        if best_checkpoint is None:
            return
        self._cleanup_checkpoints(best_checkpoint)
        self._save_tokenizer_checkpoint(best_checkpoint)

    def _cleanup_checkpoints(self, best_checkpoint):
        output_dir = getattr(getattr(self, "config", None), "output_dir", None)
        if output_dir is None or best_checkpoint is None:
            return

        for root, _, _ in os.walk(output_dir):
            if ("checkpoint" in root) and (best_checkpoint not in root):
                print(f"deleting: {root}")
                shutil.rmtree(root)

    def _save_tokenizer_checkpoint(self, best_checkpoint):
        tokenizer = getattr(self, "tokenizer", None)
        output_dir = getattr(getattr(self, "config", None), "output_dir", None)
        if (
            tokenizer is None
            or output_dir is None
            or best_checkpoint is None
            or not hasattr(tokenizer, "save_pretrained")
        ):
            return
        tokenizer.save_pretrained(os.path.join(output_dir, best_checkpoint))

    @abstractmethod
    def score(self, context) -> float:
        """
        Produce the belief estimator score for a context.
        """
        pass

    def _predict(self, context):
        """
        Return both belief score and policy action for a context.

        This method is deprecated in favor of using the self.decision_policy.decide method.
        """
        utt_score, utt_pred, _ = self._parse_decision_result(
            self.decision_policy.decide(context, self.score)
        )
        return utt_score, utt_pred

    def _parse_decision_result(self, result):
        if len(result) == 2:
            utt_score, utt_pred = result
            utt_metadata = {}
        elif len(result) == 3:
            utt_score, utt_pred, utt_metadata = result
            if utt_metadata is None:
                utt_metadata = {}
        else:
            raise ValueError(
                "decision_policy.decide() must return (utt_score, utt_pred) "
                "or (utt_score, utt_pred, metadata_dict)"
            )
        return float(utt_score), int(utt_pred), utt_metadata

    @abstractmethod
    def transform(self, contexts, forecast_attribute_name, forecast_prob_attribute_name):
        """
        Apply this trained conversational forecasting model to the given data, and return its forecasts
        in the form of a DataFrame indexed by (current) utterance ID

        :param contexts: an iterator over context tuples

        :return: a Pandas DataFrame, with one row for each context, indexed by the ID of that context's current utterance. Contains two columns, one with raw probabilities named according to forecast_prob_attribute_name, and one with discretized (binary) forecasts named according to forecast_attribute_name. Subclass implementations of ForecasterModel MUST adhere to this return value specification!
        """
        pass
