from collections import defaultdict
from typing import Dict, Optional, Hashable, Union

from convokit.model import Speaker

CoordinationWordCategories = [
    "article",
    "auxverb",
    "conj",
    "adverb",
    "ppron",
    "ipron",
    "preps",
    "quant",
]


class CoordinationScore(dict):
    """Encapsulates results of :func:`Coordination.score()` and
    :func:`Coordination.pairwise_scores()`.

    The simplest way to use it is as a dictionary mapping speakers to their
    scores:

    ::

        {
            speaker_1: { dictionary of scores by coordination marker },
            speaker_2: scores,
            ...
        }

    The keys are of the same types as the input: if a speaker_id was
    passed in, the corresponding key will be a speaker_id, etc. For pairwise
    scores, the keys are tuples (speaker, target).

    There are also helper functions for filtering scores or getting aggregate
    scores (though these can be more conveniently accessed through
    Coordination's `summarize()` method, using the `summary_report=True`
    option).
    """

    def scores_for_marker(self, marker: str) -> Dict[Union[Speaker, Hashable], float]:
        """Return a dictionary from speakers to their scores for just the given
        marker.

        :param marker: The marker to return scores for.
        """
        return {speaker: scores[marker] for speaker, scores in self.items()}

    def averages_by_speaker(self) -> Dict[Union[Speaker, Hashable], float]:
        """Return a dictionary from speakers to the average of each speaker's
        marker scores."""
        return {speaker: sum(scores.values()) / len(scores) for speaker, scores in self.items()}

    def averages_by_marker(self, strict_thresh: bool = False) -> Dict[str, float]:
        """Return a dictionary mapping markers to the average coordination score
        on that marker.

        :param strict_thresh: Whether to only include speakers with all 8 marker
            scores. This corresponds to Aggregate 1 in the Echoes paper (see
            top).
        """
        self.precompute_aggregates()
        return self.a1_avg_by_marker if strict_thresh else self.avg_by_marker

    def aggregate(self, method: int = 3) -> Optional[float]:
        """Return the aggregate coordination score.

        :param method: Can be 1, 2 or 3, corresponding to which aggregate method
            to use:

            - aggregate 1: average scores only over speakers with a score for each
              coordination marker.
            - aggregate 2: fill in missing scores for a speaker by using the group
              score for each missing marker. (assumes different people in a
              group coordinate the same way.)
            - aggregate 3: fill in missing scores for a speaker by using the
              average score over the markers we can compute coordination for for
              that speaker. (assumes a speaker coordinates the same way across
              different coordination markers.)
        """
        assert 1 <= method <= 3
        self.precompute_aggregates()
        if method == 1:
            return self.agg1
        elif method == 2:
            return self.agg2
        else:
            return self.agg3

    # helper functions
    def precompute_aggregates(self) -> None:
        a1_scores_by_marker = defaultdict(list)
        scores_by_marker = defaultdict(list)
        for speaker, scores in self.items():
            for cat, score in scores.items():
                scores_by_marker[cat].append(score)
                if len(scores) == len(CoordinationWordCategories):
                    a1_scores_by_marker[cat].append(score)
        do_agg2 = False
        if len(scores_by_marker) == len(CoordinationWordCategories):
            do_agg2 = True
            avg_score_by_marker = {
                cat: sum(scores) / len(scores) for cat, scores in scores_by_marker.items()
            }
        agg1s, agg2s, agg3s = [], [], []
        for speaker, scoredict in self.items():
            scores = list(scoredict.values())
            if len(scores) >= 1:
                avg = sum(scores) / len(scores)
                agg3s.append(avg)
                if len(scores) == len(CoordinationWordCategories):
                    agg1s.append(avg)
                if do_agg2:
                    for cat in avg_score_by_marker:
                        if cat not in scoredict:
                            scores.append(avg_score_by_marker[cat])
                    agg2s.append(sum(scores) / len(scores))
        agg1 = sum(agg1s) / len(agg1s) if agg1s else None
        agg2 = sum(agg2s) / len(agg2s) if agg2s else None
        agg3 = sum(agg3s) / len(agg3s) if agg3s else None

        a1_avg_by_marker = {
            cat: sum(scores) / len(scores) for cat, scores in a1_scores_by_marker.items()
        }
        avg_by_marker = {cat: sum(scores) / len(scores) for cat, scores in scores_by_marker.items()}
        self.precomputed_aggregates = True
        self.a1_avg_by_marker = a1_avg_by_marker
        self.avg_by_marker = avg_by_marker
        self.agg1 = agg1
        self.agg2 = agg2
        self.agg3 = agg3
