from collections import defaultdict
from typing import Callable, Tuple, List, Dict, Optional, Collection, Union
import copy

import pkg_resources

from convokit.model import Corpus, Speaker, Utterance
from convokit.transformer import Transformer
from convokit.util import deprecation
from .coordinationScore import CoordinationScore, CoordinationWordCategories


class Coordination(Transformer):
    """Linguistic coordination is a measure of the propensity of a
    speaker to echo the language of another speaker in a
    conversation, as defined in "Echoes of Power: Language Effects
    and Power Differences in Social Interaction"
    (http://www.cs.cornell.edu/~cristian/Echoes_of_power.html)

    This Transformer encapsulates computation of coordination-based features
    for a particular corpus.

    Coordination is a measure of power differences between speakers in a
    conversation, based on the propensity of a speaker to echo the same
    function words used by another speaker in a conversation. It is defined in
    Danescu-Niculescu-Mizil et al's "Echoes of Power: Language Effects and Power
    Differences in Social Interaction".

    This transformer contains various functions to measure coordination on
    different conversational scales. Calling `transform()` will annotate each
    speaker in the corpus with their coordination to all speakers they directly
    reply to. The `summarize()` function is a convenience method that computes
    aggregated coordination scores between two groups of speakers.

    Note: labeling method is slightly different from that used in the paper --
    we no longer match words occurring in the middle of other words and that
    immediately follow an apostrophe. Notably, we no longer separately
    count the "all" in "y'all."

    :param coordination_attribute_name: metadata attribute name to store coordination scores during the `transform()` step.
    :param speaker_thresh: Thresholds based on minimum number of times the speaker uses each coordination marker. Speakers that do not meet the threshold are excluded from computation for a given marker.
    :param target_thresh: Thresholds based on minimum number of times the target uses each coordination marker. Targets that do not meet the threshold are excluded from computation for a given marker.
    :param utterances_thresh: Thresholds based on the minimum number of utterances for each speaker. Speakers that do not meet the threshold are excluded from computation for a given marker.
    :param speaker_thresh_indiv: Like `speaker_thresh` but only considers the utterances between a speaker and a single target; thresholds whether the utterances for a single target should be considered for a particular speaker.
    :param target_thresh_indiv: Like `target_thresh` but thresholds whether a single target's utterances should be considered for a particular speaker.
    :param utterances_thresh_indiv: Like `utterances_thresh` but thresholds whether a single target's utterances should be considered for a particular speaker.
    """

    def __init__(
        self,
        coordination_attribute_name: str = "coord",
        speaker_thresh: int = 0,
        target_thresh: int = 3,
        utterances_thresh: int = 0,
        speaker_thresh_indiv: int = 0,
        target_thresh_indiv: int = 0,
        utterances_thresh_indiv: int = 0,
        utterance_thresh_func: Optional[Callable[[Tuple[Utterance, Utterance]], bool]] = None,
    ):
        if utterance_thresh_func is not None:
            deprecation(
                "Coordination's utterance_thresh_func parameter",
                "speaker, target and utterance threshold parameters",
            )
        self.speaker_thresh = speaker_thresh
        self.target_thresh = target_thresh
        self.utterances_thresh = utterances_thresh
        self.speaker_thresh_indiv = speaker_thresh_indiv
        self.target_thresh_indiv = target_thresh_indiv
        self.utterances_thresh_indiv = utterances_thresh_indiv
        self.utterance_thresh_func = utterance_thresh_func
        self.corpus = None
        self.precomputed = False
        self.coordination_attribute_name = coordination_attribute_name

    def fit(self, corpus: Corpus, y=None):
        """Learn coordination information for the given corpus."""
        self.corpus = corpus

        # if not self.precomputed:
        self._compute_liwc_reverse_dict()
        self._annot_liwc_cats(corpus)
        self.precomputed = True

    def precompute(self, corpus: Corpus):
        """Deprecated. Use fit() instead."""
        deprecation("Coordination's precompute function", "fit")
        self.fit(corpus)

    def transform(self, corpus: Corpus) -> Corpus:
        """Generate coordination scores for the corpus you called fit on.

        Each speaker's coordination attribute will be a dictionary from targets
        to coordination scores between that speaker and target."""
        if corpus != self.corpus:
            raise Exception("Coordination: must fit and transform on same corpus")
        if not self.precomputed:
            raise Exception("Must fit before calling transform")

        pair_scores = self.pairwise_scores(
            corpus,
            corpus.speaking_pairs(),
            speaker_thresh=self.speaker_thresh,
            target_thresh=self.target_thresh,
            utterances_thresh=self.utterances_thresh,
            speaker_thresh_indiv=self.speaker_thresh_indiv,
            target_thresh_indiv=self.target_thresh_indiv,
            utterances_thresh_indiv=self.utterances_thresh_indiv,
            utterance_thresh_func=self.utterance_thresh_func,
        )

        # Keep record of all score update for all (speakers, target) pairs to avoid redundant operations
        todo = {}

        for (speaker, target), score in pair_scores.items():
            if self.coordination_attribute_name not in speaker.meta:
                speaker.meta[self.coordination_attribute_name] = {}
            key = (speaker, target.id)
            todo.update({key: score})

        for key, score in todo.items():
            speaker = key[0]
            target = key[1]
            # For avoiding mutability for the sake of DB corpus
            temp_dict = copy.deepcopy(speaker.meta[self.coordination_attribute_name])
            temp_dict[target] = score
            speaker.meta[self.coordination_attribute_name] = temp_dict
            assert isinstance(speaker, Speaker)

        return corpus

    def fit_transform(self, corpus: Corpus, y=None) -> Corpus:
        self.fit(corpus)
        return self.transform(corpus)

    def summarize(
        self,
        corpus: Corpus,
        speaker_selector: Callable[[Speaker], bool] = lambda obj: True,
        target_selector: Callable[[Speaker], bool] = lambda obj: True,
        focus: str = "speakers",
        summary_report: bool = False,
        speaker_thresh: Optional[int] = None,
        target_thresh: Optional[int] = None,
        utterances_thresh: Optional[int] = None,
        speaker_thresh_indiv: Optional[int] = None,
        target_thresh_indiv: Optional[int] = None,
        utterances_thresh_indiv: Optional[int] = None,
        utterance_thresh_func: Optional[Callable[[Tuple[Utterance, Utterance]], bool]] = None,
        split_by_attribs: Optional[List[str]] = None,
        speaker_utterance_selector: Callable[
            [Tuple[Utterance, Utterance]], bool
        ] = lambda utt1, utt2: True,
        target_utterance_selector: Callable[
            [Tuple[Utterance, Utterance]], bool
        ] = lambda utt1, utt2: True,
        speaker_attribs: Optional[Dict] = None,
        target_attribs: Optional[Dict] = None,
    ) -> CoordinationScore:
        """Computes a summary of the coordination scores by giving an
        aggregated score between two groups of speakers.

        The threshold parameters may be used to override the thresholds set in
        the constructor. If a threshold parameter is not explicitly set, it
        will take on the value provided in the constructor.

        Additionally, this method provides optional options to tweak the method
        by which scores are aggregated. The `focus` parameter is used to
        aggregate scores relative to either speakers or targets.
        `split_by_attribs`, `speaker_attribs` and `target_attribs` are used to
        specify whether to summarize scores for particular subgroups of
        speakers or targets.

        :param corpus: Corpus to compute scores on
        :param speaker_selector: A lambda function that takes a speaker and
            returns True or False depending on whether the speaker should be
            included in the group of speakers we want to compute scores for.
        :param target_selector: A lambda function that takes a speaker and
            returns True or False depending on whether the speaker should be
            included in the group of targets.
        :param focus: Either "speakers" or "targets". If "speakers", treat the
            set of targets for a particular speaker as a single person (i.e.
            concatenate all of their utterances); the returned dictionary will
            have speakers as keys. If "targets", treat the set of
            speakers for a particular target as a single person; the returned
            dictionary will have targets as keys. See the example notebook for
            typical usage.
        :param summary_report: if True, return a dictionary of key global
        coordination statistics. Otherwise, return a dictionary of speaker
        scores.
        :param speaker_thresh: Thresholds based on minimum number of times the speaker uses each coordination marker.
        :param target_thresh: Thresholds based on minimum number of times the target uses each coordination marker.
        :param utterances_thresh: Thresholds based on the minimum number of utterances for each speaker.
        :param speaker_thresh_indiv: Like `speaker_thresh` but only considers the utterances between a speaker and a single target; thresholds whether the utterances for a single target should be considered for a particular speaker.
        :param target_thresh_indiv: Like `target_thresh` but thresholds whether a single target's utterances should be considered for a particular speaker.
        :param utterances_thresh_indiv: Like `utterances_thresh` but thresholds whether a single target's utterances should be considered for a particular speaker.
        :param utterance_thresh_func: Optional utterance-level threshold function that takes in a speaker `Utterance` and the `Utterance` the speaker replied to, and returns a `bool` corresponding to whether or not to include the utterance in scoring.
        :param split_by_attribs: Utterance meta attributes to split speakers by when tallying coordination (e.g. in supreme court transcripts, you may want to treat the same lawyer as a different person across different cases --- see coordination examples)
        :param speaker_utterance_selector: A lambda function that takes a
        speaker and target utterance pair and returns True or
        False for whether the speaker utterance should be considered. Useful for
        filtering the set of utterances before processing.
        :param target_utterance_selector: A lambda function that takes a
        speaker and target utterance pair and returns True or
        False for whether the target utterance should be considered. Useful for
        filtering the set of utterances before processing.

        :return: If summary_report=True, returns a :class:`CoordinationScore`
        object corresponding to the coordination scores for each speaker. This
        object is a dictionary mapping each speaker to its aggregated
        coordination score to all speakers in the opposite group. If
        summary_report=False, returns a dictionary of summary statistics for
        the coordination scores across each marker, the overall coordination
        score under each of three aggregation methods (described in the paper),
        and the count (sample size) for the statistics under the various
        aggregation methods.
        """
        if corpus != self.corpus:
            raise Exception("Coordination: must fit and score on same corpus")
        if not self.precomputed:
            raise Exception("Must fit before calling score")

        if split_by_attribs is None:
            split_by_attribs = []
        if speaker_attribs is None:
            speaker_attribs = dict()
        else:
            deprecation("Coordination's speaker_attribs parameter", "speaker_utterance_selector")
            speaker_utterance_selector = lambda utt, _: (
                Coordination._utterance_has_attribs(utt, speaker_attribs)
            )
        if target_attribs is None:
            target_attribs = dict()
        else:
            deprecation("Coordination's target_attribs parameter", "target_utterance_selector")
            target_utterance_selector = lambda _, utt: (
                Coordination._utterance_has_attribs(utt, target_attribs)
            )

        if speaker_thresh is None:
            speaker_thresh = self.speaker_thresh
        if target_thresh is None:
            target_thresh = self.target_thresh
        if utterances_thresh is None:
            utterances_thresh = self.utterances_thresh
        if speaker_thresh_indiv is None:
            speaker_thresh_indiv = self.speaker_thresh_indiv
        if target_thresh_indiv is None:
            target_thresh_indiv = self.target_thresh_indiv
        if utterances_thresh_indiv is None:
            utterances_thresh_indiv = self.utterances_thresh_indiv

        speakers = set(corpus.iter_speakers(speaker_selector))
        group = set(corpus.iter_speakers(target_selector))

        utterances = []
        for utt in corpus.iter_utterances():
            speaker = utt.speaker
            if speaker in speakers:
                if utt.reply_to is not None:
                    reply_to = corpus.get_utterance(utt.reply_to)
                    target = reply_to.speaker
                    if target in group:
                        utterances.append(utt)
        scores = self._scores_over_utterances(
            corpus,
            speakers,
            utterances,
            speaker_thresh,
            target_thresh,
            utterances_thresh,
            speaker_thresh_indiv,
            target_thresh_indiv,
            utterances_thresh_indiv,
            utterance_thresh_func,
            focus,
            split_by_attribs,
            speaker_utterance_selector,
            target_utterance_selector,
        )

        if summary_report:
            return self._summarize_score_report(scores)
        else:
            return scores

    def _summarize_score_report(self, scores: CoordinationScore):
        marker_a1 = scores.averages_by_marker(strict_thresh=True)
        marker = scores.averages_by_marker()
        agg1 = scores.aggregate(method=1)
        agg2 = scores.aggregate(method=2)
        agg3 = scores.aggregate(method=3)
        return {
            "marker_agg1": marker_a1,
            "marker_agg2": marker,
            "marker_agg3": marker,
            "agg1": agg1,
            "agg2": agg2,
            "agg3": agg3,
            "count_agg1": len([s for s in scores if len(scores[s]) == 8]),
            "count_agg2": len(scores),
            "count_agg3": len(scores),
        }

    def pairwise_scores(
        self,
        corpus: Corpus,
        pairs: Collection[Tuple[Union[Speaker, str], Union[Speaker, str]]],
        speaker_thresh: int = 0,
        target_thresh: int = 3,
        utterances_thresh: int = 0,
        speaker_thresh_indiv: int = 0,
        target_thresh_indiv: int = 0,
        utterances_thresh_indiv: int = 0,
        utterance_thresh_func: Optional[Callable[[Tuple[Utterance, Utterance]], bool]] = None,
    ) -> CoordinationScore:
        """Computes all pairwise coordination scores given a collection of
        (speaker, target) pairs.

        :param corpus: Corpus to compute scores on
        :param pairs: collection of (speaker id, target id) pairs
        :type pairs: Collection

        Also accepted: all threshold arguments accepted by :func:`score()`.

        :return: A :class:`CoordinationScore` object corresponding to the
            coordination scores for each (speaker, target) pair.
        """
        if corpus != self.corpus:
            raise Exception("Coordination: must fit and score on same corpus")
        if not self.precomputed:
            raise Exception("Must fit before calling score")

        pairs = set(pairs)
        any_speaker = next(iter(pairs))[0]
        if isinstance(any_speaker, str):
            pairs_utts = corpus.directed_pairwise_exchanges(
                lambda x, y: (x.id, y.id) in pairs, speaker_ids_only=True
            )
        else:
            pairs_utts = corpus.directed_pairwise_exchanges(
                lambda x, y: (x, y) in pairs, speaker_ids_only=False
            )
        all_scores = CoordinationScore()
        for (speaker, target), utterances in pairs_utts.items():
            scores = self._scores_over_utterances(
                corpus,
                [speaker],
                utterances,
                speaker_thresh,
                target_thresh,
                utterances_thresh,
                speaker_thresh_indiv,
                target_thresh_indiv,
                utterances_thresh_indiv,
                utterance_thresh_func,
            )
            if len(scores) > 0:  # scores.values() will be length 0 or 1
                all_scores[speaker, target] = list(scores.values())[0]
        return all_scores

    def score_report(self, corpus: Corpus, scores: CoordinationScore):
        """Deprecated. Use `summarize()` instead."""
        deprecation("Coordination's score_report()", "Coordination's summarize()")
        if corpus != self.corpus:
            raise Exception("Coordination: must fit and score on same corpus")
        if not self.precomputed:
            raise Exception("Must fit before calling score")

        marker_a1 = scores.averages_by_marker(strict_thresh=True)
        marker = scores.averages_by_marker()
        agg1 = scores.aggregate(method=1)
        agg2 = scores.aggregate(method=2)
        agg3 = scores.aggregate(method=3)
        return marker_a1, marker, agg1, agg2, agg3

    # helper functions
    def _compute_liwc_reverse_dict(self) -> None:
        with open(
            pkg_resources.resource_filename("convokit", "data/coord-liwc-patterns.txt"), "r"
        ) as f:
            all_words = []
            for line in f:
                cat, pat = line.strip().split("\t")
                # if cat == "auxverb": print(cat, pat)
                # use "#" to mark word boundary
                words = pat.replace("\\b", "#").split("|")
                all_words += [(w[1:], cat) for w in words]
            self.liwc_trie = Coordination.make_trie(all_words)

    @staticmethod
    def make_trie(words) -> Dict:
        root = {}
        for word, cat in words:
            cur = root
            for c in word:
                cur = cur.setdefault(c, {})
            if "$" not in cur:  # use "$" as end-of-word symbol
                cur["$"] = {cat}
            else:
                cur["$"].add(cat)
        return root

    def _annot_liwc_cats(self, corpus) -> None:
        # add liwc_categories field to each utterance
        word_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_")
        for utt in corpus.iter_utterances():
            cats = set()
            last = None
            cur = None
            text = utt.text.lower() + " "
            # if "'" in text: print(text)
            for i, c in enumerate(text):
                # slightly different from regex: won't match word after an
                #   apostrophe unless the apostrophe starts the word
                #   -- avoids false positives
                if last not in word_chars and c in word_chars and (last != "'" or not cur):
                    cur = self.liwc_trie
                if cur:
                    if c in cur and c != "#" and c != "$":
                        if c not in word_chars:
                            if "#" in cur and "$" in cur["#"]:
                                cats |= cur["#"]["$"]  # finished current word
                        cur = cur[c]
                    elif c not in word_chars and last in word_chars and "#" in cur:
                        cur = cur["#"]
                    else:
                        cur = None
                if cur and "$" in cur:
                    cats |= cur["$"]
                last = c
            utt.meta["liwc-categories"] = cats

    @staticmethod
    def _annot_speaker(speaker: Speaker, utt: Utterance, split_by_attribs):
        return (
            speaker,
            tuple(
                [utt.meta[attrib] if attrib in utt.meta else None for attrib in split_by_attribs]
            ),
        )

    @staticmethod
    def _utterance_has_attribs(utterance, desired_attribs) -> bool:
        for attrib, attrib_val in desired_attribs.items():
            if utterance.meta[attrib] != attrib_val:
                return False
        return True

    def _scores_over_utterances(
        self,
        corpus: Corpus,
        speakers: Collection[Union[Speaker, str]],
        utterances,
        speaker_thresh: int,
        target_thresh: int,
        utterances_thresh: int,
        speaker_thresh_indiv: int,
        target_thresh_indiv: int,
        utterances_thresh_indiv: int,
        utterance_thresh_func: Optional[Callable[[Utterance, Utterance], bool]] = None,
        focus: str = "speakers",
        split_by_attribs: Optional[List[str]] = None,
        speaker_utterance_selector: Callable[
            [Utterance, Utterance], bool
        ] = lambda utt1, utt2: True,
        target_utterance_selector: Callable[[Utterance, Utterance], bool] = lambda utt1, utt2: True,
    ) -> CoordinationScore:
        assert not isinstance(speakers, str)
        assert focus == "speakers" or focus == "targets"

        if split_by_attribs is None:
            split_by_attribs = []

        tally = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        cond_tally = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        cond_total = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        n_utterances = defaultdict(lambda: defaultdict(int))
        targets = defaultdict(set)
        real_speakers = set()
        for utt2 in utterances:
            if corpus.has_utterance(utt2.reply_to):
                speaker = utt2.speaker
                utt1 = corpus.get_utterance(utt2.reply_to)
                target = utt1.speaker
                if speaker == target:
                    continue
                speaker, target = Coordination._annot_speaker(
                    speaker, utt2, split_by_attribs
                ), Coordination._annot_speaker(target, utt1, split_by_attribs)

                speaker_filter = speaker_utterance_selector(utt2, utt1)
                target_filter = target_utterance_selector(utt2, utt1)

                if not speaker_filter or not target_filter:
                    continue

                real_speakers.add(speaker)

                if utterance_thresh_func is None or utterance_thresh_func(utt2, utt1):
                    if focus == "targets":
                        speaker, target = target, speaker
                    targets[speaker].add(target)
                    n_utterances[speaker][target] += 1
                    for cat in utt1.meta["liwc-categories"].union(utt2.meta["liwc-categories"]):
                        if cat in utt2.meta["liwc-categories"]:
                            tally[speaker][cat][target] += 1
                        if cat in utt1.meta["liwc-categories"]:
                            cond_total[speaker][cat][target] += 1
                            if cat in utt2.meta["liwc-categories"]:
                                cond_tally[speaker][cat][target] += 1

        out = CoordinationScore()
        if focus == "targets":
            speaker_thresh, target_thresh = target_thresh, speaker_thresh
            speaker_thresh_indiv, target_thresh_indiv = target_thresh_indiv, speaker_thresh_indiv
            real_speakers = list(targets.keys())

        for speaker in real_speakers:
            if speaker[0] not in speakers and focus != "targets":
                continue
            coord_w = {}  # coordination score wrt a category
            for cat in CoordinationWordCategories:
                threshed_cond_total = 0
                threshed_cond_tally = 0
                threshed_tally = 0
                threshed_n_utterances = 0
                for target in targets[speaker]:
                    if (
                        tally[speaker][cat][target] >= speaker_thresh_indiv
                        and cond_total[speaker][cat][target] >= target_thresh_indiv
                        and n_utterances[speaker][target] >= utterances_thresh_indiv
                    ):
                        threshed_cond_total += cond_total[speaker][cat][target]
                        threshed_cond_tally += cond_tally[speaker][cat][target]
                        threshed_tally += tally[speaker][cat][target]
                        threshed_n_utterances += n_utterances[speaker][target]
                if (
                    threshed_cond_total >= max(target_thresh, 1)
                    and threshed_tally >= speaker_thresh
                    and threshed_n_utterances >= max(utterances_thresh, 1)
                ):
                    coord_w[cat] = (
                        threshed_cond_tally / threshed_cond_total
                        - threshed_tally / threshed_n_utterances
                    )
            if len(coord_w) > 0:
                out[speaker if split_by_attribs else speaker[0]] = coord_w
        return out
