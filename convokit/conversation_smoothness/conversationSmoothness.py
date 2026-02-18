from convokit import Corpus, PolitenessStrategies, TextParser
from convokit.transformer import Transformer
from scipy.stats import kendalltau
from inspect import signature
from collections import deque


def calculate_politeness_score(polite_strat):
    """This function is meant to return a politeness score given a politness strategy map.

    Args:
        polite_strat (dict): the politeness strategies as generated from the PolitenessStrategies trasnformer

    Returns:
        float: represents the politeness score
    """
    politeness_mapping = {
        "feature_politeness_==Please==": 0.49,
        "feature_politeness_==Please_start==": -0.3,
        "feature_politeness_==HASHEDGE==": 0,
        "feature_politeness_==Indirect_(btw)==": 0.63,
        "feature_politeness_==Hedges==": 0.14,
        "feature_politeness_==Factuality==": -0.38,
        "feature_politeness_==Deference==": 0.78,
        "feature_politeness_==Gratitude==": 0.87,
        "feature_politeness_==Apologizing==": 0.36,
        "feature_politeness_==1st_person_pl.==": 0.08,
        "feature_politeness_==1st_person==": 0.08,
        "feature_politeness_==1st_person_start==": 0.12,
        "feature_politeness_==2nd_person==": 0.05,
        "feature_politeness_==2nd_person_start==": -0.3,
        "feature_politeness_==Indirect_(greeting)==": 0.43,
        "feature_politeness_==Direct_question==": -0.27,
        "feature_politeness_==Direct_start==": -0.43,
        "feature_politeness_==HASPOSITIVE==": 0.12,
        "feature_politeness_==HASNEGATIVE==": -0.13,
        "feature_politeness_==SUBJUNCTIVE==": 0,
        "feature_politeness_==INDICATIVE==": 0,
    }

    politeness = 0

    for key, value in polite_strat.items():
        politeness += politeness_mapping[key] * value

    return politeness


class ConversationSmoothness(Transformer):
    """
    A simple transformer to label a Corpus on a conversation level

    Will only work on the Candor corpus
    :param metric: a string that chooses which computation method to use to compute smoothness. It will either be 'ratio', 'decline', or 'tone'. By default, it is 'ratio'
    :param end_len: the number of utterances to take from the conversation end (must be an even number)
    :param output_field: field for writing the computed output in metadata. Will default to write to conversation metadata with name 'smoothness'.
    :param input_filter: a boolean function of signature `input_filter(conversation, aux_input)`. attributes will only be computed for conversations where `input_filter` returns `True`. By default, will always return `True`, meaning that attributes will be computed for all utterances.
    :param verbosity: frequency at which to print status messages when computing attributes.

    (previous params for the object in the demo for reference, you can ignore)
    obj_type: type of Corpus object to calculate: 'conversation', 'speaker', or 'utterance', default to be 'utterance'
    input_field: Input fields from every utterance object. Will default to reading 'utt.text'. If a string is provided, than consider metadata with field name input_field.
    output_field: field for writing the computed output in metadata. Will default to write to utterance metadata with name 'capitalization'.
    input_filter: a boolean function of signature `input_filter(utterance, aux_input)`. attributes will only be computed for utterances where `input_filter` returns `True`. By default, will always return `True`, meaning that attributes will be computed for all utterances.
    verbosity: frequency at which to print status messages when computing attributes.
    """

    def __init__(
        self,
        metric="ratio",
        end_len=12,
        output_field="smoothness",
        input_filter=None,
        verbosity=200,
    ):
        if input_filter:
            if len(signature(input_filter).parameters) == 1:
                self.input_filter = lambda convo: input_filter(convo)
            else:
                self.input_filter = input_filter
        else:
            self.input_filter = lambda convo: True
        self.metric = metric
        self.end_len = end_len
        self.output_field = output_field
        self.verbosity = verbosity
        self.ps = PolitenessStrategies(verbose=0)
        self.parser = TextParser(verbosity=0)

    def _print_output(self, i):
        return (self.verbosity > 0) and (i > 0) and (i % self.verbosity == 0)

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Takes the  and annotate in the corresponding object metadata fields.

        :param corpus: Corpus
        :return: the corpus
        """

        total = len(list(corpus.iter_conversations()))

        for idx, convo in enumerate(corpus.iter_conversations()):
            if self._print_output(idx):
                print(f"%03d/%03d conversations processed" % (idx, total))

            if not self.input_filter(convo):
                continue

            last_utts = convo.get_utterance_ids()[-self.end_len :]
            len_last_utts = len(last_utts)

            # for the calculation
            calc = 0

            # difference for pairs in decline metric
            diffs = deque([])

            # has pos and has neg freqs
            has_pos = [0, 0]
            has_neg = [0, 0]

            politeness1 = 0
            politeness2 = 0

            # for tau decline metric
            canonical_ordering = []
            ordering = []

            # for metric calculations

            # new for loop here for the ratio metric
            if self.metric == "ratio":
                for i in range(len_last_utts - 1):
                    utt = corpus.get_utterance(last_utts[i])
                    next_utt = corpus.get_utterance(last_utts[i + 1])
                    utt1len, utt2len = utt.meta["delta"], next_utt.meta["delta"]
                    ratio = utt1len / utt2len if utt1len <= utt2len else utt2len / utt1len
                    calc += ratio

            # here are the other loops
            for i in range(len_last_utts // 2):
                utt = corpus.get_utterance(last_utts[i])
                paired_utt = corpus.get_utterance(last_utts[i + 1])

                if self.metric == "ratio_old":
                    # old metric
                    # get your pairs (only look at even numbers)
                    utt1len, utt2len = utt.meta["delta"], paired_utt.meta["delta"]
                    ratio = utt1len / utt2len if utt1len <= utt2len else utt2len / utt1len
                    calc += ratio

                elif self.metric == "decline":
                    # append the differences
                    diffs.append(abs(utt.meta["delta"] - paired_utt.meta["delta"]))
                    # calculate the difference of differences when possible
                    if len(diffs) == 2:
                        # remove last element and calculate the most recent element
                        popped = diffs.popleft()
                        calc += abs(popped - diffs[0])

                    # NEW DECLINE METRIC
                    ordering.append(
                        (abs(utt.meta["delta"] - paired_utt.meta["delta"]), (utt.id, paired_utt.id))
                    )
                    canonical_ordering.append((utt.id, paired_utt.id))

                elif self.metric == "tone":
                    # old metric

                    # run the text transformer for this utterance
                    self.parser.transform_utterance(utt)
                    self.parser.transform_utterance(paired_utt)
                    # run politeness on here
                    utt_polite = self.ps.transform_utterance(utt, markers=True)
                    paired_utt_polite = self.ps.transform_utterance(paired_utt, markers=True)
                    # find the ratios

                    has_pos[0] += utt_polite.meta["politeness_strategies"][
                        "feature_politeness_==HASPOSITIVE=="
                    ]
                    has_pos[1] += paired_utt_polite.meta["politeness_strategies"][
                        "feature_politeness_==HASPOSITIVE=="
                    ]
                    has_neg[0] += utt_polite.meta["politeness_strategies"][
                        "feature_politeness_==HASNEGATIVE=="
                    ]
                    has_neg[1] += paired_utt_polite.meta["politeness_strategies"][
                        "feature_politeness_==HASNEGATIVE=="
                    ]
                    # Returns (1) absolute difference between Has Positive prevalences and (2) absolute difference between Has Negative prevalences

                    # difference in politeness score
                    politeness1 = calculate_politeness_score(
                        utt_polite.meta["politeness_strategies"]
                    )
                    politeness2 = calculate_politeness_score(
                        paired_utt_polite.meta["politeness_strategies"]
                    )

                else:
                    raise KeyError("metric must be ratio, ratio_old, decline, or tone, ")

            if self.metric == "decline":
                ordering = [pair for ratio, pair in sorted(ordering, reverse=True)]
                tau, _ = kendalltau(ordering, canonical_ordering)
                calc = tau

            if self.metric == "tone":
                pos_diff = abs(
                    has_pos[0] / (len_last_utts // 2) - has_pos[1] / (len_last_utts // 2)
                )
                neg_diff = abs(
                    has_neg[0] / (len_last_utts // 2) - has_neg[1] / (len_last_utts // 2)
                )
                convo.add_meta(f"{self.output_field}_{self.metric}_pos_count1", has_pos[0])
                convo.add_meta(f"{self.output_field}_{self.metric}_neg_count1", has_neg[0])
                convo.add_meta(f"{self.output_field}_{self.metric}_pos_count2", has_pos[1])
                convo.add_meta(f"{self.output_field}_{self.metric}_neg_count2", has_neg[1])
                convo.add_meta(f"{self.output_field}_{self.metric}_pos", pos_diff)
                convo.add_meta(f"{self.output_field}_{self.metric}_neg", neg_diff)
                convo.add_meta(f"{self.output_field}_{self.metric}_politeness1", politeness1)
                convo.add_meta(f"{self.output_field}_{self.metric}_politeness2", politeness2)
                convo.add_meta(
                    f"{self.output_field}_{self.metric}_politeness_diff",
                    abs(politeness1 / (len_last_utts // 2) - politeness2 / (len_last_utts // 2)),
                )
            else:
                # take the average of all summed components
                calc /= (
                    (len_last_utts - 1)
                    if self.metric == "ratio"
                    else len_last_utts // 2
                    if self.metric == "ratio_old"
                    else 1
                )
                # do the catching and add to output_field
                convo.add_meta(f"{self.output_field}_{self.metric}", calc)

            last_utt_time_delta = (
                corpus.get_utterance(last_utts[-1]).meta["stop"]
                - corpus.get_utterance(last_utts[0]).meta["start"]
            )
            convo.add_meta(f"{self.output_field}_last_utts_time", last_utt_time_delta)

        return corpus
