import math
import nltk
from convokit.transformer import Transformer, Corpus
from inspect import signature
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")


class PosNegIronyTransformer(Transformer):
    """
    A transformer to label all instances of the token "/s" (ironic utterances)
    with a score corresponding to whether or not it is positive or negative irony,
    based the degree of sentiment of the utterance and its replies.

    :param obj_type: type of Corpus object to calculate: 'conversation', 'speaker', or 'utterance', default to be 'utterance'
    :param input_field: Input fields from every utterance object. Will default to reading 'utt.text'. If a string is provided, than consider metadata with field name input_field.
    :param output_field: field for writing the computed output in metadata. Will default to write to utterance metadata with name 'capitalization'.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`. attributes will only be computed for utterances where `input_filter` returns `True`. By default, will always return `True`, meaning that attributes will be computed for all utterances.
    :param verbosity: frequency at which to print status messages when computing attributes.
    """

    def __init__(
        self,
        obj_type="utterance",
        output_field=None,
        input_field=None,
        input_filter=filter,
        verbosity=10000,
    ):
        if input_filter:
            if len(signature(input_filter).parameters) == 1:
                self.input_filter = lambda utt: input_filter(utt)
            else:
                self.input_filter = input_filter
        else:
            self.input_filter = lambda utt: True
        self.obj_type = obj_type
        self.input_field = input_field
        self.output_field = output_field
        self.verbosity = verbosity
        self.sia = SentimentIntensityAnalyzer()
        self.mean = 0
        self.sd = 0

    def _print_output(self, i):
        return (self.verbosity > 0) and (i > 0) and (i % self.verbosity == 0)

    def fit(self, corpus: Corpus) -> Corpus:
        corpus_sent = {}
        corpus_sent["pos"] = 0
        corpus_sent["neg"] = 0
        corpus_sent["neu"] = 0
        corpus_sent["compound"] = 0
        l = 0
        values = []

        whitelist(self, corpus)

        if self.obj_type == "utterance":
            for idx, utterance in enumerate(corpus.iter_utterances()):
                if self._print_output(idx):
                    print(f"%03d {self.obj_type} processed" % (idx))

                if self.input_field is None:
                    text_entry = utterance.text
                elif isinstance(self.input_field, str):
                    text_entry = utterance.meta(self.input_field)
                if text_entry is None:
                    continue

                l += 1
                sentiment = self.sia.polarity_scores(text_entry)
                corpus_sent["pos"] += sentiment["pos"]
                corpus_sent["neg"] += sentiment["neg"]
                corpus_sent["neu"] += sentiment["neu"]
                corpus_sent["compound"] += sentiment["compound"]
                values.append(sentiment["compound"])

            corpus_sent = {key: value / l for key, value in corpus_sent.items()}
            self.mean = corpus_sent["compound"]

            squared_differences = [(x - self.mean) ** 2 for x in values]
            variance = sum(squared_differences) / (len(values) - 1)
            standard_deviation = math.sqrt(variance)
            self.sd = standard_deviation

            return self

    def transform(self, corpus: Corpus) -> Corpus:
        """

        :param corpus: Corpus
        :return: the corpus
        """

        if self.obj_type == "utterance":
            total = len(list(corpus.iter_utterances()))

            for idx, utterance in enumerate(corpus.iter_utterances()):
                if self._print_output(idx):
                    print(f"%03d/%03d {self.obj_type} processed" % (idx, total))

                if not self.input_filter(self, utterance):
                    continue

                if self.input_field is None:
                    if "&gt" in utterance.text:
                        try:
                            text_entry = utterance.text.split("\n")[1]
                        except:
                            text_entry = utterance.text.split(".")[1]
                    else:
                        text_entry = utterance.text
                    if " /s " in text_entry:
                        text_entry = text_entry.split(" \s ")[0]
                    elif "\n/s" in text_entry:
                        text_entry = text_entry.split("\n/s")[0]
                    else:
                        text_entry = text_entry
                elif isinstance(self.input_field, str):
                    text_entry = utterance.meta(self.input_field)
                if text_entry is None:
                    continue

                if " /s " in utterance.text or "\n/s" in utterance.text:
                    sentiment = self.sia.polarity_scores(text_entry)
                    convo = utterance.get_conversation()
                    replies = list(convo.get_subtree(utterance.id).children)
                    acc_sent = 0
                    average_sent = 0

                    if len(replies) > 0:
                        for reply in replies:
                            reply_sent = self.sia.polarity_scores(reply.utt.text)
                            acc_sent += reply_sent["compound"]
                            reply.utt.add_meta("sentiment", reply_sent)
                        average_sent = acc_sent / len(replies)

                    utterance.add_meta("sentiment", sentiment)
                    utterance.add_meta("replies_sentiment", average_sent)
                    agree_score = 0

                    if average_sent == 0:
                        agree_score = 0
                    elif (
                        (
                            average_sent <= (self.mean - self.sd * 0.5)
                            and average_sent >= (self.mean - self.sd * 2)
                            and sentiment["compound"] <= (self.mean - self.sd * 0.5)
                        )
                        or (
                            average_sent >= (self.mean + self.sd * 0.5)
                            and average_sent <= (self.mean - self.sd * 2)
                            and sentiment["compound"] >= (self.mean + self.sd * 0.5)
                        )
                        or (
                            sentiment["compound"] <= (self.mean - self.sd * 0.5)
                            and sentiment["compound"] >= (self.mean - self.sd * 2)
                            and average_sent <= (self.mean - self.sd * 0.5)
                        )
                        or (
                            sentiment["compound"] >= (self.mean + self.sd * 0.5)
                            and sentiment["compound"] <= (self.mean - self.sd * 2)
                            and average_sent >= (self.mean + self.sd * 0.5)
                        )
                    ):
                        agree_score = (average_sent + sentiment["compound"]) / 2
                    elif (
                        average_sent < (self.mean - self.sd * 2)
                        and sentiment["compound"] < (self.mean - self.sd * 2)
                    ) or (
                        average_sent > (self.mean + self.sd * 2)
                        and sentiment["compound"] > (self.mean + self.sd * 2)
                    ):
                        agree_score = -abs((average_sent + sentiment["compound"]) / 2)
                    elif (
                        average_sent > (self.mean + self.sd * 0.5)
                        and sentiment["compound"] < (self.mean - self.sd * 0.5)
                    ) or (
                        average_sent < (self.mean - self.sd * 0.5)
                        and sentiment["compound"] > (self.mean + self.sd * 0.5)
                    ):
                        agree_score = (average_sent + -sentiment["compound"]) / 2
                    else:
                        agree_score = 0

                    utterance.add_meta("agree_score", agree_score)
        else:
            raise KeyError("obj_type must be utterance")

        if self.verbosity > 0:
            print(f"%03d/%03d {self.obj_type} processed" % (total, total))
        return corpus


def whitelist(self, corpus: Corpus):
    whitelist = []
    for convo in corpus.iter_conversations():
        for utt in convo.iter_utterances():
            if " /s " in utt.text or "\n/s" in utt.text:
                whitelist.append(utt.id)
                convo = utt.get_conversation()
                replies = list(convo.get_subtree(utt.id).bfs_traversal())
                for reply in replies:
                    if reply.utt.id != utt.id:
                        whitelist.append(reply.utt.id)

    self.whitelist = whitelist


def filter(self, utt):
    return utt.id in self.whitelist
