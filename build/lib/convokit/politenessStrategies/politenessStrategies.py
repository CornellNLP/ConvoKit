from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spacy.tokens import Doc

from convokit.model import Corpus, Utterance, Speaker
from convokit.politeness_collections.politeness_api.features.politeness_strategies import (
    get_politeness_strategy_features,
)
from convokit.politeness_collections.politeness_cscw_zh.strategy_extractor import (
    get_chinese_politeness_strategy_features,
)
from convokit.politeness_collections.politeness_local.strategy_extractor import (
    get_local_politeness_strategy_features,
)
from convokit.text_processing.textParser import process_text
from convokit.transformer import Transformer


class PolitenessStrategies(Transformer):
    """
    Encapsulates extraction of politeness strategies from utterances in a
    Corpus.

    :param parse_attribute_name: metadata attribute name to read parses from. Default is 'parsed'.
    :param strategy_attribute_name: metadata attribute name to store politeness strategies features under during the `transform()` step.  Default is 'politeness_strategies'.
    :param marker_attribute_name: metadata attribute name to store politeness markers under during the `transform()` step. Default is 'politeness_markers'.
    :param strategy_collection: collection of politeness strategies to extract. Options include:
        "politeness_api": English politeness strategies proposed in A computational approach to politeness with application to social factors (https://www.cs.cornell.edu/~cristian/Politeness.html)
        "politeness_local": English politeness strategies realized through local markers as used in Facilitating the Communication of Politeness through Fine-Grained Paraphrasing (https://www.cs.cornell.edu/~cristian/Politeness_Paraphrasing.html)
        "politeness_cscw_zh":  Chinese politeness strategies adapted from `Studying Politeness across Cultures using English Twitter and Mandarin Weibo (https://dl.acm.org/doi/abs/10.1145/3415190)
        Default is "politeness_api".
    :param verbose: whether and how often to print status messages while computing features.
    """

    def __init__(
        self,
        parse_attribute_name: str = "parsed",
        strategy_attribute_name: str = "politeness_strategies",
        marker_attribute_name: str = "politeness_markers",
        strategy_collection: str = "politeness_api",
        verbose: int = 0,
    ):
        self.parse_attribute_name = parse_attribute_name
        self.strategy_attribute_name = strategy_attribute_name
        self.marker_attribute_name = marker_attribute_name
        self.strategy_collection = strategy_collection
        self.verbose = verbose

        self._extractor_lookup = {
            "politeness_api": get_politeness_strategy_features,
            "politeness_local": get_local_politeness_strategy_features,
            "politeness_cscw_zh": get_chinese_politeness_strategy_features,
        }

    def transform(
        self,
        corpus: Corpus,
        selector: Optional[Callable[[Utterance], bool]] = lambda utt: True,
        markers: bool = False,
    ):
        """
        Extract politeness strategies from each utterances in the corpus and annotate
        the utterances with the extracted strategies. Requires that the corpus has previously
        been transformed by a Parser, such that each utterance has dependency parse info in
        its metadata table.

        :param corpus: the corpus to compute features for.
        :param selector: a (lambda) function that takes an Utterance and returns a bool indicating whether the utterance should be included in this annotation step.
        :param markers: whether or not to add politeness occurrence markers
        """

        total_utts = len(list(corpus.iter_utterances()))

        for idx, utt in enumerate(corpus.iter_utterances()):
            if self.verbose > 0 and idx > 0 and idx % self.verbose == 0:
                print("%03d/%03d utterances processed" % (idx, total_utts))

            if selector(utt):
                parsed = utt.retrieve_meta(self.parse_attribute_name)
                for i, sent in enumerate(parsed):
                    for p in sent["toks"]:
                        # p["tok"] = re.sub("[^a-z,.:;]", "", p["tok"].lower())
                        p["tok"] = p["tok"].lower()

                parses = [x["toks"] for x in parsed]

                utt.meta[self.strategy_attribute_name], marks = self._extractor_lookup[
                    self.strategy_collection
                ](parses)

                if markers:
                    utt.meta[self.marker_attribute_name] = marks
            else:
                utt.meta[self.strategy_attribute_name] = None
                utt.meta[self.marker_attribute_name] = None

        return corpus

    def transform_utterance(
        self, utt: Utterance, spacy_nlp: Callable[[str], Doc] = None, markers: bool = False
    ):
        """
        Extract politeness strategies for raw string inputs (or individual utterances)

        :param utt: the utterance to be annotated with politeness strategies.
        :spacy_nlp: if provided, will use this SpaCy object to do parsing; otherwise will initialize an object via `load('en')`.
        :return: the utterance with politeness annotations.
        """

        if isinstance(utt, str):
            utt = Utterance(text=utt, speaker=Speaker(id="speaker"))

        if self.parse_attribute_name not in utt.meta:
            if spacy_nlp is None:
                raise ValueError("spacy object required")

            parses = process_text(utt.text, spacy_nlp=spacy_nlp)
            utt.add_meta(self.parse_attribute_name, parses)

        parsed = utt.retrieve_meta(self.parse_attribute_name)
        for i, sent in enumerate(parsed):
            for p in sent["toks"]:
                p["tok"] = p["tok"].lower()
        parses = [x["toks"] for x in parsed]

        utt.meta[self.strategy_attribute_name], marks = self._extractor_lookup[
            self.strategy_collection
        ](parses)

        if markers:
            utt.meta[self.marker_attribute_name] = marks

        return utt

    def _get_feat_df(
        self, corpus: Corpus, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True
    ):
        """
        Construct binary feature dataframe. Used in summarize()

        :param corpus: the target Corpus
        :param selector: (lambda) function specifying whether the utterance should be included
        """

        utts = list(corpus.iter_utterances(selector))

        if self.strategy_attribute_name not in utts[0].meta:
            print(
                "Could not find politeness strategies metadata. Running transform() on corpus first...",
                end="",
            )
            self.transform(corpus)
            print("Done.")

        df_feat = pd.DataFrame.from_dict(
            {utt.id: utt.meta["politeness_strategies"] for utt in utts}, orient="index"
        )

        return df_feat

    def summarize(
        self,
        corpus: Corpus,
        selector: Callable[[Utterance], bool] = lambda utt: True,
        plot: bool = False,
        y_lim=None,
    ):
        """
        Calculates strategy prevalence and plot graph if plot == True, with an optional selector that specifies which utterances to include in the analysis.

        :param corpus: the target Corpus
        :param selector: a function (typically, a lambda function) that takes an Utterance and returns True or False (i.e. include / exclude).
        By default, the selector includes all Utterances in the Corpus.
        :param plot: whether or not to output graph.
        :return: a pandas DataFrame of scores with graph optionally outputted
        """

        df_feat = self._get_feat_df(corpus, selector)
        proportions = df_feat.sum(axis=0) / len(df_feat)
        num_strategies = len(proportions)

        if plot:
            plt.figure(dpi=200, figsize=(9, 6))
            plt.bar(proportions.index, proportions.values)
            plt.xticks(np.arange(0.4, num_strategies + 0.4), rotation=45, ha="right")
            plt.ylabel("% utterance using strategy", size=20)
            plt.yticks(size=15)

            if y_lim != None:
                plt.ylim(0, y_lim)
            plt.show()

        return proportions
