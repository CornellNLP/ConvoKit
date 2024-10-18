import numpy as np
import pandas as pd
from convokit.transformer import Transformer
from convokit.speaker_convo_helpers.speaker_convo_attrs import SpeakerConvoAttrs
from itertools import chain
from collections import Counter
from convokit.speaker_convo_helpers.speaker_convo_lifestage import SpeakerConvoLifestage
from convokit import Utterance
from convokit.surprise import Surprise
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import List


def _join_all_tokens(parses):
    joined = []
    for parse in parses:
        for sent in parse:
            joined += [tok["tok"].lower() for tok in sent["toks"]]
    return joined


def _nan_mean(arr):
    arr = [x for x in arr if not np.isnan(x)]
    if len(arr) > 0:
        return np.mean(arr)
    else:
        return np.nan


def _perplexity(test_text, train_text):
    N_train, N_test = len(train_text), len(test_text)
    if min(N_train, N_test) == 0:
        return np.nan
    train_counts = Counter(train_text)
    return sum(-np.log(train_counts.get(tok, 1) / N_train) for tok in test_text) / N_test


class SpeakerConvoDiversity(Transformer):
    """
    Implements methodology to compute the linguistic divergence between a speaker's activity in each conversation in a corpus (i.e., the language of their utterances) and a reference language model trained over a different set of conversations/speakers.  See `SpeakerConvoDiversityWrapper` for more specific implementation which compares language used by individuals within fixed lifestages, and see the implementation of this wrapper for examples of calls to this transformer.

    The transformer assumes that a corpus has already been tokenized (via a call to `TextParser`).

    In general, this is appropriate for cases when the reference language model you wish to compare against varies across different speaker/conversations; in contrast, if you wish to compare many conversations to a _single_ language model (e.g., one trained on past conversations) then this will be inefficient.

    This will produce attributes per speaker-conversation (i.e., the behavior of a speaker in a conversation); hence it takes as parameters functions which will subset the data at a speaker-conversation level. these functions operate on a table which has as columns:
      * `speaker`: speaker ID
      * `convo_id`: conversation ID
      * `convo_idx`: n where this conversation is the nth that the speaker participated in
      * `tokens`: all utterances the speaker contributed to the conversation, concatenated together as a single list of words
      * any other speaker-conversation, speaker, or conversation-level metadata required to filter input and select reference language models per speaker-conversation (passed in via the `speaker_convo_cols`, `speaker_cols` and `convo_cols` parameters)
    The table is the output of calling  `Corpus.get_full_attribute_table`; see documentation of that function for further reference.

    The transformer supports two broad types of comparisons:
      * if `groupby=[]`, then each text will be compared against a single reference text (specified by `select_fn`)
      * if `groupby=[key]` then each text will be compared against a set of reference texts, where each reference text represents a different chunk of the data, aggregated by `key` (e.g., each text could be compared against the utterances contributed by different speakers, such that in each iteration of a divergence computation, the text is compared against just the utterances of a single speaker.)

    :param cmp_select_fn: the subset of speaker-conversation entries to compute divergences for. function of the form fn(df, aux) where df is a data frame indexed by speaker-conversation, and aux is any auxiliary parametsr required; returns a boolean mask over the dataframe.
    :param ref_select_fn: the subset of speaker-conversation entries to compute reference language models over. function of the form fn(df, aux) where df is a data frame indexed by speaker-conversation, and aux is any auxiliary parameters required; returns a boolean mask over the dataframe.
    :param select_fn: function of the form fn(df, row, aux) where df is a data frame indexed by speaker-conversation, row is a row of a dataframe indexed by speaker-conversation, and aux is any auxiliary parameters required; returns a boolean mask over the dataframe.
    :param divergence_fn: function to compute divergence between a speaker-conversation and reference texts. By default, the transformer will compute unigram perplexity scores, as implemented by the `compute_divergences` function. However, you can also specify your own divergence function (e.g., some sort of bigram divergence) using the same function signature.
    :param speaker_convo_cols: additional speaker-convo attributes used as input to the selector functions
    :param speaker_cols: additional speaker-level attributes
    :param convo_cols: additional conversation-level attributes
    :param model_key_cols: list of attributes that is a subset of the attributes retrieved using `Corpus.get_full_attribute_table`. these attributes specify which speaker-convo entries correspond to the same reference text. `select_fn` should return the same boolean mask over the dataframe for speaker-convo entries which have the same values for all these attributes.
    :param groupby: whether to aggregate the reference texts according to the specified keys (leave empty to avoid aggregation).
    :param aux_input: a dictionary of auxiliary input to the selector functions and the divergence computation
    :param recompute_tokens: whether to reprocess tokens by aggregating all tokens across different utterances made by a speaker in a conversation. by default, will cache existing output.
    :param verbosity: frequency of status messages.
    """

    def __init__(
        self,
        output_field,
        cmp_select_fn=lambda df, aux: np.ones(len(df)).astype(bool),
        ref_select_fn=lambda df, aux: np.ones(len(df)).astype(bool),
        select_fn=lambda df, row, aux: np.ones(len(df)).astype(bool),
        speaker_convo_cols=[],
        speaker_cols=[],
        convo_cols=[],
        model_key_cols=["speaker", "convo_id"],
        groupby=[],
        aux_input={},
        recompute_tokens=False,
        verbosity=0,
    ):
        self.output_field = output_field
        self.surprise_attr_name = f"surprise_{output_field}"
        self.cmp_select_fn = cmp_select_fn
        self.ref_select_fn = ref_select_fn
        self.select_fn = select_fn
        self.speaker_convo_cols = speaker_convo_cols
        self.speaker_cols = speaker_cols
        self.convo_cols = convo_cols
        self.model_key_cols = model_key_cols
        self.groupby = groupby
        self.aux_input = aux_input
        self.verbosity = verbosity

        self.agg_tokens = SpeakerConvoAttrs(
            "tokens", agg_fn=_join_all_tokens, recompute=recompute_tokens
        )

        self.model_key_map = {}

    def transform(self, corpus):
        if self.verbosity > 0:
            print("joining tokens across conversation utterances")
        corpus = self.agg_tokens.transform(corpus)

        speaker_convo_cols = list(set(self.speaker_convo_cols + ["tokens"]))

        input_table = corpus.get_full_attribute_table(
            list(set(self.speaker_convo_cols + ["tokens"])), self.speaker_cols, self.convo_cols
        )

        surprise_transformer = self._init_surprise(
            lambda utt: self._get_model_key(utt, self.model_key_cols, input_table)
        )
        surprise_transformer.fit(
            corpus, text_func=lambda utt: self._get_text_func(utt, input_table)
        )
        surprise_transformer.transform(
            corpus,
            "speaker",
            target_text_func=lambda utt: self._get_utt_row(utt, input_table).tokens,
        )
        self._set_output(corpus, input_table)
        return corpus

    def _get_utt_row(self, utt: Utterance, df: pd.DataFrame):
        """
        Returns the row in `df` corresponding to `utt` using the speaker and conversation id of `utt`.
        """
        return df.loc[f"{utt.speaker.id}__{utt.conversation_id}"]

    def _get_model_key(self, utt: Utterance, model_key_cols: List[str], df: pd.DataFrame):
        """
        Returns the model key used by `Surprise` that corresponds to `utt` and `model_key_cols`.
        Finds the row in `df` corresponding to `utt` and creates a model key using the values for the attributes in `model_key_cols` in that row.
        """
        utt_row = self._get_utt_row(utt, df)
        key = ".".join([str(utt_row[col]) for col in model_key_cols])
        self.model_key_map[key] = (utt_row["speaker"], utt_row["convo_id"])
        return key

    def _init_surprise(self, model_key_selector):
        """
        Initializes an instance of the `Surprise` transformer with paramters corresponding to this instance of `SpeakerConvoDiversity`.
        """
        target_sample_size = (
            self.aux_input["cmp_sample_size"] if "cmp_sample_size" in self.aux_input else 200
        )
        context_sample_size = (
            self.aux_input["ref_sample_size"] if "ref_sample_size" in self.aux_input else 1000
        )
        n_samples = self.aux_input["n_iters"] if "n_iters" in self.aux_input else 50
        return Surprise(
            model_key_selector,
            tokenizer=lambda x: x,
            surprise_attr_name=self.surprise_attr_name,
            target_sample_size=target_sample_size,
            context_sample_size=context_sample_size,
            n_samples=n_samples,
            smooth=False,
        )

    def _get_text_func(self, utt: Utterance, df: pd.DataFrame):
        """
        Returns the reference text that should be to calculate speaker convo diversity for the speaker-convo group that `utt` belongs to.
        """
        utt_row = self._get_utt_row(utt, df)
        ref_subset = df[self.ref_select_fn(df, self.aux_input)]
        ref_subset = ref_subset[self.select_fn(ref_subset, utt_row, self.aux_input)]
        if not self.groupby:
            return [np.array(list(chain(*ref_subset.tokens.values)))]
        ref_subset = (
            ref_subset.groupby(self.groupby).tokens.agg(lambda x: list(chain(*x))).reset_index()
        )
        ref_subset["tokens"] = ref_subset.tokens.map(np.array)
        return ref_subset.tokens.values

    def _get_row(self, df, fields, vals):
        """
        Retrieves the row of `df` where each attribute `fields[i]` has the value `vals[i]`.
        Assumes that there is exactly one row in `df` with fields equal to vals.
        """
        str_df = df.astype("str")
        mask = np.ones(df.shape[0], dtype=bool)
        for field, val in zip(fields, vals):
            mask &= str_df[field] == val
        return df[mask].iloc[0]

    def _set_output(self, corpus, df):
        """
        Adds `self.output_field` to speaker convo info using scores returned by `Surprise` transformer.
        """
        entries = []
        for speaker in tqdm(corpus.iter_speakers(), desc="set output"):
            if self.surprise_attr_name in speaker.meta:
                scores = speaker.meta[self.surprise_attr_name]
                for key, score in scores.items():
                    if np.isnan(score):
                        continue
                    speaker, convo_id = self.model_key_map[key]
                    corpus.set_speaker_convo_info(speaker, convo_id, self.output_field, score)


class SpeakerConvoDiversityWrapper(Transformer):
    """
    Implements methodology for calculating linguistic diversity per life-stage. A wrapper around `SpeakerConvoDiversity`.

    Outputs the following (speaker, conversation) attributes:
      * `div__self` (within-diversity)
      * `div__other` (across-diversity)
      * `div__adj` (relative diversity)

    Note that `np.nan` is returned for (speaker, conversation) pairs with not enough text.

    :param output_field: prefix of attributes to output, defaults to 'div'
    :param lifestage_size: number of conversations per lifestage
    :param max_exp: highest experience level (i.e., # convos taken) to compute diversity scores for.
    :param sample_size: number of words to sample per convo
    :param min_n_utterances: minimum number of utterances a speaker contributes per convo for that (speaker, convo) to get scored
    :param n_iters: number of samples to take for perplexity scoring
    :param cohort_delta: timespan between when speakers start for them to be counted as part of the same cohort. defaults to 2 months
    :param verbosity: amount of output to print
    """

    def __init__(
        self,
        output_field="div",
        lifestage_size=20,
        max_exp=120,
        sample_size=200,
        min_n_utterances=1,
        n_iters=50,
        cohort_delta=60 * 60 * 24 * 30 * 2,
        verbosity=100,
    ):
        aux_input = {
            "n_iters": n_iters,
            "cmp_sample_size": sample_size,
            "ref_sample_size": (lifestage_size // 2) * sample_size,
            "max_exp": max_exp,
            "min_n_utterances": min_n_utterances,
            "cohort_delta": cohort_delta,
            "lifestage_size": lifestage_size,
        }
        self.lifestage_transform = SpeakerConvoLifestage(lifestage_size)
        self.output_field = output_field

        # SpeakerConvoDiversity transformer to compute within-diversity
        self.self_div = SpeakerConvoDiversity(
            output_field + "__self",
            cmp_select_fn=lambda df, aux: (df.convo_idx < aux["max_exp"])
            & (df.n_convos__speaker >= aux["max_exp"])
            & (df.tokens.map(len) >= aux["cmp_sample_size"])
            & (df.n_utterances >= aux["min_n_utterances"]),
            ref_select_fn=lambda df, aux: np.ones(len(df)).astype(bool),
            select_fn=lambda df, row, aux: (df.convo_idx % 2 != row.convo_idx % 2)
            & (df.speaker == row.speaker)
            & (df.lifestage == row.lifestage),
            speaker_convo_cols=["n_utterances", "lifestage"],
            speaker_cols=["n_convos"],
            model_key_cols=["convo_idx", "speaker", "lifestage"],
            groupby=[],
            aux_input=aux_input,
            verbosity=verbosity,
        )

        # SpeakerConvoDiversity transformer to compute across-diversity
        self.other_div = SpeakerConvoDiversity(
            output_field + "__other",
            cmp_select_fn=lambda df, aux: (df.convo_idx < aux["max_exp"])
            & (df.n_convos__speaker >= aux["max_exp"])
            & (df.tokens.map(len) >= aux["cmp_sample_size"])
            & (df.n_utterances >= aux["min_n_utterances"]),
            ref_select_fn=lambda df, aux: np.ones(len(df)).astype(bool),
            select_fn=lambda df, row, aux: (df.convo_idx % 2 != row.convo_idx % 2)
            & (df.speaker != row.speaker)
            & (df.lifestage == row.lifestage)
            & (df.n_convos__speaker >= (row.lifestage + 1) * aux["lifestage_size"])
            & (
                df.start_time__speaker.between(
                    row.start_time__speaker - aux["cohort_delta"],
                    row.start_time__speaker + aux["cohort_delta"],
                )
            ),
            speaker_convo_cols=["n_utterances", "lifestage"],
            speaker_cols=["n_convos", "start_time"],
            model_key_cols=["convo_idx", "speaker", "lifestage"],
            groupby=["speaker", "lifestage"],
            aux_input=aux_input,
            verbosity=verbosity,
        )
        self.verbosity = verbosity

    def transform(self, corpus):
        if self.verbosity > 0:
            print("getting lifestages")
        corpus = self.lifestage_transform.transform(corpus)
        if self.verbosity > 0:
            print("getting within diversity")
        corpus = self.self_div.transform(corpus)
        if self.verbosity > 0:
            print("getting across diversity")
        corpus = self.other_div.transform(corpus)
        if self.verbosity > 0:
            print("getting relative diversity")
        div_table = corpus.get_full_attribute_table(
            [self.output_field + "__self", self.output_field + "__other"]
        )
        div_table = div_table[
            div_table[self.output_field + "__self"].notnull()
            | div_table[self.output_field + "__other"].notnull()
        ]
        div_table[self.output_field + "__adj"] = (
            div_table[self.output_field + "__other"] - div_table[self.output_field + "__self"]
        )
        for idx, (_, row) in enumerate(div_table.iterrows()):
            if (idx > 0) and (self.verbosity > 0) and (idx % self.verbosity == 0):
                print(idx, "/", len(div_table))
            if not np.isnan(row[self.output_field + "__adj"]):
                corpus.set_speaker_convo_info(
                    row.speaker,
                    row.convo_id,
                    self.output_field + "__adj",
                    row[self.output_field + "__adj"],
                )
        return corpus
