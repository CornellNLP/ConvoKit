import numpy as np
from convokit.transformer import Transformer
from convokit.speaker_convo_helpers.speaker_convo_attrs import SpeakerConvoAttrs
from itertools import chain
from collections import Counter
from convokit.speaker_convo_helpers.speaker_convo_lifestage import SpeakerConvoLifestage


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


def compute_divergences(
    cmp_tokens,
    ref_token_list,
    aux_input={"cmp_sample_size": 200, "ref_sample_size": 1000, "n_iters": 50},
):
    """
    computes the linguistic divergence between a text `cmp_tokens` and a set of reference texts `ref_token_list`. in particular, implements a sampling-based unigram perplexity score (where the sampling is done to ensure that we do not incur length-based effects)

    this function takes in several parameters, through the `aux_input` argument:
            * cmp_sample_size: the number of tokens to sample from the analyzed text `cmp_tokens`. the function returns `np.nan` if `cmp_tokens` doesn't have that many tokens.
            * ref_sample_size: the nubmer of tokens to sample from each reference text. typically setting this to be longer than `cmp_tokens` makes sense, especially in the (typical) use case where language models are trained on longer texts. if none of the texts in `ref_token_list` pass this length threshold then the fucntion returns `np.nan`.
            * n_iters: the number of times to compute divergence.

    :param cmp_tokens: the text to compute divergence of (relative to texts in `ref_token_list`). is a list of tokens.
    :param ref_token_list: the texts on which to train reference language models against which `cmp_tokens` is compared. each entry in the list is a list of tokens.
    :param aux_input: additional parameters (see above)
    :return: if texts are of sufficient length, returns a perplexity score, else returns `np.nan`
    """
    if len(cmp_tokens) < aux_input["cmp_sample_size"]:
        return np.nan
    ref_token_list = [toks for toks in ref_token_list if len(toks) >= aux_input["ref_sample_size"]]
    if len(ref_token_list) == 0:
        return np.nan
    cmp_samples = np.random.choice(cmp_tokens, (aux_input["n_iters"], aux_input["cmp_sample_size"]))
    sample_idxes = np.random.randint(0, len(ref_token_list), size=(aux_input["n_iters"]))
    ref_samples = [
        np.random.choice(ref_token_list[idx], aux_input["ref_sample_size"]) for idx in sample_idxes
    ]
    return _nan_mean(
        [
            _perplexity(cmp_sample, ref_sample)
            for cmp_sample, ref_sample in zip(cmp_samples, ref_samples)
        ]
    )


class SpeakerConvoDiversity(Transformer):
    """
    implements methodology to compute the linguistic divergence between a speaker's activity in each conversation in a corpus (i.e., the language of their utterances) and a reference language model trained over a different set of conversations/speakers.  See `SpeakerConvoDiversityWrapper` for more specific implementation which compares language used by individuals within fixed lifestages, and see the implementation of this wrapper for examples of calls to this transformer.

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
    :param select_fn: function of the form fn(df,row, aux) where df is a data frame indexed by speaker-conversation, row is a row of a dataframe indexed by speaker-conversation, and aux is any auxiliary parameters required; returns a boolean mask over the dataframe.
    :param divergence_fn: function to compute divergence between a speaker-conversation and reference texts. By default, the transformer will compute unigram perplexity scores, as implemented by the `compute_divergences` function. However, you can also specify your own divergence function (e.g., some sort of bigram divergence) using the same function signature.
    :param speaker_convo_cols: additional speaker-convo attributes used as input to the selector functions
    :param speaker_cols: additional speaker-level attributes
    :param convo_cols: additional conversation-level attributes
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
        divergence_fn=compute_divergences,
        speaker_convo_cols=[],
        speaker_cols=[],
        convo_cols=[],
        groupby=[],
        aux_input={},
        recompute_tokens=False,
        verbosity=0,
    ):
        self.output_field = output_field
        self.cmp_select_fn = cmp_select_fn
        self.ref_select_fn = ref_select_fn
        self.select_fn = select_fn
        self.divergence_fn = divergence_fn
        self.speaker_convo_cols = speaker_convo_cols
        self.speaker_cols = speaker_cols
        self.convo_cols = convo_cols
        self.groupby = groupby
        self.aux_input = aux_input
        self.verbosity = verbosity

        self.agg_tokens = SpeakerConvoAttrs(
            "tokens", agg_fn=_join_all_tokens, recompute=recompute_tokens
        )

    def transform(self, corpus):
        if self.verbosity > 0:
            print("joining tokens across conversation utterances")
        corpus = self.agg_tokens.transform(corpus)

        speaker_convo_cols = list(set(self.speaker_convo_cols + ["tokens"]))

        input_table = corpus.get_full_attribute_table(
            list(set(self.speaker_convo_cols + ["tokens"])), self.speaker_cols, self.convo_cols
        )
        results = compute_speaker_convo_divergence(
            input_table,
            self.cmp_select_fn,
            self.ref_select_fn,
            self.select_fn,
            self.divergence_fn,
            self.groupby,
            self.aux_input,
            self.verbosity,
        )
        for entry in results:
            corpus.set_speaker_convo_info(
                entry["speaker"], entry["convo_id"], self.output_field, entry["divergence"]
            )
        return corpus


def compute_speaker_convo_divergence(
    input_table,
    cmp_select_fn=lambda df, aux: np.ones(len(df)).astype(bool),
    ref_select_fn=lambda df, aux: np.ones(len(df)).astype(bool),
    select_fn=lambda df, row, aux: np.ones(len(df)).astype(bool),
    divergence_fn=compute_divergences,
    groupby=[],
    aux_input={},
    verbosity=0,
):
    """
    given a table of speaker-conversation entries, computes linguistic divergences between each speaker-conversation entry and reference text. See `SpeakerConvoDiversity` for further explanation of arguments.

    The function operates on a table which has as columns:
            * `speaker`: speaker ID
            * `convo_id`: conversation ID
            * `convo_idx`: n where this conversation is the nth that the speaker participated in
            * `tokens`: all utterances the speaker contributed to the conversation, concatenated together as a single list of words
            * any other speaker-conversation, speaker, or conversation-level metadata required to filter input and select reference language models per speaker-conversation.

    :param cmp_select_fn: the subset of speaker-conversation entries to compute divergences for. function of the form fn(df, aux) where df is a data frame indexed by speaker-conversation, and aux is any auxiliary parametsr required; returns a boolean mask over the dataframe.
    :param ref_select_fn: the subset of speaker-conversation entries to compute reference language models over. function of the form fn(df, aux) where df is a data frame indexed by speaker-conversation, and aux is any auxiliary parameters required; returns a boolean mask over the dataframe.
    :param select_fn: function of the form fn(df,row, aux) where df is a data frame indexed by speaker-conversation, row is a row of a dataframe indexed by speaker-conversation, and aux is any auxiliary parameters required; returns a boolean mask over the dataframe.
    :param divergence_fn: function to compute divergence between a speaker-conversation and reference texts. By default, the transformer will compute unigram perplexity scores, as implemented by the `compute_divergences` function. However, you can also specify your own divergence function (e.g., some sort of bigram divergence) using the same function signature.
    :param groupby: whether to aggregate the reference texts according to the specified keys (leave empty to avoid aggregation).
    :param aux_input: a dictionary of auxiliary input to the selector functions and the divergence computation
    :param verbosity: frequency of status messages.
    """

    cmp_subset = input_table[cmp_select_fn(input_table, aux_input)]
    ref_subset = input_table[ref_select_fn(input_table, aux_input)]
    entries = []
    for idx, (_, row) in enumerate(cmp_subset.iterrows()):
        if (verbosity > 0) and (idx % verbosity == 0) and (idx > 0):
            print(idx, "/", len(cmp_subset))

        cmp_tokens = np.array(row.tokens)

        curr_ref_subset = ref_subset[select_fn(ref_subset, row, aux_input)]

        if len(groupby) == 0:
            ref_tokens = [np.array(list(chain(*curr_ref_subset.tokens.values)))]
        else:
            curr_ref_subset = (
                curr_ref_subset.groupby(groupby).tokens.agg(lambda x: list(chain(*x))).reset_index()
            )
            curr_ref_subset["tokens"] = curr_ref_subset.tokens.map(np.array)
            ref_tokens = curr_ref_subset.tokens.values

        divergence = divergence_fn(cmp_tokens, ref_tokens, aux_input)
        if not np.isnan(divergence):
            entries.append(
                {"speaker": row.speaker, "convo_id": row.convo_id, "divergence": divergence}
            )
    return entries


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
            divergence_fn=compute_divergences,
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
            divergence_fn=compute_divergences,
            speaker_convo_cols=["n_utterances", "lifestage"],
            speaker_cols=["n_convos", "start_time"],
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
