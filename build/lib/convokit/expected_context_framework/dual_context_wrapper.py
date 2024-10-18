from sklearn.metrics.pairwise import paired_distances
import pandas as pd
import numpy as np
import json, os

from convokit.expected_context_framework import ExpectedContextModelTransformer
from convokit.transformer import Transformer


class DualContextWrapper(Transformer):
    """
    Transformer that derives and compares characterizations of terms and utterances with respect to two different choices of conversational context. Designed in particular to contrast replies and predecessors, though other choices of context are also possible.

    This is a wrapper that encompasses two instances of `ExpectedContextModelTransformer`, stored at the `ec_models` attribute.
    It computes two particular comparative term-level statistics, orientation and shift, stored as the `term_orientations` and `term_shifts` attributes.
    It also computes these statistics at the utterance level in the transform step.

    :param context_fields: list containing the names of the utterance-level attributes containing the IDs of the context-utterances used by each of the `ExpectedContextModelTransformer` instances.
    :param output_prefixes: list containing the name of the attributes and vectors that each `ExpectedContextModelTransformer` instances will write to in the transform step.
    :param vect_field: the name of the vectors to use as input vector representation for utterances, as stored in a corpus.
    :param context_vect_field: the name of the vectors to use as input vector representations for context-utterances, as stored in a corpus. by default, the transformer will use the same vector representations as utterances, specified in `vect_field`. if you expect that utterances and context-utterances will differ in some way (e.g., they come from speakers in a conversation who play clearly delineated roles), then it's a good idea to use a different input representation.
    :param wrapper_output_prefix: the metadata fields where the utterance-level orientation and shift statistics are stored. By default, these attributes are stored as `orn` and `shift` in the metadata; if `wrapper_output_prefix` is specified, then they are stored as `<wrapper_output_prefix>_orn` (orientation) and `<wrapper_output_prefix>_shift` (shift).
    :param n_svd_dims: the dimensionality of the representations to derive (via LSA/SVD).
    :param snip_first_dim: whether or not to remove the first dimension of the derived representations. by default this is set to `True`, since we've found that the first dimension tends to reflect term frequency, making the output less informative. Note that if `snip_first_dim=True` then in practice, we output `n_svd_dims-1`-dimensional representations.
    :param n_clusters: the number of clusters to infer.
    :param cluster_on: whether to cluster on utterance or term representations, (corresponding to values `'utts'` or `'terms'`). By default, we infer clusters based on representations of the utterances from the training data, and then assign term and context-utterance representations to the resultant clusters. In some cases (e.g., if utterances are highly unstructured and lengthy) it might be better to cluster term representations first.
    :param random_state: the random seed to use in the LSA step (which calls a randomized implementation of SVD)
    :param cluster_random_state: the random seed to use to infer clusters.

    """

    def __init__(
        self,
        context_fields,
        output_prefixes,
        vect_field,
        context_vect_field=None,
        wrapper_output_prefix="",
        n_svd_dims=25,
        snip_first_dim=True,
        n_clusters=8,
        cluster_on="utts",
        random_state=None,
        cluster_random_state=None,
    ):
        self.context_fields = context_fields
        self.output_prefixes = output_prefixes
        self.vect_field = vect_field
        self.context_vect_field = context_vect_field
        if self.context_vect_field is None:
            self.context_vect_field = vect_field

        self.n_svd_dims = n_svd_dims
        self.snip_first_dim = snip_first_dim
        self.n_clusters = n_clusters
        self.cluster_on = cluster_on
        self.random_state = random_state
        self.cluster_random_state = cluster_random_state
        self.wrapper_output_prefix = wrapper_output_prefix

        first_model = ExpectedContextModelTransformer(
            context_field=self.context_fields[0],
            output_prefix=self.output_prefixes[0],
            vect_field=self.vect_field,
            context_vect_field=self.context_vect_field,
            n_svd_dims=self.n_svd_dims,
            snip_first_dim=self.snip_first_dim,
            n_clusters=self.n_clusters,
            cluster_on=self.cluster_on,
            random_state=self.random_state,
            cluster_random_state=self.cluster_random_state,
        )
        second_model = ExpectedContextModelTransformer(
            context_field=self.context_fields[1],
            output_prefix=self.output_prefixes[1],
            vect_field=self.vect_field,
            context_vect_field=self.context_vect_field,
            n_svd_dims=self.n_svd_dims,
            snip_first_dim=self.snip_first_dim,
            n_clusters=self.n_clusters,
            cluster_on=self.cluster_on,
            random_state=self.random_state,
            cluster_random_state=self.cluster_random_state,
        )
        self.ec_models = [first_model, second_model]

    def fit(self, corpus, y=None, selector=lambda x: True, context_selector=lambda x: True):
        """
        Fits a transformer over training data: fits the two `ExpectedContextModelTransformer` instances, and computes term-level orientation and shift.

        :param corpus: Corpus containing training data
        :param selector: a boolean function of signature `filter(utterance)` that determines which utterances will be considered in the fit step. defaults to using all utterances.
        :param context_selector: a boolean function of signature `filter(utterance)` that determines which context-utterances will be considered in the fit step. defaults to using all utterances.
        :return: None
        """

        self.ec_models[0].fit(corpus, selector=selector, context_selector=context_selector)
        self.ec_models[1].ec_model.set_model(self.ec_models[0].ec_model)

        self.ec_models[1].fit(corpus, selector=selector, context_selector=context_selector)
        self._compute_term_stats()

    def transform(self, corpus, selector=lambda x: True):
        """
        Computes vector representations, ranges, and cluster assignments for utterances in a corpus, using the two `ExpectedContextModelTransformer` instances. Also computes utterance-level orientation and shift.

        :param corpus: Corpus
        :param selector: a boolean function of signature `filter(utterance)` that determines which utterances to transform. defaults to all utterances.
        :return: the Corpus, with per-utterance attributes.
        """
        self.ec_models[0].transform(corpus, selector=selector)
        self.ec_models[1].transform(corpus, selector=selector)
        if self.wrapper_output_prefix == "":
            orn_field = "orn"
            shift_field = "shift"
        else:
            orn_field = self.wrapper_output_prefix + "_orn"
            shift_field = self.wrapper_output_prefix + "_shift"

        for ut in corpus.iter_utterances(selector=selector):
            ut.meta[orn_field] = (
                ut.meta[self.output_prefixes[0] + "_range"]
                - ut.meta[self.output_prefixes[1] + "_range"]
            )

        utt_shifts = paired_distances(
            corpus.get_vectors(self.output_prefixes[0] + "_repr"),
            corpus.get_vectors(self.output_prefixes[1] + "_repr"),
        )
        for id, shift in zip(
            corpus.get_vector_matrix(self.output_prefixes[0] + "_repr").ids, utt_shifts
        ):
            corpus.get_utterance(id).meta[shift_field] = shift

    def transform_utterance(self, utt):
        """
        Computes vector representations, ranges, and cluster assignments for an utterance, using the two `ExpectedContextModelTransformer` instances. Also computes utterance-level orientation and shift. Note that the utterance must contain the input representation as a metadata field, specified by what was passed into the constructor as the `vect_field` argument.
        Will write all of these characterizations (including vectors) to the utterance's metadata.

        :param utt: Utterance
        :return: the utterance, with per-utterance attributes.
        """
        utt = self.ec_models[0].transform_utterance(utt)
        utt = self.ec_models[1].transform_utterance(utt)
        if self.wrapper_output_prefix == "":
            orn_field = "orn"
            shift_field = "shift"
        else:
            orn_field = self.wrapper_output_prefix + "_orn"
            shift_field = self.wrapper_output_prefix + "_shift"

        utt.meta[orn_field] = (
            utt.meta[self.output_prefixes[0] + "_range"]
            - utt.meta[self.output_prefixes[1] + "_range"]
        )

        utt.meta[shift_field] = float(
            paired_distances(
                np.array([utt.meta[self.output_prefixes[0] + "_repr"]]),
                np.array([utt.meta[self.output_prefixes[1] + "_repr"]]),
            )[0]
        )
        return utt

    def _compute_term_stats(self):
        self.term_orientations = (
            self.ec_models[0].get_term_ranges() - self.ec_models[1].get_term_ranges()
        )
        self.term_shifts = paired_distances(
            self.ec_models[0].get_term_reprs(), self.ec_models[1].get_term_reprs()
        )

    def get_terms(self):
        """
        Gets the names of the terms for which the transformer has computed representations.

        :return: list of terms
        """
        return self.ec_models[0].get_terms()

    def get_term_df(self):
        """
        Gets a Pandas dataframe containing term-level statistics computed by the transformer (shift, orientation) and its constituent `ExpectedContextModelTransformer` instances (ranges).

        :return: dataframe of term-level statistics
        """
        return pd.DataFrame(
            {
                "index": self.get_terms(),
                "orn": self.term_orientations,
                "shift": self.term_shifts,
                self.output_prefixes[0] + "_range": self.ec_models[0].get_term_ranges(),
                self.output_prefixes[1] + "_range": self.ec_models[1].get_term_ranges(),
            }
        ).set_index("index")

    def summarize(self, k=10, max_chars=1000, corpus=None):
        """
        For each constituent ExpectedContextModelTransformer, prints inferred clusters and statistics about their sizes.

        :param k: number of examples to print out.
        :param max_chars: maximum number of characters per utterance/context-utterance to print. Can be toggled to control the size of the output.
        :param corpus: optional, the corpus that the transformer was trained on. if set, will print example utterances and context-utterances as well as terms.

        :return: None
        """
        print("MODEL", self.output_prefixes[0])
        self.ec_models[0].summarize(k, max_chars, corpus)
        print()
        print("MODEL", self.output_prefixes[1])
        self.ec_models[1].summarize(k, max_chars, corpus)

    def load(self, dirname, model_dirs=None):
        """
        Loads a model from disk.

        :param dirname: directory to read model from
        :param model_dirs: optional list containing the directories (relative to `dirname`) in which each `ExpectedContextModelTransformer` is stored. defaults to the `output_prefixes` argument passed at initialization.
        :return: None
        """
        if model_dirs is None:
            model_dirs = self.output_prefixes
        self.ec_models[0].load(os.path.join(dirname, model_dirs[0]))
        self.ec_models[1].load(os.path.join(dirname, model_dirs[1]))
        self._compute_term_stats()

    def dump(self, dirname):
        """
        Writes a model to disk. Will store each `ExpectedContextModelTransformer` in a separate directory with names given by the `output_prefixes` argument passed at initialization.

        :param dirname: directory to write model to.
        :return: None
        """
        try:
            os.mkdir(dirname)
            os.mkdir(os.path.join(dirname, self.output_prefixes[0]))
            os.mkdir(os.path.join(dirname, self.output_prefixes[1]))
        except:
            pass
        self.ec_models[0].dump(os.path.join(dirname, self.output_prefixes[0]))
        self.ec_models[1].dump(os.path.join(dirname, self.output_prefixes[1]))
