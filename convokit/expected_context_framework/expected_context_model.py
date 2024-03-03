import os
import pandas as pd
import numpy as np


from sklearn.preprocessing import Normalizer, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances, paired_distances
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from scipy import sparse
import joblib
import json

from convokit.transformer import Transformer


class ExpectedContextModelTransformer(Transformer):
    """
    Transformer that derives representations of terms and utterances in terms of their conversational context, i.e.,
    context-utterances that occur near an utterance, or utterances containing a term. Typically, the conversational
    context consists of immediate replies ("forwards context") or predecessors ("backwards context"), though
    this can be specified by the user via the `context_field` argument.

    The underlying model in the transformer, implemented as the `ExpectedContextModel` class, is fitted given input training
    data consisting of pairs of utterances and context-utterances, represented as feature vectors (e.g., tf-idf reweighted
    term-document matrices), specified via the `vect_field` and `context_vect_field` arguments. This model is stored as the `ec_model` attribute of the transformer, and can be accessed as such.
    In the fit step, the model, which is based off of latent semantic analysis (LSA), computes the following:

    * representations of terms and utterances in the training data, with respect to the context, along with representations of the context (which are derived in the underlying LSA step). the dimensionality of these representations is specified via the `n_svd_dims` argument (see also the `snip_first_dim` and `random_state` arguments). these can be accessed via various `get` functions that the transformer provides.
    * a term-level statistic, "range", measuring the variation in context-utterances associated with a term. One interpretation of this statistic is that it quantifies the "strengths of our expectations" of what reply a term typically gets, or what predecessors it typically follows.
    * a clustering of utterance, term and context representations. The resultant clusters can help interpret the representations the model derives, by highlighting salient groupings that emerge. The number of clusters is specified via the `n_clusters` argument; the `print_clusters` function can be called to inspect this output. (see also the `cluster_on` and `cluster_random_state` arguments)



    An instance of the transformer can be initialized with an instance of another, fitted transformer, via the `model` argument. This ensures that both
    transformers derive representations that are comparable, i.e., can be interpreted as being part of the same vector space, with distances between
    representations that are well-defined. As an example of when this might be useful, we may wish to compare representations derived with respect
    to expected replies, with representations pertaining to expected predecessors.

    The transfomer contains various functions to access term-level characterizations. In the transform step, it outputs
    vector representations of utterances, stored as `<output_prefix>_repr` in the corpus. It also outputs various attributes
    of utterances (names prefixed with `<output_prefix>_`), stored as metadata fields in each transformed utterance:

    * `range`: the range of the utterance
    * `clustering.cluster`: the name of the cluster the utterance has been assigned to
    * `clustering.cluster_id_`: the numerical ID (0-# of clusters) of the cluster the utterance has been assigned to
    * `clustering.cluster_dist`: the distance between the utterance representation and the centroid of its cluster

    :param context_field: the name of an utterance-level attribute containing the ID of the corresponding context-utterance. in particular, to use immediate predecessors as context, set `context_field` to `'reply_to'`. as another example, to use immediate replies, provided that utterances contain an attribute `next_id` containing the ID of their reply, set `context_field` to `'next_id'`.
    :param output_prefix: the name of the attributes and vectors to write to in the transform step. the transformer outputs several fields, which will be prefixed with the given string.
    :param vect_field: the name of the vectors to use as input vector representation for utterances, as stored in a corpus.
    :param context_vect_field: the name of the vectors to use as input vector representations for context-utterances, as stored in a corpus. by default, the transformer will use the same vector representations as utterances, specified in `vect_field`. if you expect that utterances and context-utterances will differ in some way (e.g., they come from speakers in a conversation who play clearly delineated roles), then it's a good idea to use a different input representation.
    :param n_svd_dims: the dimensionality of the representations to derive (via LSA/SVD).
    :param snip_first_dim: whether or not to remove the first dimension of the derived representations. by default this is set to `True`, since we've found that the first dimension tends to reflect term frequency, making the output less informative. Note that if `snip_first_dim=True` then in practice, we output `n_svd_dims-1`-dimensional representations.
    :param n_clusters: the number of clusters to infer.
    :param cluster_on: whether to cluster on utterance or term representations, (corresponding to values `'utts'` or `'terms'`). By default, we infer clusters based on representations of the utterances from the training data, and then assign term and context-utterance representations to the resultant clusters. In some cases (e.g., if utterances are highly unstructured and lengthy) it might be better to cluster term representations first.
    :param model: an existing, fitted `ExpectedContextModelTransformer` object to initialize with (optional)
    :param random_state: the random seed to use in the LSA step (which calls a randomized implementation of SVD)
    :param cluster_random_state: the random seed to use to infer clusters.

    """

    def __init__(
        self,
        context_field,
        output_prefix,
        vect_field,
        context_vect_field=None,
        n_svd_dims=25,
        snip_first_dim=True,
        n_clusters=8,
        cluster_on="utts",
        model=None,
        random_state=None,
        cluster_random_state=None,
    ):
        if model is not None:
            in_model = model.ec_model
        else:
            in_model = None
        self.ec_model = ExpectedContextModel(
            model=in_model,
            n_svd_dims=n_svd_dims,
            snip_first_dim=snip_first_dim,
            n_clusters=n_clusters,
            cluster_on=cluster_on,
            random_state=random_state,
            cluster_random_state=cluster_random_state,
        )
        self.context_field = context_field
        if context_field == "reply_to":
            self.context_func = lambda x: x.reply_to
        else:
            self.context_func = lambda x: x.meta.get(context_field, None)
        self.output_prefix = output_prefix
        self.vect_field = vect_field
        self.context_vect_field = context_vect_field
        if self.context_vect_field is None:
            self.context_vect_field = vect_field

    ### fit functionality

    def fit(self, corpus, y=None, selector=lambda x: True, context_selector=lambda x: True):
        """
        Fits an `ExpectedContextModelTransformer` transformer over training data: derives representations of terms, utterances and contexts,
        range statistics for terms, and a clustering of the resultant representations.

        :param corpus: Corpus containing training data
        :param selector: a boolean function of signature `filter(utterance)` that determines which utterances
            will be considered in the fit step. defaults to using all utterances.
        :param context_selector: a boolean function of signature `filter(utterance)` that determines which context-utterances
            will be considered in the fit step. defaults to using all utterances.
        :return: None
        """

        id_to_idx = corpus.get_vector_matrix(self.vect_field).ids_to_idx
        context_id_to_idx = corpus.get_vector_matrix(self.context_vect_field).ids_to_idx

        ids = []
        context_ids = []
        mapping_ids = []
        context_mapping_ids = []
        for ut in corpus.iter_utterances(selector=selector):
            ids.append(ut.id)
            context_id = self.context_func(ut)
            if context_id is not None:
                try:
                    if context_selector(corpus.get_utterance(context_id)):
                        try:
                            mapping_ids.append(ut.id)
                            context_mapping_ids.append(context_id)
                        except:
                            continue
                except:
                    continue

        for ut in corpus.iter_utterances(selector=context_selector):
            context_ids.append(ut.id)

        id_to_idx = {id: i for i, id in enumerate(ids)}
        context_id_to_idx = {id: i for i, id in enumerate(context_ids)}
        mapping_idxes = [id_to_idx[x] for x in mapping_ids]
        context_mapping_idxes = [context_id_to_idx[x] for x in context_mapping_ids]

        utt_vects = corpus.get_vectors(self.vect_field, ids)
        context_utt_vects = corpus.get_vectors(self.context_vect_field, context_ids)
        mapping_table = np.vstack([mapping_idxes, context_mapping_idxes]).T
        self.mapping_table = mapping_table
        terms = corpus.get_vector_matrix(self.vect_field).columns
        context_terms = corpus.get_vector_matrix(self.context_vect_field).columns
        self.ec_model.fit(
            utt_vects,
            context_utt_vects,
            mapping_table,
            terms,
            context_terms,
            utt_ids=ids,
            context_utt_ids=context_ids,
        )

    def _get_matrix(self, corpus, field, selector):
        ids = [ut.id for ut in corpus.iter_utterances(selector=selector) if field in ut.vectors]
        utt_vects = corpus.get_vectors(field, ids)
        return ids, utt_vects

    def _add_vector(self, corpus, field, ids):
        for id in ids:
            corpus.get_utterance(id).add_vector(field)

    ### transformers

    def transform(self, corpus, selector=lambda x: True):
        """
        Computes vector representations, ranges, and cluster assignments for utterances in a corpus.

        :param corpus: Corpus
        :param selector: a boolean function of signature `filter(utterance)` that determines which utterances
            to transform. defaults to all utterances.
        :return: the Corpus, with per-utterance representations, ranges and cluster assignments.
        """
        ids, utt_vects = self._get_matrix(corpus, self.vect_field, selector)
        utt_reprs = self.ec_model.transform(utt_vects)
        corpus.set_vector_matrix(self.output_prefix + "_repr", matrix=utt_reprs, ids=ids)
        self._add_vector(corpus, self.output_prefix + "_repr", ids)
        self.compute_utt_ranges(corpus, selector)
        self.compute_clusters(corpus, selector)
        return corpus

    def transform_utterance(self, utt):
        """
        Computes vector representation, range, and cluster assignment for a single utterance. Note that the utterance must contain the input representation as a metadata field, specified by what was passed into the constructor as the `vect_field` argument.
        Will write all of these characterizations (including vectors) to the utterance's metadata; attribute names are prefixed with the `output_prefix` constructor argument.

        :param utt: Utterance
        :return: the utterance, with per-utterance representation, range and cluster assignments.
        """

        utt_vect = np.array([utt.meta[self.vect_field]])
        utt_repr = np.array(self.ec_model.transform(utt_vect))
        utt.meta[self.output_prefix + "_repr"] = [float(x) for x in utt_repr[0]]
        utt.meta[self.output_prefix + "_range"] = float(
            self.ec_model.compute_utt_ranges(utt_vect)[0]
        )
        cluster_df = self.ec_model.transform_clusters(utt_repr)
        for col in cluster_df.columns:
            if col == "cluster_dist":
                utt.meta[self.output_prefix + "_clustering." + col] = float(cluster_df.iloc[0][col])
            else:
                utt.meta[self.output_prefix + "_clustering." + col] = cluster_df.iloc[0][col]
        return utt

    def compute_utt_ranges(self, corpus, selector=lambda x: True):
        """
        Computes utterance ranges.

        :param corpus: Corpus
        :param selector: determines which utterances to compute ranges for.

        :return: the Corpus, with per-utterance ranges.
        """
        ids, utt_vects = self._get_matrix(corpus, self.vect_field, selector)
        ranges = self.ec_model.compute_utt_ranges(utt_vects)
        for id, r in zip(ids, ranges):
            corpus.get_utterance(id).meta[self.output_prefix + "_range"] = r
        return corpus

    def transform_context_utts(self, corpus, selector=lambda x: True):
        """
        Computes representations of context-utterances, along with cluster assignments.

        :param corpus: Corpus
        :param selector: determines which utterances to compute representations for.

        :return: the Corpus, with per-utterance representations and cluster assignments.
        """

        ids, context_utt_vects = self._get_matrix(corpus, self.context_vect_field, selector)
        context_utt_reprs = self.ec_model.transform_context_utts(context_utt_vects)
        corpus.set_vector_matrix(
            self.output_prefix + "_context_repr", matrix=context_utt_reprs, ids=ids
        )
        self._add_vector(corpus, self.output_prefix + "_context_repr", ids)
        self.compute_clusters(corpus, selector, is_context=True)
        return corpus

    def fit_clusters(self, n_clusters="default", random_state="default"):
        """
        Infers a clustering of term or utterance representations (specified by the `cluster_on` argument used to initialize the transformer) on the training data originally used to fit the transformer. Can be called to infer a different number of clusters than what was initially specified.

        :param n_clusters: number of clusters to infer. defaults to the number of clusters specified when initializing the transformer.
        :param random_state: random seed used to infer clusters. defaults to the random seed used to initialize the transformer.

        :return: None
        """
        if n_clusters == "default":
            n_clusters = self.ec_model.n_clusters
        if random_state == "default":
            random_state = self.ec_model.cluster_random_state
        self.ec_model.fit_clusters(n_clusters, random_state)

    def compute_clusters(self, corpus, selector=lambda x: True, is_context=False):
        """
        Assigns utterances in a corpus, for which expected context representations have already been computed, to inferred clusters.

        :param corpus: Corpus
        :param selector: determines which utterances to compute clusterings for
        :param is_context: whether to treat input data as utterances, or context-utterances

        :return: a DataFrame containing cluster assignment information for each utterance.
        """
        if is_context:
            ids, reprs = self._get_matrix(corpus, self.output_prefix + "_context_repr", selector)
        else:
            ids, reprs = self._get_matrix(corpus, self.output_prefix + "_repr", selector)
        cluster_df = self.ec_model.transform_clusters(reprs, ids)
        if is_context:
            cluster_field = self.output_prefix + "_context_clustering"
        else:
            cluster_field = self.output_prefix + "_clustering"
        for id, entry in cluster_df.iterrows():
            for k, v in entry.to_dict().items():
                corpus.get_utterance(id).meta[cluster_field + "." + k] = v
        return cluster_df

    ### cluster management

    def set_cluster_names(self, cluster_names):
        """
        Assigns names to inferred clusters. May be called after inspecting the output of `print_clusters`.

        :param cluster_names: a list of names, where `cluster_names[i]` is the name of the cluster with `cluster_id_` `i`.
        :return: None
        """
        self.ec_model.set_cluster_names(cluster_names)

    def get_cluster_names(self):
        """
        Returns the names of the inferred clusters.

        :return: list of cluster names where `cluster_names[i]` is the name of the cluster with `cluster_id_` `i`.
        """
        return self.ec_model.get_cluster_names()

    def print_clusters(self, k=10, max_chars=1000, corpus=None):
        """
        Prints representative terms, utterances and context-utterances for each inferred type. Can be inspected to help interpret the transformer's output.
        By default, will only print out terms and context terms; if the corpus containing the training data is passed in, will output utterances
        and context-utterances as well.

        :param k: number of examples to print out.
        :param max_chars: maximum number of characters per utterance/context-utterance to print. Can be toggled to control the size of the output.
        :param corpus: optional, the corpus that the transformer was trained on. if set, will print example utterances and context-utterances as well as terms.

        :return: None
        """
        n_clusters = self.ec_model.n_clusters
        cluster_obj = self.ec_model.clustering
        for i in range(n_clusters):
            print("CLUSTER", i, self.ec_model.get_cluster_names()[i])
            print("---")
            print("terms")
            term_subset = (
                cluster_obj["terms"][cluster_obj["terms"].cluster_id_ == i]
                .sort_values("cluster_dist")
                .head(k)
            )
            print(term_subset[["cluster_dist"]])
            print()
            print("context terms")
            context_term_subset = (
                cluster_obj["context_terms"][cluster_obj["context_terms"].cluster_id_ == i]
                .sort_values("cluster_dist")
                .head(k)
            )
            print(context_term_subset[["cluster_dist"]])
            print()
            if corpus is None:
                continue
            print()
            print("utterances")
            utt_subset = (
                cluster_obj["utts"][cluster_obj["utts"].cluster_id_ == i]
                .drop_duplicates("cluster_dist")
                .sort_values("cluster_dist")
                .head(k)
            )
            for id, row in utt_subset.iterrows():
                print(">", id, "%.3f" % row.cluster_dist, corpus.get_utterance(id).text[:max_chars])
            print()
            print("context-utterances")
            context_utt_subset = (
                cluster_obj["context_utts"][cluster_obj["context_utts"].cluster_id_ == i]
                .drop_duplicates("cluster_dist")
                .sort_values("cluster_dist")
                .head(k)
            )
            for id, row in context_utt_subset.iterrows():
                print(
                    ">>", id, "%.3f" % row.cluster_dist, corpus.get_utterance(id).text[:max_chars]
                )
            print("\n====\n")

    def print_cluster_stats(self):
        """
        Returns a Pandas dataframe containing the % of terms, context terms, and training utterances/context-utterances that have been assigned to each cluster.

        :return: dataframe containing cluster statistics
        """
        cluster_obj = self.ec_model.clustering
        return pd.concat(
            [
                cluster_obj[k].cluster.value_counts(normalize=True).rename(k).sort_index()
                for k in ["utts", "terms", "context_utts", "context_terms"]
            ],
            axis=1,
        )

    def summarize(self, k=10, max_chars=1000, corpus=None):
        """
        Wrapper function to print inferred clusters and statistics about their sizes.

        :param k: number of examples to print out.
        :param max_chars: maximum number of characters per utterance/context-utterance to print. Can be toggled to control the size of the output.
        :param corpus: optional, the corpus that the transformer was trained on. if set, will print example utterances and context-utterances as well as terms.

        :return: None
        """
        print("STATS")
        print(self.print_cluster_stats())
        print("\nCLUSTERS")
        self.print_clusters(k=k, max_chars=max_chars, corpus=corpus)

    ### getters for representations from training data

    def get_terms(self):
        """
        Gets the names of the terms for which the transformer has computed representations.

        :return: list of terms
        """
        return self.ec_model.terms

    def get_term_ranges(self):
        """
        Gets the range statistics of terms.

        :return: list of term ranges. order corresponds to the ordering of terms returned via `get_terms()`.
        """
        return self.ec_model.term_ranges

    def get_term_reprs(self):
        """
        Gets the derived representations of terms.

        :return: numpy array containing term representations. order of rows corresponds to the ordering of terms returned via `get_terms`.
        """
        return self.ec_model.term_reprs

    def get_context_terms(self):
        """
        Gets the names of the context terms for which the transformer has computed (LSA) representations.

        :return: list of context terms
        """
        return self.ec_model.context_terms

    def get_context_term_reprs(self):
        """
        Gets the derived (LSA) representations of context terms.

        :return: numpy array containing term representations. order of rows corresponds to the ordering of terms returned via `get_context_terms`.
        """
        return self.ec_model.context_term_reprs

    def get_clustering(self):
        """
        Returns a dictionary containing various objects pertaining to the inferred clustering, with fields as follows:

        * `km_obj`: the fitted KMeans object
        * `utts`: a Pandas dataframe of cluster assignments for utterances from the training data
        * `terms`: a dataframe of cluster assignments for terms
        * `context_utts`: dataframe of cluster assignments for context-utterances from the training data
        * `context_terms`: dataframe of cluster assignments for terms.

        :return: dictionary containing clustering information
        """
        return self.ec_model.clustering

    ### loading and dumping

    def load(self, dirname):
        """
        Loads a model from disk.

        :param dirname: directory to read model from
        :return: None
        """
        self.ec_model.load(dirname)

    def dump(self, dirname):
        """
        Writes a model to disk.

        :param dirname: directory to write model to.
        :return: None
        """
        self.ec_model.dump(dirname)


class ExpectedContextModel:
    """
    Model that derives representations of terms and utterances in terms of their conversational context, i.e.,
    context-utterances that occur near an utterance, or utterances containing a term. Typically, the conversational
    context consists of immediate replies ("forwards context") or predecessors ("backwards context"), though
    this can be specified by the user. Can be used in ConvoKit through the `ExpectedContextModelTransformer` transformer;
    see documentation of that transformer for further details.
    """

    def __init__(
        self,
        n_svd_dims=25,
        snip_first_dim=True,
        n_clusters=8,
        context_U=None,
        context_V=None,
        context_s=None,
        model=None,
        context_terms=None,
        cluster_on="utts",
        random_state=None,
        cluster_random_state=None,
    ):
        self.n_svd_dims = n_svd_dims
        self.snip_first_dim = snip_first_dim
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.cluster_random_state = cluster_random_state
        self.cluster_on = cluster_on

        if (context_U is None) and (model is None):
            self.fitted_context = False
        elif model is not None:
            self.set_model(model)
            # self.fitted_context = True
            # self.n_svd_dims = model.n_svd_dims
            # self.context_U = model.context_U
            # self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)
            # self.context_V = model.context_V
            # self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)
            # self.context_s = model.context_s
            # self.context_terms = self._get_default_ids(model.context_terms, len(self.context_V))
        elif context_U is not None:
            self.fitted_context = True
            self.n_svd_dims = context_U.shape[1]
            self.context_U = context_U
            self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)

            self.context_V = context_V
            self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)

            self.context_s = context_s
            self.context_terms = self._get_default_ids(context_terms, len(self.context_V))

        self.terms = None
        self.clustering = {}

    def set_model(self, model):
        self.fitted_context = True
        self.n_svd_dims = model.n_svd_dims
        self.context_U = model.context_U
        self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)
        self.context_V = model.context_V
        self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)
        self.context_s = model.context_s
        self.context_terms = self._get_default_ids(model.context_terms, len(self.context_V))

    def fit_context_utts(self, context_utt_vects, context_terms=None):
        self.context_U, self.context_s, self.context_V = randomized_svd(
            context_utt_vects, n_components=self.n_svd_dims, random_state=self.random_state
        )
        self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)

        self.context_V = self.context_V.T
        self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)

        self.context_terms = self._get_default_ids(context_terms, len(self.context_V))
        self.fitted_context = True

    def fit(
        self,
        utt_vects,
        context_utt_vects,
        utt_context_pairs,
        terms=None,
        context_terms=None,
        refit_context=False,
        fit_clusters=True,
        n_clusters=None,
        cluster_random_state=None,
        utt_ids=None,
        context_utt_ids=None,
    ):
        if (not self.fitted_context) or refit_context:
            self.fit_context_utts(context_utt_vects, context_terms)

        self.terms = self._get_default_ids(terms, utt_vects.shape[1])

        utt_vect_subset = utt_vects[utt_context_pairs[:, 0]]
        context_repr_subset = self.context_U[utt_context_pairs[:, 1]]
        self.term_reprs_full = utt_vect_subset.T * context_repr_subset / self.context_s
        self.term_reprs = self._snip(self.term_reprs_full, snip_first_dim=self.snip_first_dim)
        self.train_utt_reprs = self.transform(utt_vects)

        full_dists = cosine_distances(
            self.term_reprs, self._snip(context_repr_subset, snip_first_dim=self.snip_first_dim)
        )
        weights = normalize(np.array(utt_vect_subset > 0), norm="l1", axis=0)
        clipped_dists = np.clip(full_dists, None, 1)
        self.term_ranges = (clipped_dists * weights.T).sum(axis=1)
        if fit_clusters:
            if self.n_clusters is None:
                self.n_clusters = n_clusters
            if self.cluster_random_state is None:
                self.cluster_random_state = cluster_random_state
            self.fit_clusters(
                self.n_clusters,
                self.cluster_random_state,
                utt_ids=utt_ids,
                context_utt_ids=context_utt_ids,
            )

    def transform(self, utt_vects):
        return self._snip(utt_vects * self.term_reprs_full / self.context_s, self.snip_first_dim)

    def compute_utt_ranges(self, utt_vects):
        return np.dot(normalize(np.array(utt_vects), norm="l1"), self.term_ranges)

    def transform_context_utts(self, context_utt_vects):
        return self._snip(context_utt_vects * self.context_V / self.context_s, self.snip_first_dim)

    def fit_clusters(
        self, n_clusters="default", random_state="default", utt_ids=None, context_utt_ids=None
    ):
        if n_clusters == "default":
            n_clusters = self.n_clusters
        if random_state == "default":
            random_state = self.cluster_random_state
        km_obj = ClusterWrapper(n_clusters=n_clusters, random_state=random_state)
        if self.cluster_on == "terms":
            km_obj.fit(self.term_reprs)
        elif self.cluster_on == "utts":
            km_obj.fit(self.train_utt_reprs)
        self.clustering["km_obj"] = km_obj
        self.clustering["utts"] = km_obj.transform(self.train_utt_reprs, utt_ids)
        self.clustering["terms"] = km_obj.transform(self.term_reprs, self.terms)
        self.clustering["context_utts"] = km_obj.transform(
            self.train_context_reprs, context_utt_ids
        )
        self.clustering["context_terms"] = km_obj.transform(
            self.context_term_reprs, self.context_terms
        )

    def transform_clusters(self, reprs, ids=None):
        return self.clustering["km_obj"].transform(reprs, ids)

    def set_cluster_names(self, cluster_names):
        cluster_names = np.array(cluster_names)
        self.clustering["km_obj"].set_cluster_names(cluster_names)
        for k in ["utts", "terms", "context_utts", "context_terms"]:
            self.clustering[k]["cluster"] = cluster_names[self.clustering[k].cluster_id_]

    def get_cluster_names(self):
        return self.clustering["km_obj"].cluster_names

    def print_clusters(self, k=10, max_chars=1000, text_df=None):
        n_clusters = self.n_clusters
        cluster_obj = self.clustering
        cluster_names = self.get_cluster_names()
        for i in range(n_clusters):
            print("CLUSTER", i, cluster_names[i])
            print("---")
            print("terms")
            term_subset = (
                cluster_obj["terms"][cluster_obj["terms"].cluster_id_ == i]
                .sort_values("cluster_dist")
                .head(k)
            )
            print(term_subset[["cluster_dist"]])
            print()
            print("context terms")
            context_term_subset = (
                cluster_obj["context_terms"][cluster_obj["context_terms"].cluster_id_ == i]
                .sort_values("cluster_dist")
                .head(k)
            )
            print(context_term_subset[["cluster_dist"]])
            print()
            if text_df is None:
                continue
            print()
            print("utterances")
            utt_subset = (
                cluster_obj["utts"][cluster_obj["utts"].cluster_id_ == i]
                .drop_duplicates("cluster_dist")
                .sort_values("cluster_dist")
                .head(k)
            )
            for id, row in utt_subset.iterrows():
                print(">", id, "%.3f" % row.cluster_dist, text_df.loc[id].text[:max_chars])
            print()
            print("context-utterances")
            context_utt_subset = (
                cluster_obj["context_utts"][cluster_obj["context_utts"].cluster_id_ == i]
                .drop_duplicates("cluster_dist")
                .sort_values("cluster_dist")
                .head(k)
            )
            for id, row in context_utt_subset.iterrows():
                print(">>", id, "%.3f" % row.cluster_dist, text_df.loc[id].text[:max_chars])
            print("\n====\n")

    def print_cluster_stats(self):
        cluster_obj = self.clustering
        return pd.concat(
            [
                cluster_obj[k].cluster.value_counts(normalize=True).rename(k).sort_index()
                for k in ["utts", "terms", "context_utts", "context_terms"]
            ],
            axis=1,
        )

    def load(self, dirname):
        with open(os.path.join(dirname, "meta.json")) as f:
            meta_dict = json.load(f)
        self.n_svd_dims = meta_dict["n_svd_dims"]
        self.random_state = meta_dict["random_state"]
        self.snip_first_dim = meta_dict["snip_first_dim"]
        self.cluster_on = meta_dict["cluster_on"]

        self.context_U = np.load(os.path.join(dirname, "context_U.npy"), allow_pickle=True)
        self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)
        self.context_V = np.load(os.path.join(dirname, "context_V.npy"), allow_pickle=True)
        self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)
        self.context_s = np.load(os.path.join(dirname, "context_s.npy"), allow_pickle=True)
        self.context_terms = np.load(os.path.join(dirname, "context_terms.npy"), allow_pickle=True)
        self.terms = np.load(os.path.join(dirname, "terms.npy"), allow_pickle=True)
        self.term_reprs_full = np.matrix(
            np.load(os.path.join(dirname, "term_reprs.npy"), allow_pickle=True)
        )
        self.term_reprs = self._snip(self.term_reprs_full, self.snip_first_dim)
        self.term_ranges = np.load(os.path.join(dirname, "term_ranges.npy"), allow_pickle=True)
        self.train_utt_reprs = np.load(
            os.path.join(dirname, "train_utt_reprs.npy"), allow_pickle=True
        )

        try:
            km_obj = ClusterWrapper(self.n_clusters)
            km_obj.load(dirname)
            self.clustering["km_obj"] = km_obj
            for k in ["utts", "terms", "context_utts", "context_terms"]:
                self.clustering[k] = pd.read_csv(
                    os.path.join(dirname, "clustering_%s.tsv" % k), sep="\t", index_col=0
                )
        except Exception as e:
            pass

    def dump(self, dirname, dump_clustering=True):
        try:
            os.mkdir(dirname)
        except:
            pass
        with open(os.path.join(dirname, "meta.json"), "w") as f:
            json.dump(
                {
                    "n_svd_dims": self.n_svd_dims,
                    "random_state": self.random_state,
                    "snip_first_dim": self.snip_first_dim,
                    "cluster_on": self.cluster_on,
                },
                f,
            )
        for name, obj in [
            ("context_U", self.context_U),
            ("context_V", self.context_V),
            ("context_s", self.context_s),
            ("context_terms", self.context_terms),
            ("terms", self.terms),
            ("term_reprs", self.term_reprs_full),
            ("term_ranges", self.term_ranges),
            ("train_utt_reprs", self.train_utt_reprs),
        ]:
            np.save(os.path.join(dirname, name + ".npy"), obj)
        if dump_clustering and (len(self.clustering) > 0):
            self.clustering["km_obj"].dump(dirname)
            for k in ["utts", "terms", "context_utts", "context_terms"]:
                self.clustering[k].to_csv(os.path.join(dirname, "clustering_%s.tsv" % k), sep="\t")

    def _get_default_ids(self, ids, n):
        if ids is None:
            return np.arange(n)
        else:
            return ids

    def _snip(self, vects, snip_first_dim=True, dim=None):
        if dim is None:
            dim = vects.shape[1]
        return normalize(np.array(vects[:, int(snip_first_dim) : dim]))


class ClusterWrapper:
    """
    Wrapper that performs K-Means clustering. Handles model loading and dumping,
    formats clustering output as dataframes for convenience, and keeps track of
    names that an end-user can assign to clusters.
    """

    def __init__(self, n_clusters, cluster_names=None, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.cluster_names = np.arange(n_clusters)
        if cluster_names is not None:
            self.cluster_names = cluster_names
        self.km_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.km_df = None

    def fit(self, vects, ids=None):
        self.km_model.fit(vects)
        self.km_df = self.transform(vects, ids)

    def set_cluster_names(self, names):
        self.cluster_names = np.array(names)
        if self.km_df is not None:
            self.km_df["cluster"] = self.cluster_names[self.km_df.cluster_id_]

    def transform(self, vects, ids=None):
        if ids is None:
            ids = np.arange(len(vects))
        km_df = self._get_km_assignment_df(self.km_model, vects, ids, self.cluster_names)
        return km_df

    def _get_km_assignment_df(self, km, vects, ids, cluster_names):
        dists = km.transform(vects)
        min_dist = dists[np.arange(len(dists)), dists.argmin(axis=1)]
        cluster_assigns = km.predict(vects)
        cluster_assign_names = cluster_names[cluster_assigns]
        df = pd.DataFrame(
            {
                "index": ids,
                "cluster_id_": cluster_assigns,
                "cluster": cluster_assign_names,
                "cluster_dist": min_dist,
            }
        ).set_index("index")
        return df

    def load(self, dirname):
        with open(os.path.join(dirname, "cluster_meta.json")) as f:
            meta_dict = json.load(f)
        self.n_clusters = meta_dict["n_clusters"]
        self.random_state = meta_dict["random_state"]

        self.km_df = pd.read_csv(os.path.join(dirname, "cluster_km_df.tsv"), sep="\t", index_col=0)
        self.cluster_names = np.load(os.path.join(dirname, "cluster_names.npy"), allow_pickle=True)
        self.km_model = joblib.load(os.path.join(dirname, "km_model.joblib"))

    def dump(self, dirname):
        try:
            os.mkdir(dirname)
        except:
            pass
        with open(os.path.join(dirname, "cluster_meta.json"), "w") as f:
            json.dump({"n_clusters": self.n_clusters, "random_state": self.random_state}, f)
        self.km_df.to_csv(os.path.join(dirname, "cluster_km_df.tsv"), sep="\t")
        np.save(os.path.join(dirname, "cluster_names.npy"), self.cluster_names)
        joblib.dump(self.km_model, os.path.join(dirname, "km_model.joblib"))
