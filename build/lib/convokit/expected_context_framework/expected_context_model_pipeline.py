from convokit.expected_context_framework import (
    ColNormedTfidfTransformer,
    ExpectedContextModelTransformer,
    DualContextWrapper,
)

from convokit.transformer import Transformer
from convokit.convokitPipeline import ConvokitPipeline
from convokit.text_processing import TextProcessor
from convokit import Utterance, Speaker

import os


class ExpectedContextModelPipeline(Transformer):
    """
    Wrapper class implementing a pipeline that derives characterizations of terms and utterances in terms of their conversational context. The pipeline handles the following steps:

    * processing input text (via a pipeline supplied by the user in the `text_pipe` argument);
    * transforming text to input representation (via `ColNormedTfidfTransformer`);
    * deriving characterizations (via `ExpectedContextModelTransformer`)

    The `ColNormedTfidfTransformer` components are stored as the `tfidf_model` and `context_tfidf_model` attributes of the class; the `ExpectedContextModelTransformer` is stored as the `ec_model` attribute.

    For further details, see the `ColNormedTfidfTransformer` and `ExpectedContextModelTransformer` classes.

    :param context_field: the name of an utterance-level attribute containing the ID of the corresponding context-utterance. in particular, to use immediate predecessors as context, set `context_field` to `'reply_to'`. as another example, to use immediate replies, provided that utterances contain an attribute `next_id` containing the ID of their reply, set `context_field` to `'next_id'`.
    :param output_prefix: the name of the attributes and vectors to write to in the transform step. the transformer outputs several fields, which will be prefixed with the given string.
    :param text_field: the  name of the utterance-level attribute containing the text to use as input.
    :param context_text_field: the  name of the utterance-level attribute containing the text to use as input for context-utterances. by default, is equivalent to `text_field`.
    :param text_pipe: a `convokitPipeline` object used to compute the contents of `text_field`. defaults to populating the `text_field` attribute of each utterance utt with `utt.text`.
    :param context_text_pipe: a `convokitPipeline` object used to compute the contents of `context_text_field`; by default equivalent to `text_pipe`
    :param tfidf_params: a dictionary specifying parameters to be passed to the `ColNormedTfidfTransformer` object to compute input representations of utterances.
    :param context_tfidf_parms: a dictionary specifying parameters to be passed to the `ColNormedTfidfTransformer` object to compute input representations of context-utterances. equivalent to `tfidf_params` by default.
    :param share_tfidf_models: whether or not to use the same `ColNormedTfidfTransformer` for both utterances and context-utterances. defaults to `True`.
    :param min_terms: the minimum number of terms in the vocabulary, derived by `ColNormedTfidfTransformer`, that an utterance must contain for it to be considered in fitting and transforming the underlying `ExpectedContextModelTransformer` object. defaults to 0, meaning the transformer will consider all utterances.
    :param context_min_terms: minimum number of terms in the vocabulary for a context-utterance to be considered in fitting and transforming the underlying `ExpectedContextModelTransformer` object. equivalent to `min_terms` by default.
    :param n_svd_dims: the dimensionality of the representations to derive (via LSA/SVD).
    :param snip_first_dim: whether or not to remove the first dimension of the derived representations. by default this is set to `True`, since we've found that the first dimension tends to reflect term frequency, making the output less informative. Note that if `snip_first_dim=True` then in practice, we output `n_svd_dims-1`-dimensional representations.
    :param n_clusters: the number of clusters to infer.
    :param cluster_on: whether to cluster on utterance or term representations, (corresponding to values `'utts'` or `'terms'`). By default, we infer clusters based on representations of the utterances from the training data, and then assign term and context-utterance representations to the resultant clusters. In some cases (e.g., if utterances are highly unstructured and lengthy) it might be better to cluster term representations first.
    :param ec_model: an existing, fitted `ExpectedContextModelPipeline` object to initialize with (optional)
    :param random_state: the random seed to use in the LSA step (which calls a randomized implementation of SVD)
    :param cluster_random_state: the random seed to use to infer clusters.

    """

    def __init__(
        self,
        context_field,
        output_prefix,
        text_field,
        context_text_field=None,
        text_pipe=None,
        context_text_pipe=None,
        tfidf_params={},
        context_tfidf_params=None,
        share_tfidf_models=True,
        min_terms=0,
        context_min_terms=None,
        n_svd_dims=25,
        snip_first_dim=True,
        n_clusters=8,
        cluster_on="utts",
        ec_model=None,
        random_state=None,
        cluster_random_state=None,
    ):
        self.context_field = context_field
        self.output_prefix = output_prefix

        self.vect_field = "col_normed_tfidf"
        self.share_tfidf_models = share_tfidf_models

        if share_tfidf_models:
            self.context_vect_field = self.vect_field
        else:
            self.context_vect_field = "context_col_normed_tfidf"

        self.text_field = text_field
        if context_text_field is None:
            self.context_text_field = text_field
        else:
            self.context_text_field = context_text_field

        if text_pipe is None:
            self.text_pipe = ConvokitPipeline(
                [("text_pipe", TextProcessor(output_field=self.text_field, proc_fn=lambda x: x))]
            )
        else:
            self.text_pipe = text_pipe

        if context_text_pipe is None:
            self.context_text_pipe = self.text_pipe
        else:
            self.context_text_pipe = context_text_pipe

        self.tfidf_params = tfidf_params
        if context_tfidf_params is None:
            self.context_tfidf_params = tfidf_params
        else:
            self.context_tfidf_params = context_tfidf_params

        self.min_terms = min_terms
        if context_min_terms is None:
            self.context_min_terms = min_terms
        else:
            self.context_min_terms = context_min_terms

        if ec_model is not None:
            in_model = ec_model.ec_model
        else:
            in_model = None
        self.ec_model = ExpectedContextModelTransformer(
            context_field=context_field,
            output_prefix=output_prefix,
            vect_field=self.vect_field,
            context_vect_field=self.context_vect_field,
            model=in_model,
            n_svd_dims=n_svd_dims,
            snip_first_dim=snip_first_dim,
            n_clusters=n_clusters,
            cluster_on=cluster_on,
            random_state=random_state,
            cluster_random_state=cluster_random_state,
        )

        self.tfidf_model = ColNormedTfidfTransformer(
            input_field=self.text_field, output_field=self.vect_field, **self.tfidf_params
        )
        if not share_tfidf_models:
            self.context_tfidf_model = ColNormedTfidfTransformer(
                input_field=self.context_text_field,
                output_field=self.context_vect_field,
                **self.context_tfidf_params
            )
        else:
            self.context_tfidf_model = self.tfidf_model

    def fit(self, corpus, y=None, selector=lambda x: True, context_selector=lambda x: True):
        """
        Fits an `ExpectedContextModelPipeline` over training data: derives input and latent representations of terms, utterances and contexts,
        range statistics for terms, and a clustering of the resultant representations.

        :param corpus: Corpus containing training data
        :param selector: a boolean function of signature `filter(utterance)` that determines which utterances will be considered in the fit step. defaults to using all utterances, subject to `min_terms` parameter passed at initialization.
        :param context_selector: a boolean function of signature `filter(utterance)` that determines which context-utterances will be considered in the fit step. defaults to using all utterances, subject to `context_min_terms` parameter passed at initialization.
        :return: None
        """

        self.text_pipe.fit_transform(corpus)
        if not self.share_tfidf_models:
            self.context_text_pipe.fit_transform(corpus)
        self.tfidf_model.fit_transform(corpus, selector=selector)
        if not self.share_tfidf_models:
            self.context_tfidf_model.fit_transform(corpus, selector=context_selector)
        self.ec_model.fit(
            corpus,
            selector=lambda x: selector(x)
            and (x.meta.get(self.vect_field + "__n_feats", 0) >= self.min_terms),
            context_selector=lambda x: context_selector(x)
            and (x.meta.get(self.context_vect_field + "__n_feats", 0) >= self.context_min_terms),
        )

    def transform(self, corpus, y=None, selector=lambda x: True):
        """
        Computes vector representations, ranges, and cluster assignments for utterances in a corpus.

        :param corpus: Corpus
        :param selector: a boolean function of signature `filter(utterance)` that determines which utterances to transform.
        :return: the Corpus, with per-utterance representations, ranges and cluster assignments.
        """
        _ = self.text_pipe.transform(corpus)
        _ = self.tfidf_model.transform(corpus, selector=selector)
        _ = self.ec_model.transform(
            corpus,
            selector=lambda x: selector(x)
            and (x.meta.get(self.vect_field + "__n_feats", 0) >= self.min_terms),
        )
        return corpus

    def transform_utterance(self, utt):
        """
        Computes vector representation, range, and cluster assignment for a single utterance, which can be a ConvoKit Utterance or a string.
        Will return an Utterance object a nd write all of these characterizations (including vectors) to the utterance's metadata; attribute names are prefixed with the `output_prefix` constructor argument.

        :param utt: Utterance or string
        :return: the utterance, with per-utterance representation, range and cluster assignments.
        """
        if isinstance(utt, str):
            utt = Utterance(text=utt, speaker=Speaker())
        self.text_pipe.transform_utterance(utt)
        self.tfidf_model.transform_utterance(utt)
        return self.ec_model.transform_utterance(utt)

    def summarize(self, k=10, max_chars=1000, corpus=None):
        """
        Prints inferred clusters and statistics about their sizes.

        :param k: number of examples to print out.
        :param max_chars: maximum number of characters per utterance/context-utterance to print. Can be toggled to control the size of the output.
        :param corpus: optional, the corpus that the transformer was trained on. if set, will print example utterances and context-utterances as well as terms.

        :return: None
        """
        self.ec_model.summarize(k, max_chars, corpus)

    def set_cluster_names(self, names):
        """
        Assigns names to inferred clusters. May be called after inspecting the output of `print_clusters`.

        :param cluster_names: a list of names, where `cluster_names[i]` is the name of the cluster with `cluster_id_` `i`.
        :return: None
        """
        self.ec_model.set_cluster_names(names)

    def get_cluster_names(self):
        """
        Returns the names of the inferred clusters.

        :return: list of cluster names where `cluster_names[i]` is the name of the cluster with `cluster_id_` `i`.
        """
        return self.ec_model.get_cluster_names()

    def get_terms(self):
        """
        Gets the names of the terms for which the transformer has computed representations.

        :return: list of terms
        """
        return self.ec_model.get_terms()

    def load(self, dirname, model_dirs=None):
        """
        Loads a model from disk.

        :param dirname: directory to read model from
        :param model_dirs: optional list containing the directories (relative to `dirname`) in which each component is stored. the order of the list is as follows: [the `ExpectedContextModelTransformer`, the utterance `ColNormedTfidfTransformer`, the context-utterance `ColNormedTfidfTransformer` (if `share_tfidf_models` is set to `False` at initialization)]. defaults to `['ec_model', 'tfidf_model', 'context_tfidf_model']`.
        :return: None
        """
        if model_dirs is None:
            model_dirs = ["ec_model", "tfidf_model", "context_tfidf_model"]

        self.tfidf_model.load(os.path.join(dirname, model_dirs[1]))
        if not self.share_tfidf_models:
            self.context_tfidf_model.load(os.path.join(dirname, model_dirs[2]))
        else:
            self.context_tfidf_model = self.tfidf_model
        self.ec_model.load(os.path.join(dirname, model_dirs[0]))

    def dump(self, dirname):
        """
        Writes a model to disk.

        :param dirname: directory to write model to.
        :return: None
        """
        try:
            os.mkdir(dirname)
        except:
            pass
        self.tfidf_model.dump(os.path.join(dirname, "tfidf_model"))
        if not self.share_tfidf_models:
            self.context_tfidf_model.dump(os.path.join(dirname, "context_tfidf_model"))
        self.ec_model.dump(os.path.join(dirname, "ec_model"))


class DualContextPipeline(Transformer):
    """
    Wrapper class implementing a pipeline that derives characterizations of terms and utterances in terms of two choices of conversational context. The pipeline handles the following steps:

    * processing input text (via a pipeline supplied by the user in the `text_pipe` argument);
    * transforming text to input representation (via `ColNormedTfidfTransformer`);
    * deriving characterizations (via `DualContextWrapper`)

    The `ColNormedTfidfTransformer` components are stored as the `tfidf_model` and `context_tfidf_model` attributes of the class; the `DualContextWrapper` is stored as the `dualmodel` attribute.

    For further details, see the `ColNormedTfidfTransformer` and `DualContextWrapper` classes.

    :param context_field: the name of an utterance-level attribute containing the ID of the corresponding context-utterance. in particular, to use immediate predecessors as context, set `context_field` to `'reply_to'`. as another example, to use immediate replies, provided that utterances contain an attribute `next_id` containing the ID of their reply, set `context_field` to `'next_id'`.
    :param output_prefixes: list containing the name of the attributes and vectors that the `DualContextWrapper` component will write to in the transform step.
    :param text_field: the  name of the utterance-level attribute containing the text to use as input.
    :param context_text_field: the  name of the utterance-level attribute containing the text to use as input for context-utterances. by default, is equivalent to `text_field`.
    :param wrapper_output_prefix: the metadata fields where the utterance-level orientation and shift statistics are stored. By default, these attributes are stored as `orn` and `shift` in the metadata; if `wrapper_output_prefix` is specified, then they are stored as `<wrapper_output_prefix>_orn` (orientation) and `<wrapper_output_prefix>_shift` (shift).
    :param text_pipe: a `convokitPipeline` object used to compute the contents of `text_field`. defaults to populating the `text_field` attribute of each utterance utt with `utt.text`.
    :param context_text_pipe: a `convokitPipeline` object used to compute the contents of `context_text_field`; by default equivalent to `text_pipe`
    :param tfidf_params: a dictionary specifying parameters to be passed to the `ColNormedTfidfTransformer` object to compute input representations of utterances.
    :param context_tfidf_parms: a dictionary specifying parameters to be passed to the `ColNormedTfidfTransformer` object to compute input representations of context-utterances. equivalent to `tfidf_params` by default.
    :param share_tfidf_models: whether or not to use the same `ColNormedTfidfTransformer` for both utterances and context-utterances. defaults to `True`.
    :param min_terms: the minimum number of terms in the vocabulary, derived by `ColNormedTfidfTransformer`, that an utterance must contain for it to be considered in fitting and transforming the underlying `ExpectedContextModelTransformer` object. defaults to 0, meaning the transformer will consider all utterances.
    :param context_min_terms: minimum number of terms in the vocabulary for a context-utterance to be considered in fitting and transforming the underlying `ExpectedContextModelTransformer` object. equivalent to `min_terms` by default.
    :param n_svd_dims: the dimensionality of the representations to derive (via LSA/SVD).
    :param snip_first_dim: whether or not to remove the first dimension of the derived representations. by default this is set to `True`, since we've found that the first dimension tends to reflect term frequency, making the output less informative. Note that if `snip_first_dim=True` then in practice, we output `n_svd_dims-1`-dimensional representations.
    :param n_clusters: the number of clusters to infer.
    :param cluster_on: whether to cluster on utterance or term representations, (corresponding to values `'utts'` or `'terms'`). By default, we infer clusters based on representations of the utterances from the training data, and then assign term and context-utterance representations to the resultant clusters. In some cases (e.g., if utterances are highly unstructured and lengthy) it might  be better to cluster term representations first.
    :param random_state: the random seed to use in the LSA step (which calls a randomized implementation of SVD)
    :param cluster_random_state: the random seed to use to infer clusters.

    """

    def __init__(
        self,
        context_fields,
        output_prefixes,
        text_field,
        context_text_field=None,
        wrapper_output_prefix="",
        text_pipe=None,
        context_text_pipe=None,
        tfidf_params={},
        context_tfidf_params=None,
        share_tfidf_models=True,
        min_terms=0,
        context_min_terms=None,
        n_svd_dims=25,
        snip_first_dim=True,
        n_clusters=8,
        cluster_on="utts",
        random_state=None,
        cluster_random_state=None,
    ):
        self.vect_field = "col_normed_tfidf"
        self.share_tfidf_models = share_tfidf_models

        if share_tfidf_models:
            self.context_vect_field = self.vect_field
        else:
            self.context_vect_field = "context_col_normed_tfidf"

        self.text_field = text_field
        if context_text_field is None:
            self.context_text_field = text_field
        else:
            self.context_text_field = context_text_field

        if text_pipe is None:
            self.text_pipe = ConvokitPipeline(
                [("text_pipe", TextProcessor(output_field=self.text_field, proc_fn=lambda x: x))]
            )
        self.text_pipe = text_pipe
        self.text_pipe.steps[-1][1].output_field = self.text_field

        if context_text_pipe is None:
            self.context_text_pipe = self.text_pipe
        else:
            self.context_text_pipe = context_text_pipe
            self.context_text_pipe.steps[-1][1].output_field = self.context_text_field

        self.tfidf_params = tfidf_params
        if context_tfidf_params is None:
            self.context_tfidf_params = tfidf_params
        else:
            self.context_tfidf_params = context_tfidf_params

        self.min_terms = min_terms
        if context_min_terms is None:
            self.context_min_terms = min_terms
        else:
            self.context_min_terms = context_min_terms

        self.dualmodel = DualContextWrapper(
            context_fields=context_fields,
            output_prefixes=output_prefixes,
            vect_field=self.vect_field,
            context_vect_field=self.context_vect_field,
            wrapper_output_prefix=wrapper_output_prefix,
            n_svd_dims=n_svd_dims,
            snip_first_dim=snip_first_dim,
            n_clusters=n_clusters,
            cluster_on=cluster_on,
            random_state=random_state,
            cluster_random_state=cluster_random_state,
        )

        self.tfidf_model = ColNormedTfidfTransformer(
            input_field=self.text_field, output_field=self.vect_field, **self.tfidf_params
        )
        if not share_tfidf_models:
            self.context_tfidf_model = ColNormedTfidfTransformer(
                input_field=self.context_text_field,
                output_field=self.context_vect_field,
                **self.context_tfidf_params
            )
        else:
            self.context_tfidf_model = self.tfidf_model

    def fit(self, corpus, y=None, selector=lambda x: True, context_selector=lambda x: True):
        """
        Fits the model over training data.

        :param corpus: Corpus containing training data
        :param selector: a boolean function of signature `filter(utterance)` that determines which utterances will be considered in the fit step. defaults to using all utterances, subject to `min_terms` parameter passed at initialization.
        :param context_selector: a boolean function of signature `filter(utterance)` that determines which context-utterances will be considered in the fit step. defaults to using all utterances, subject to `context_min_terms` parameter passed at initialization.
        :return: None
        """
        self.text_pipe.fit_transform(corpus)
        if not self.share_tfidf_models:
            self.context_text_pipe.fit_transform(corpus)
        self.tfidf_model.fit_transform(corpus, selector=selector)
        if not self.share_tfidf_models:
            self.context_tfidf_model.fit_transform(corpus, selector=context_selector)
        self.dualmodel.fit(
            corpus,
            selector=lambda x: selector(x)
            and (x.meta.get(self.vect_field + "__n_feats", 0) >= self.min_terms),
            context_selector=lambda x: context_selector(x)
            and (x.meta.get(self.context_vect_field + "__n_feats", 0) >= self.context_min_terms),
        )

    def transform(self, corpus, y=None, selector=lambda x: True):
        """
        Computes vector representations, and statistics for utterances in a corpus, using the `DualContextWrapper` component.

        :param corpus: Corpus
        :param selector: a boolean function of signature `filter(utterance)` that determines which utterances to transform. defaults to all utterances.
        :return: the Corpus, with per-utterance attributes.
        """
        _ = self.text_pipe.transform(corpus)
        _ = self.tfidf_model.transform(corpus, selector=selector)
        _ = self.dualmodel.transform(
            corpus,
            selector=lambda x: selector(x)
            and (x.meta.get(self.vect_field + "__n_feats", 0) >= self.min_terms),
        )
        return corpus

    def transform_utterance(self, utt):
        """
        Computes representations and statistics for a single utterance, which can be a ConvoKit Utterance or a string.
        Will return an Utterance object a nd write all of these characterizations (including vectors) to the utterance's metadata; attribute names are prefixed with the `output_prefix` constructor argument.

        :param utt: Utterance or string
        :return: the utterance, with per-utterance representation, range and cluster assignments.
        """
        if isinstance(utt, str):
            utt = Utterance(text=utt, speaker=Speaker())
        self.text_pipe.transform_utterance(utt)
        self.tfidf_model.transform_utterance(utt)
        return self.dualmodel.transform_utterance(utt)

    def summarize(self, k=10, max_chars=1000, corpus=None):
        """
        Prints inferred clusters and statistics about their sizes, for each component in the underlying `DualContextWrapper`.

        :param k: number of examples to print out.
        :param max_chars: maximum number of characters per utterance/context-utterance to print. Can be toggled to control the size of the output.
        :param corpus: optional, the corpus that the transformer was trained on. if set, will print example utterances and context-utterances as well as terms.

        :return: None
        """
        self.dualmodel.summarize(k, max_chars, corpus)

    def get_terms(self):
        """
        Gets the names of the terms for which the transformer has computed representations.

        :return: list of terms
        """
        return self.dualmodel.get_terms()

    def get_term_df(self):
        """
        Gets a Pandas dataframe containing term-level statistics computed by the transformer (shift, orientation, ranges)

        :return: dataframe of term-level statistics
        """
        return self.dualmodel.get_term_df()

    def load(self, dirname, model_dirs=None):
        """
        Loads a model from disk.

        :param dirname: directory to read model from
        :param model_dirs: optional list containing the directories (relative to `dirname`) in which each component is stored. the order of the list is as follows: [the `DualContextWrapper` components, the utterance `ColNormedTfidfTransformer`, the context-utterance `ColNormedTfidfTransformer` (if `share_tfidf_models` is set to `False` at initialization)]. defaults to `[output_prefixes[0], output_prefixes[1], 'tfidf_model', 'context_tfidf_model']` where `output_prefixes` is passed at initialization.
        :return: None
        """
        if model_dirs is None:
            model_dirs = self.dualmodel.output_prefixes + ["tfidf_model", "context_tfidf_model"]

        self.tfidf_model.load(os.path.join(dirname, model_dirs[2]))
        if not self.share_tfidf_models:
            self.context_tfidf_model.load(os.path.join(dirname, model_dirs[3]))
        else:
            self.context_tfidf_model = self.tfidf_model
        self.dualmodel.load(dirname, model_dirs[:2])

    def dump(self, dirname):
        """
        Writes a model to disk.

        :param dirname: directory to write model to.
        :return: None
        """
        self.dualmodel.dump(dirname)
        try:
            os.mkdir(os.path.join(dirname, "tfidf_model"))
        except:
            pass
        self.tfidf_model.dump(os.path.join(dirname, "tfidf_model"))
        if not self.share_tfidf_models:
            self.context_tfidf_model.dump(os.path.join(dirname, "context_tfidf_model"))
