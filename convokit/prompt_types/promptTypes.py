import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import joblib

from convokit.transformer import Transformer


class PromptTypes(Transformer):
    """
    Model that infers a vector representation of utterances in terms of the responses that similar utterances tend to
    prompt, as well as types of rhetorical intentions encapsulated by utterances in a corpus, in terms of their
    anticipated responses (operationalized as k-means clusters of vectors).

    Under the surface, the model takes as input pairs of prompts and responses during the fit step. In this stage the
    following subcomponents are involved:
        1. a prompt embedding model that will learn the vector representations;
        2. a prompt type model that learns a clustering of these representations.

    The model can transform individual (unpaired) utterances in the transform step. While the focus is on representing
    properties of prompts, as a side-effect the model can also compute representations that encapsulate properties of
    responses and assign responses to prompt types (as "typical responses" to the prompts in that type).

    Internally, the model contains the following elements:
        * prompt_embedding_model: stores models that compute the vector representations. includes tf-idf models that convert the prompt and response input to term document matrices, an SVD model that produces a low-dimensional representation of responses and prompts, and vector representations of prompt and response terms
        * type_models: stores kmeans models along with type assignments of prompt and response terms
        * train_results: stores the vector representations of the corpus used to train the model in the fit step
        * train_types: stores the type assignments of the corpus used in the fit step

    The transformer will output several attributes of an utterance (names prefixed with <output_field>__). If the utterance is a prompt (in the default case, if it has a response), then the following will be outputted.
        * prompt_repr: a vector representation of the utterance (stored as a corpus-wide matrix, or in the metadata of an individual utterance if `transform_utterance` is called)
        * prompt_dists.<number of types>: a vector storing the distance between the utterance vector and the centroid of each k-means cluster (stored as a corpus-wide matrix, or in the metadata of an individual utterance if `transform_utterance` is called)
        * prompt_type.<number of types>: the index of the type the utterance is assigned to
        * prompt_type_dist.<number of types>: the distance from the vector representation to the centroid of the assigned type
    If the utterance is a response to a previous utterance, then the utterance will also be annotated an analogous set of attributes denoting its response representation and type.
    For downstream tasks, a reasonable first step is to only look at the prompt-side representations.

    For an end-to-end implementation that runs several default values of the parameters, see the `PromptTypeWrapper` module.

    :param prompt_field: the name of the attribute of prompts to use as input to fit.
    :param reference_field: the name of the attribute of responses to use as input to fit. a reasonable choice is to set to
        the same value as prompt_field.
    :param output_field: the name of the attribute to write to in the transform step. the transformer outputs several
        fields, as listed above.
    :param n_types: the number of types to infer. defaults to 8.
    :param prompt_transform_field: the name of the attribute of prompts to use as input to transform; defaults to the
        same attribute as in fit.
    :param reference_transform_field: the name of the attribute of responses to use as input to transform; defaults to the
        same attribute as in fit.

    :param prompt__tfidf_min_df: the minimum frequency of prompt terms to use. can be specified as a fraction or as an
        absolute count, defaults to 100.
    :param prompt__tfidf_max_df: the maximum frequency of prompt terms to use. can be specified as a fraction or as an
        absolute count, defaults to 0.1. Setting higher is more permissive, but may result in many stopword-like terms
        adding noise to the model.
    :param reference__tfidf_min_df: the minimum frequency of response terms to use. can be specified as a fraction or as an
        absolute count, defaults to 100.
    :param reference__tfidf_max_df: the maximum frequency of response terms to use. can be specified as a fraction or as an
        absolute count, defaults to 0.1.
    :param snip_first_dim: whether or not to remove the first SVD dimension (which may add noise to the model; typically
        this reflects frequency rather than any semantic interpretation). defaults to `True`.
    :param svd__n_components: the number of SVD dimensions to use, defaults to 25. higher values result in richer
        vector representations, perhaps at the cost of the model learning overly-specific types.
    :param max_dist: the maximum distance between a vector representation of an utterance and the cluster centroid; a
        cluster whose distance to all centroids is above this cutoff will get assigned to a null type, denoted by -1.
        Defaults to 0.9.
    :param random_state: the random seed to use.
    :param verbosity: frequency of status messages.
    """

    def __init__(
        self,
        prompt_field,
        reference_field,
        output_field,
        n_types=8,
        prompt_transform_field=None,
        reference_transform_field=None,
        prompt__tfidf_min_df=100,
        prompt__tfidf_max_df=0.1,
        reference__tfidf_min_df=100,
        reference__tfidf_max_df=0.1,
        snip_first_dim=True,
        svd__n_components=25,
        max_dist=0.9,
        random_state=None,
        verbosity=0,
    ):
        self.prompt_embedding_model = {}
        self.type_models = {}
        self.train_results = {}
        self.train_types = {}

        self.prompt_field = prompt_field
        self.reference_field = reference_field

        self.prompt_transform_field = (
            prompt_transform_field if prompt_transform_field is not None else self.prompt_field
        )
        self.reference_transform_field = (
            reference_transform_field
            if reference_transform_field is not None
            else self.reference_field
        )

        self.output_field = output_field

        self.prompt__tfidf_min_df = prompt__tfidf_min_df
        self.prompt__tfidf_max_df = prompt__tfidf_max_df
        self.reference__tfidf_min_df = reference__tfidf_min_df
        self.reference__tfidf_max_df = reference__tfidf_max_df
        self.snip_first_dim = snip_first_dim
        self.svd__n_components = svd__n_components
        self.default_n_types = n_types
        self.random_state = random_state
        self.max_dist = max_dist
        self.verbosity = verbosity

    def fit(
        self, corpus, y=None, prompt_selector=lambda utt: True, reference_selector=lambda utt: True
    ):
        """
        Fits a PromptTypes model for a corpus -- that is, learns latent representations of prompt and response terms, as well as prompt types.

        :param corpus: Corpus
        :param prompt_selector: a boolean function of signature `filter(utterance)` that determines which utterances will be considered as prompts in the fit step. defaults to using all utterances which have a response.
        :param reference_selector: a boolean function of signature `filter(utterance)` that determines which utterances will be considered as responses in the fit step. defaults to using all utterances which are responses to a prompt.

        :return: None
        """
        self.prompt_selector = prompt_selector
        self.reference_selector = reference_selector

        _, prompt_input, _, reference_input = self._get_pair_input(
            corpus,
            self.prompt_field,
            self.reference_field,
            self.prompt_selector,
            self.reference_selector,
        )
        self.prompt_embedding_model = fit_prompt_embedding_model(
            prompt_input,
            reference_input,
            self.snip_first_dim,
            self.prompt__tfidf_min_df,
            self.prompt__tfidf_max_df,
            self.reference__tfidf_min_df,
            self.reference__tfidf_max_df,
            self.svd__n_components,
            self.random_state,
            self.verbosity,
        )
        (
            self.train_results["prompt_ids"],
            self.train_results["prompt_vects"],
            self.train_results["reference_ids"],
            self.train_results["reference_vects"],
        ) = self._get_embeddings(corpus, prompt_selector, reference_selector)
        self.refit_types(self.default_n_types, self.random_state)

    def transform(
        self,
        corpus,
        use_fit_selectors=True,
        prompt_selector=lambda utt: True,
        reference_selector=lambda utt: True,
    ):
        """
        Computes vector representations and prompt type assignments for utterances in a corpus.

        :param corpus: Corpus
        :param use_fit_selectors: defaults to True, will use the same filters as the fit step to determine which utterances will be considered as prompts and responses in the transform step.
        :param prompt_selector: filter that determines which utterances will be considered as prompts in the
            transform step. defaults to prompt_selector, the same as is used in fit.
        :param reference_selector: filter that determines which utterances will be considered as responses in the
            transform step. defaults to reference_selector, the same as is used in fit.
        :return: the corpus, with per-utterance representations and type assignments.
        """
        if use_fit_selectors:
            prompt_selector = self.prompt_selector
            reference_selector = self.reference_selector
        prompt_ids, prompt_vects, reference_ids, reference_vects = self._get_embeddings(
            corpus, prompt_selector, reference_selector
        )

        corpus.set_vector_matrix(
            self.output_field + "__prompt_repr", matrix=prompt_vects, ids=prompt_ids
        )
        corpus.set_vector_matrix(
            self.output_field + "__reference_repr", matrix=reference_vects, ids=reference_ids
        )

        prompt_df, reference_df = self._get_type_assignments(
            prompt_ids, prompt_vects, reference_ids, reference_vects
        )
        prompt_dists, prompt_assigns = (
            prompt_df[prompt_df.columns[:-1]].values,
            prompt_df["type_id"].values,
        )
        prompt_min_dists = prompt_dists.min(axis=1)
        reference_dists, reference_assigns = (
            reference_df[reference_df.columns[:-1]].values,
            reference_df["type_id"].values,
        )
        reference_min_dists = reference_dists.min(axis=1)

        corpus.set_vector_matrix(
            self.output_field + "__prompt_dists__%s" % self.default_n_types,
            ids=prompt_df.index,
            matrix=prompt_dists,
            columns=["type_%d_dist" % x for x in range(prompt_dists.shape[1])],
        )
        corpus.set_vector_matrix(
            self.output_field + "__reference_dists__%s" % self.default_n_types,
            ids=reference_df.index,
            matrix=reference_dists,
            columns=["type_%d_dist" % x for x in range(prompt_dists.shape[1])],
        )
        for id, assign, dist in zip(prompt_df.index, prompt_assigns, prompt_min_dists):
            corpus.get_utterance(id).add_meta(
                self.output_field + "__prompt_type__%s" % self.default_n_types, assign
            )
            corpus.get_utterance(id).add_meta(
                self.output_field + "__prompt_type_dist__%s" % self.default_n_types, float(dist)
            )
        for id, assign, dist in zip(reference_df.index, reference_assigns, reference_min_dists):
            corpus.get_utterance(id).add_meta(
                self.output_field + "__reference_type__%s" % self.default_n_types, assign
            )
            corpus.get_utterance(id).add_meta(
                self.output_field + "__reference_type_dist__%s" % self.default_n_types, float(dist)
            )
        return corpus

    def transform_utterance(self, utterance):
        """
        Computes vector representations and prompt type assignments for a single utterance.

        :param utterance: the utterance.
        :return: the utterance, annotated with representations and type assignments.
        """

        # if self.prompt_transform_filter(utterance):
        utterance = self._transform_utterance_side(utterance, "prompt")
        # if self.reference_transform_filter(utterance):
        utterance = self._transform_utterance_side(utterance, "reference")
        return utterance

    def _transform_utterance_side(self, utterance, side):
        if side == "prompt":
            input_field = self.prompt_transform_field
        elif side == "reference":
            input_field = self.reference_transform_field
        utt_id = utterance.id
        utt_input = utterance.retrieve_meta(input_field)
        if isinstance(utt_input, list):
            utt_input = "\n".join(utt_input)
        utt_ids, utt_vects = transform_embeddings(
            self.prompt_embedding_model, [utt_id], [utt_input], side=side
        )
        assign_df = assign_prompt_types(
            self.type_models[self.default_n_types], utt_ids, utt_vects, self.max_dist
        )
        vals = assign_df.values[0]
        dists = vals[:-1]
        min_dist = min(dists)
        assign = vals[-1]
        utterance.add_meta(
            self.output_field + "__%s_type__%s" % (side, self.default_n_types), assign
        )
        utterance.add_meta(
            self.output_field + "__%s_type_dist__%s" % (side, self.default_n_types), float(min_dist)
        )
        utterance.add_meta(
            self.output_field + "__%s_dists__%s" % (side, self.default_n_types),
            [float(x) for x in dists],
        )
        utterance.add_meta(self.output_field + "__%s_repr" % side, [float(x) for x in utt_vects[0]])
        return utterance

    def refit_types(self, n_types, random_state=None, name=None):
        """
        Using the latent representations of prompt terms learned during the initial `fit` call, infers `n_types` prompt types. permits retraining the clustering model that determines the number of types, on top of the initial model. calling this *and* updating the `default_n_types` field of the model will result in future `transform` calls assigning utterances to one of `n_types` prompt types.

        :param n_types: number of types to learn
        :param random_state: random seed
        :param name: the name of the new type model. defaults to n_types.
        :return: None
        """

        if name is None:
            key = n_types
        else:
            key = name
        if random_state is None:
            random_state = self.random_state
        self.type_models[key] = fit_prompt_type_model(
            self.prompt_embedding_model, n_types, random_state, self.max_dist, self.verbosity
        )
        prompt_df, reference_df = self._get_type_assignments(type_key=key)
        self.train_types[key] = {"prompt_df": prompt_df, "reference_df": reference_df}

    def _get_embeddings(self, corpus, prompt_selector, reference_selector):
        prompt_ids, prompt_inputs = self._get_input(
            corpus, self.prompt_transform_field, prompt_selector
        )
        reference_ids, reference_inputs = self._get_input(
            corpus, self.reference_transform_field, reference_selector
        )
        prompt_ids, prompt_vects = transform_embeddings(
            self.prompt_embedding_model, prompt_ids, prompt_inputs, side="prompt"
        )
        reference_ids, reference_vects = transform_embeddings(
            self.prompt_embedding_model, reference_ids, reference_inputs, side="reference"
        )
        return prompt_ids, prompt_vects, reference_ids, reference_vects

    def _get_type_assignments(
        self,
        prompt_ids=None,
        prompt_vects=None,
        reference_ids=None,
        reference_vects=None,
        type_key=None,
    ):
        if prompt_ids is None:
            prompt_ids, prompt_vects, reference_ids, reference_vects = [
                self.train_results[k]
                for k in ["prompt_ids", "prompt_vects", "reference_ids", "reference_vects"]
            ]
        if type_key is None:
            type_key = self.default_n_types
        prompt_df = assign_prompt_types(
            self.type_models[type_key], prompt_ids, prompt_vects, self.max_dist
        )
        reference_df = assign_prompt_types(
            self.type_models[type_key], reference_ids, reference_vects, self.max_dist
        )
        return prompt_df, reference_df

    def display_type(self, type_id, corpus=None, type_key=None, k=10):
        """
        For a particular prompt type, displays the representative prompt and response terms. can also display representative prompt and response utterances.

        :param type_id: ID of the prompt type to display.
        :param corpus: pass in the training corpus to also display representative utterances.
        :param type_key: the name of the prompt type clustering model to use. defaults to `n_types` that the model was initialized with, but if `refit_types` is called with different number of types, can be modified to display this updated model as well.
        :param k: the number of sample terms (or utteranceS) to display.
        :return: None
        """

        if type_key is None:
            type_key = self.default_n_types
        prompt_df = self.type_models[type_key]["prompt_df"]
        reference_df = self.type_models[type_key]["reference_df"]

        top_prompt = prompt_df[prompt_df.type_id == type_id].sort_values(type_id).head(k)
        top_ref = reference_df[reference_df.type_id == type_id].sort_values(type_id).head(k)
        print("top prompt:")
        print(top_prompt)
        print("top response:")
        print(top_ref)

        if corpus is not None:
            prompt_df = self.train_types[type_key]["prompt_df"]
            reference_df = self.train_types[type_key]["reference_df"]
            top_prompt = prompt_df[prompt_df.type_id == type_id].sort_values(type_id).head(k).index
            top_ref = (
                reference_df[reference_df.type_id == type_id].sort_values(type_id).head(k).index
            )
            print("top prompts:")
            for utt in top_prompt:
                print(utt, corpus.get_utterance(utt).text)
                print(corpus.get_utterance(utt).retrieve_meta(self.prompt_transform_field))
                print()
            print("top responses:")
            for utt in top_ref:
                print(utt, corpus.get_utterance(utt).text)
                print(corpus.get_utterance(utt).retrieve_meta(self.reference_transform_field))
                print()

    def summarize(self, corpus, type_ids=None, type_key=None, k=10):
        """
        Displays representative prompt and response terms and utterances for each type learned. A wrapper for `display_type`.

        :param corpus: corpus to display utterances for (must have `transform()` called on it)
        :param type_ids: ID of the prompt type to display. if None, will display all types.
        :param type_key: the name of the prompt type clustering model to use. defaults to `n_types` that the model was initialized with, but if `refit_types` is called with different number of types, can be modified to display this updated model as well.
        :param k: the number of sample terms (or utteranceS) to display.
        :return: None
        """
        if type_key is None:
            type_key = self.default_n_types

        n_types = self.type_models[type_key]["km_model"].n_clusters
        if type_ids is None:
            type_ids = list(range(n_types))
        if not isinstance(type_ids, list):
            type_ids = [type_ids]
        for type_id in type_ids:
            print("TYPE", type_id)
            self.display_type(type_id, corpus, type_key, k)
            print("====")

    def dump_model(self, model_dir, type_keys="default", dump_train_corpus=True):
        """
        Dumps the model to disk.

        :param model_dir: directory to write model to
        :param type_keys: if 'default', will only write the type clustering model corresponding to the `n_types` the model was initialized with. if 'all', will write all clustering models that have been trained via calls to `refit_types`. can also take a list of clustering models.
        :param dump_train_corpus: whether to also write the representations and type assignments of the training corpus. defaults to True.
        :return: None
        """

        if self.verbosity > 0:
            print("dumping embedding model")
        if not os.path.exists(model_dir):
            try:
                os.mkdir(model_dir)
            except:
                pass
        for k in ["prompt_tfidf_model", "reference_tfidf_model", "svd_model"]:
            joblib.dump(self.prompt_embedding_model[k], os.path.join(model_dir, k + ".joblib"))

        for k in ["U_prompt", "U_reference"]:
            np.save(os.path.join(model_dir, k), self.prompt_embedding_model[k])

        if dump_train_corpus:
            if self.verbosity > 0:
                print("dumping training embeddings")
            for k in ["prompt_ids", "prompt_vects", "reference_ids", "reference_vects"]:
                np.save(os.path.join(model_dir, "train_" + k), self.train_results[k])

        if type_keys == "default":
            to_dump = [self.default_n_types]
        elif type_keys == "all":
            to_dump = self.type_models.keys()
        else:
            to_dump = type_keys
        for key in to_dump:
            if self.verbosity > 0:
                print("dumping type model", key)
            type_model = self.type_models[key]
            joblib.dump(type_model["km_model"], os.path.join(model_dir, "km_model.%s.joblib" % key))
            for k in ["prompt_df", "reference_df"]:
                type_model[k].to_csv(os.path.join(model_dir, "%s.%s.tsv" % (k, key)), sep="\t")
            if dump_train_corpus:
                train_types = self.train_types[key]
                for k in ["prompt_df", "reference_df"]:
                    train_types[k].to_csv(
                        os.path.join(model_dir, "train_%s.%s.tsv" % (k, key)), sep="\t"
                    )

    def get_model(self, type_keys="default"):
        """
        Returns the model as a dictionary containing:
            * embedding_model: stores information pertaining to the vector representations.
                * prompt_tfidf_model: sklearn tf-idf model that converts prompt input to term-document matrix
                * reference_tfidf_model: tf-idf model that converts response input to term-document matrix
                * svd_model: sklearn TruncatedSVD model that produces a low-dimensional representation of responses and prompts
                * U_prompt: vector representations of prompt terms
                * U_reference: vector representations of response terms
            * type_models: a dictionary mapping each type clustering model to:
                * km_model: a sklearn KMeans model of the learned types
                * prompt_df: distances to cluster centroids, and type assignments, of prompt terms
                * reference_df: distances to cluster centroids, and type assignments, of reference terms
        :param type_keys: if 'default', will return the type clustering model corresponding to the `n_types` the model was initialized with. if 'all', returns all clustering models that have been trained via calls to `refit_types`. can also take a list of clustering models.
        :return: the prompt types model
        """
        if type_keys == "default":
            to_get = [self.default_n_types]
        elif type_keys == "all":
            to_get = self.type_models.keys()
        else:
            to_get = type_keys
        to_return = {
            "embedding_model": self.prompt_embedding_model,
            "type_models": {k: self.type_models[k] for k in to_get},
        }
        return to_return

    def load_model(self, model_dir, type_keys="default", load_train_corpus=True):
        """
        Loads the model from disk.

        :param model_dir: directory to read model to
        :param type_keys: if 'default', will only read the type clustering model corresponding to the `n_types` the model was initialized with. if 'all', will read all clustering models that are available in directory. can also take a list of clustering models.
        :param load_train_corpus: whether to also read the representations and type assignments of the training corpus. defaults to True.
        :return: None
        """
        if self.verbosity > 0:
            print("loading embedding model")
        for k in ["prompt_tfidf_model", "reference_tfidf_model", "svd_model"]:
            self.prompt_embedding_model[k] = joblib.load(os.path.join(model_dir, k + ".joblib"))
        for k in ["U_prompt", "U_reference"]:
            self.prompt_embedding_model[k] = np.load(os.path.join(model_dir, k + ".npy"))

        if load_train_corpus:
            if self.verbosity > 0:
                print("loading training embeddings")
            for k in ["prompt_ids", "prompt_vects", "reference_ids", "reference_vects"]:
                self.train_results[k] = np.load(os.path.join(model_dir, "train_" + k + ".npy"))

        if type_keys == "default":
            to_load = [self.default_n_types]
        elif type_keys == "all":
            to_load = [
                x.replace("km_model.", "").replace(".joblib", "")
                for x in os.listdir(model_dir)
                if x.startswith("km_model")
            ]
        else:
            to_load = type_keys
        for key in to_load:
            try:
                key = int(key)
            except:
                pass
            if self.verbosity > 0:
                print("loading type model", key)
            self.type_models[key] = {}  # this should be an int-ish
            self.type_models[key]["km_model"] = joblib.load(
                os.path.join(model_dir, "km_model.%s.joblib" % key)
            )

            for k in ["prompt_df", "reference_df"]:
                self.type_models[key][k] = pd.read_csv(
                    os.path.join(model_dir, "%s.%s.tsv" % (k, key)), sep="\t", index_col=0
                )
                self.type_models[key][k].columns = [
                    int(x) for x in self.type_models[key][k].columns[:-1]
                ] + ["type_id"]
            if load_train_corpus:
                self.train_types[key] = {}
                for k in ["prompt_df", "reference_df"]:
                    self.train_types[key][k] = pd.read_csv(
                        os.path.join(model_dir, "train_%s.%s.tsv" % (k, key)), sep="\t", index_col=0
                    )
                    self.train_types[key][k].columns = [
                        int(x) for x in self.train_types[key][k].columns[:-1]
                    ] + ["type_id"]

    def _get_input(self, corpus, field, filter_fn, check_nonempty=True):
        ids = []
        inputs = []
        for utterance in corpus.iter_utterances():
            input = utterance.retrieve_meta(field)
            if isinstance(input, list):
                input = "\n".join(input)
            if filter_fn(utterance) and ((not check_nonempty) or (len(input) > 0)):
                ids.append(utterance.id)
                inputs.append(input)
        return ids, inputs

    def _get_pair_input(
        self,
        corpus,
        prompt_field,
        reference_field,
        prompt_selector,
        reference_selector,
        check_nonempty=True,
    ):
        prompt_ids = []
        prompt_utts = []
        reference_ids = []
        reference_utts = []
        for reference_utt in corpus.iter_utterances():
            if reference_utt.reply_to is None:
                continue
            prompt_utt_id = reference_utt.reply_to
            try:
                prompt_utt = corpus.get_utterance(prompt_utt_id)
            except:
                continue
            if prompt_selector(prompt_utt) and reference_selector(reference_utt):
                prompt_input = prompt_utt.retrieve_meta(prompt_field)
                reference_input = reference_utt.retrieve_meta(reference_field)

                if (prompt_input is None) or (reference_input is None):
                    continue

                if isinstance(prompt_input, list):
                    prompt_input = "\n".join(prompt_input)
                if isinstance(reference_input, list):
                    reference_input = "\n".join(reference_input)

                if (not check_nonempty) or ((len(prompt_input) > 0) and (len(reference_input) > 0)):
                    prompt_ids.append(prompt_utt.id)
                    prompt_utts.append(prompt_input)
                    reference_ids.append(reference_utt.id)
                    reference_utts.append(reference_input)
        return prompt_ids, prompt_utts, reference_ids, reference_utts


def fit_prompt_embedding_model(
    prompt_input,
    reference_input,
    snip_first_dim=True,
    prompt__tfidf_min_df=100,
    prompt__tfidf_max_df=0.1,
    reference__tfidf_min_df=100,
    reference__tfidf_max_df=0.1,
    svd__n_components=25,
    random_state=None,
    verbosity=0,
):
    """
    Standalone function that fits an embedding model given paired prompt and response inputs. See docstring of the `PromptTypes` class for details.

    :param prompt_input: list of prompts (represented as space-separated strings of terms)
    :param reference_input: list of responses (represented as space-separated strings of terms). note that each entry of reference_input should be a response to the corresponding entry in prompt_input.
    :return: prompt embedding model
    """

    if verbosity > 0:
        print("fitting %d input pairs" % len(prompt_input))
        print("fitting reference tfidf model")
    reference_tfidf_model = TfidfVectorizer(
        min_df=reference__tfidf_min_df,
        max_df=reference__tfidf_max_df,
        binary=True,
        token_pattern=r"(?u)(\S+)",
    )
    reference_vect = reference_tfidf_model.fit_transform(reference_input)

    if verbosity > 0:
        print("fitting prompt tfidf model")
    prompt_tfidf_model = TfidfVectorizer(
        min_df=prompt__tfidf_min_df,
        max_df=prompt__tfidf_max_df,
        binary=True,
        token_pattern=r"(?u)(\S+)",
    )
    prompt_vect = prompt_tfidf_model.fit_transform(prompt_input)

    if verbosity > 0:
        print("fitting svd model")
    svd_model = TruncatedSVD(
        n_components=svd__n_components, random_state=random_state, algorithm="arpack"
    )

    U_reference = svd_model.fit_transform(normalize(reference_vect.T))
    s = svd_model.singular_values_
    U_reference /= s
    U_prompt = (svd_model.components_ * normalize(prompt_vect, axis=0) / s[:, np.newaxis]).T

    if snip_first_dim:
        U_prompt = U_prompt[:, 1:]
        U_reference = U_reference[:, 1:]
    U_prompt_norm = normalize(U_prompt)
    U_reference_norm = normalize(U_reference)

    return {
        "prompt_tfidf_model": prompt_tfidf_model,
        "reference_tfidf_model": reference_tfidf_model,
        "svd_model": svd_model,
        "U_prompt": U_prompt_norm,
        "U_reference": U_reference_norm,
    }


def transform_embeddings(model, ids, input, side="prompt", filter_empty=True):
    """
    Standalone function that returns vector representations of input text given a trained PromptTypes prompt_embedding_model. See docstring of `PromptTypes` class for details.

    :param model: prompt embedding model
    :param ids: ids of input text
    :param input: a list where each entry has corresponding id in the ids argument, and is a string of terms corresponding to an utterance.
    :param side: whether to return prompt or response embeddings ("prompt" and "reference" respectively); defaults to "prompt"
    :param filter_empty: if `True`, will not return embeddings for prompts with no terms.
    :return: input IDs `ids`, and corresponding vector representations of input `vect`
    """

    tfidf_vects = normalize(model["%s_tfidf_model" % side].transform(input), norm="l1")
    mask = np.array(tfidf_vects.sum(axis=1)).flatten() > 0
    vects = normalize(tfidf_vects * model["U_%s" % side])
    if filter_empty:
        ids = np.array(ids)[mask]
        vects = vects[mask]
    return ids, vects


def fit_prompt_type_model(model, n_types, random_state=None, max_dist=0.9, verbosity=0):
    """
    Standalone function that fits a prompt type model given paired prompt and response inputs. See docstring of the `PromptTypes` class for details.

    :param model: prompt embedding model (from `fit_prompt_embedding_model()`)
    :param n_types: number of prompt types to infer
    :return: prompt type model
    """

    if verbosity > 0:
        print("fitting %d prompt types" % n_types)
    km = KMeans(n_clusters=n_types, random_state=random_state)
    km.fit(model["U_prompt"])
    prompt_dists = km.transform(model["U_prompt"])
    prompt_clusters = km.predict(model["U_prompt"])
    prompt_clusters[prompt_dists.min(axis=1) >= max_dist] = -1
    reference_dists = km.transform(model["U_reference"])
    reference_clusters = km.predict(model["U_reference"])
    reference_clusters[reference_dists.min(axis=1) >= max_dist] = -1

    prompt_df = pd.DataFrame(
        index=model["prompt_tfidf_model"].get_feature_names_out(),
        data=np.hstack([prompt_dists, prompt_clusters[:, np.newaxis]]),
        columns=list(range(n_types)) + ["type_id"],
    )
    reference_df = pd.DataFrame(
        index=model["reference_tfidf_model"].get_feature_names_out(),
        data=np.hstack([reference_dists, reference_clusters[:, np.newaxis]]),
        columns=list(range(n_types)) + ["type_id"],
    )
    return {"km_model": km, "prompt_df": prompt_df, "reference_df": reference_df}


def assign_prompt_types(model, ids, vects, max_dist=0.9):
    """
    Standalone function that returns type assignments of input vectors given a trained PromptTypes type model. See docstring of `PromptTypes` class for details.

    :param model: prompt type model
    :param ids: ids of input vectors
    :param vects: input vectors
    :return: a dataframe storing cluster centroid distances and the assigned type.
    """

    dists = model["km_model"].transform(vects)
    clusters = model["km_model"].predict(vects)
    dist_mask = dists.min(axis=1) >= max_dist
    clusters[dist_mask] = -1
    df = pd.DataFrame(
        index=ids,
        data=np.hstack([dists, clusters[:, np.newaxis]]),
        columns=list(range(dists.shape[1])) + ["type_id"],
    )
    return df
