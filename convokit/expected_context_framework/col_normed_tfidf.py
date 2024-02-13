from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Normalizer, normalize
from scipy import sparse
import numpy as np
import joblib
import os
import json

from convokit.transformer import Transformer


class ColNormedTfidfTransformer(Transformer):
    """
    Transformer that derives tf-idf reweighted representations of utterances,
    which are normalized by column, i.e., per term. This may be helpful in deriving downstream representations that are less sensitive to relative term frequency; for instance, it could be used to derive input representations to `ExpectedContextModelWrapper`.

    :param input_field: the name of the attribute of utterances to use as input to fit. note that unless `token_pattern` is specified as an additional argument, this attribute must be a string consisting of whitespace-separated features.
    :param output_field: the name of the attribute to write to in the transform step.
    :param model: optional, an exisitng `ColNormedTfidfTransformer`
    :param kwargs: other keyword arguments used to initialize the underlying `TfidfVectorizer` from scikit-learn, see that documentation for details.
    """

    def __init__(self, input_field, output_field="col_normed_tfidf", model=None, **kwargs):
        if model is not None:
            self.tfidf_obj = model.tfidf_obj
        else:
            self.tfidf_obj = ColNormedTfidf(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        if self.input_field == "text":
            self.text_func = lambda x: x.text
        else:
            self.text_func = lambda x: x.meta[self.input_field]

    def fit(self, corpus, y=None, selector=lambda x: True):
        """
        Fits a transformer over training data.

        :param corpus: Corpus
        :param selector: which utterances to fit the transformer over. a boolean function of the form filter(utterance) that defaults to True (i.e., all utterances).
        :return: None
        """
        docs = [self.text_func(ut) for ut in corpus.iter_utterances(selector=selector)]
        self.tfidf_obj.fit(docs)
        return self

    def transform(self, corpus, selector=lambda x: True):
        """
        Computes column-normalized tf-idf representations for utterances in a corpus, stored in the corpus as `<output_field>`. Also annotates each utterance with a metadata field, `<output_field>__n_feats`, indicating the number of terms in the vocabulary that utterance contains.


        :param corpus: Corpus
        :param selector: which utterances to transform

        :return: corpus, with per-utterance representations and vocabulary counts
        """
        ids = []
        docs = []
        for ut in corpus.iter_utterances(selector=selector):
            ids.append(ut.id)
            docs.append(self.text_func(ut))
            ut.add_vector(self.output_field)
        vects = self.tfidf_obj.transform(docs)
        column_names = self.tfidf_obj.get_feature_names_out()
        corpus.set_vector_matrix(self.output_field, matrix=vects, ids=ids, columns=column_names)
        n_feats = np.array((vects > 0).sum(axis=1)).flatten()
        for id, n in zip(ids, n_feats):
            corpus.get_utterance(id).meta[self.output_field + "__n_feats"] = int(n)
        return corpus

    def transform_utterance(self, utt):
        """
        Computes tf-idf representations for a single utterance. Representation is stored in the utterance as `<output_field>__vect`;
        number of vocabulary terms that utterance contains is stored as `<output_field>__n_feats`

        :param utt: Utterance

        :return: utterance, with representation and vocabulary count
        """
        docs = [self.text_func(utt)]
        vect_ = np.array(self.tfidf_obj.transform(docs))
        n_feats = np.array((vect_ > 0).sum(axis=1)).flatten()
        utt.meta[self.output_field] = [float(x) for x in vect_[0]]
        utt.meta[self.output_field + "__n_feats"] = int(n_feats[0])
        return utt

    def fit_transform(self, corpus, y=None, selector=lambda x: True):
        self.fit(corpus, y, selector)
        return self.transform(corpus, selector)

    def get_vocabulary(self):
        """
        :return: array of feature names
        """
        return self.tfidf_obj.get_feature_names_out()

    def load(self, dirname):
        """
        Loads model from disk.

        :param dirname: directory to load from
        :return: None
        """
        self.tfidf_obj.load(dirname)

    def dump(self, dirname):
        """
        Dumps model to disk.

        :param dirname: directory to write to
        :return: None
        """
        self.tfidf_obj.dump(dirname)


class ColNormedTfidf(TransformerMixin):
    """
    Model that derives tf-idf reweighted representations of utterances,
    which are normalized by column. Can be used in ConvoKit through the `ColNormedTfidfTransformer` transformer; see documentation of that transformer for further details.
    """

    def __init__(self, **kwargs):
        if "token_pattern" in kwargs:
            self.tfidf_model = TfidfVectorizer(**kwargs)
        else:
            self.tfidf_model = TfidfVectorizer(token_pattern=r"(?u)(\S+)", **kwargs)

    def fit(self, X, y=None):
        tfidf_vects_raw = self.tfidf_model.fit_transform(X)
        self.col_norms = sparse.linalg.norm(tfidf_vects_raw, axis=0)

    def transform(self, X):
        tfidf_vects_raw = self.tfidf_model.transform(X)
        tfidf_vect = tfidf_vects_raw / self.col_norms
        return tfidf_vect

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.tfidf_model.get_feature_names_out()

    def get_params(self, deep=True):
        return self.tfidf_model.get_params(deep=deep)

    def set_params(self, **params):
        return self.tfidf_model.set_params(**params)

    def load(self, dirname):
        self.tfidf_model = joblib.load(os.path.join(dirname, "tfidf_model.joblib"))
        self.col_norms = np.load(os.path.join(dirname, "tfidf_col_norms.npy"))

    def dump(self, dirname):
        try:
            os.mkdir(dirname)
        except:
            pass
        np.save(os.path.join(dirname, "tfidf_col_norms.npy"), self.col_norms)
        joblib.dump(self.tfidf_model, os.path.join(dirname, "tfidf_model.joblib"))
