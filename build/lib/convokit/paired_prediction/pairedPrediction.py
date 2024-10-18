from typing import List, Callable

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from convokit import Transformer, CorpusComponent, Corpus
from convokit.classifier.util import get_coefs_helper
from .util import *


class PairedPrediction(Transformer):
    """
    At a high level, Paired Prediction is a quasi-experimental method that controls for certain priors,
    see Cheng et al. 2014 for an illustrated example of PairedPrediction in research.
    (https://cs.stanford.edu/people/jure/pubs/disqus-icwsm14.pdf)

    See :doc:`Pairer's documentation <pairer>` for more information about pairing.

    :param pred_feats: list of metadata attributes (i.e. predictive features) to be used in prediction.
        Features can either be values or a dictionary of key-value pairs.
    :param clf: optional classifier to be used in the paired prediction
    :param pair_id_attribute_name: metadata attribute name to use in annotating object with pair id, default: "pair_id"
    :param label_attribute_name: metadata attribute name to use in annotating object with predicted label, default: "label"
    :param pair_orientation_attribute_name: metadata attribute name to use in annotating object with pair orientation,
        default: "pair_orientation"

    """

    def __init__(
        self,
        obj_type: str,
        pred_feats: List[str],
        clf=None,
        pair_id_attribute_name: str = "pair_id",
        label_attribute_name: str = "pair_obj_label",
        pair_orientation_attribute_name: str = "pair_orientation",
    ):
        assert obj_type in ["speaker", "utterance", "conversation"]
        self.obj_type = obj_type
        self.clf = (
            Pipeline(
                [
                    ("standardScaler", StandardScaler(with_mean=False)),
                    ("logreg", LogisticRegression(solver="liblinear")),
                ]
            )
            if clf is None
            else clf
        )
        self.pred_feats = pred_feats
        self.pair_id_attribute_name = pair_id_attribute_name
        self.label_attribute_name = label_attribute_name
        self.pair_orientation_attribute_name = pair_orientation_attribute_name

    def fit(
        self, corpus: Corpus, y=None, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ):
        """
        Fit the internal classifier on the paired object features, with an optional selector selecting for which corpus objects to include in the analysis

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns a bool: True if the object is to be included in the paired prediction. By default, includes all objects.
        :return: fitted PairedPrediction Transformer
        """
        # Check if Pairer.transform() needs to be run first
        self._check_for_pair_information(corpus)
        pair_id_to_objs = generate_pair_id_to_objs(
            corpus,
            self.obj_type,
            selector,
            self.pair_orientation_attribute_name,
            self.label_attribute_name,
            self.pair_id_attribute_name,
        )

        X, y = generate_paired_X_y(
            self.pred_feats, self.pair_orientation_attribute_name, pair_id_to_objs
        )
        self.clf.fit(X, y)
        return self

    def transform(self, corpus: Corpus) -> Corpus:
        """
        PairedPrediction does not add any annotations to the Corpus.
        """
        return corpus

    def _check_for_pair_information(self, corpus):
        # Check if transform() needs to be run first
        sample_obj = next(corpus.iter_objs(self.obj_type))
        meta_keys = set(sample_obj.meta)
        required_keys = {
            self.pair_orientation_attribute_name,
            self.pair_id_attribute_name,
            self.label_attribute_name,
        }
        required_keys -= meta_keys
        if len(required_keys) > 0:
            raise ValueError(
                "Some metadata attributes required for paired prediction are missing: {}. "
                "You may need to run Pairer.transform() first.".format(required_keys)
            )

    def summarize(
        self,
        corpus: Corpus,
        selector: Callable[[CorpusComponent], bool] = lambda x: True,
        cv=KFold(n_splits=5, shuffle=True),
    ):
        """
        Run PairedPrediction on the corpus with cross-validation and returns the mean cross-validation score.

        :param corpus: target Corpus (must be annotated with pair information using PairedPrediction.transform())
        :param selector: a (lambda) function that takes a Corpus object and returns a bool: True if the object is to be included in summary. By default, includes all objects.
        :param cv: optional CV model: default is KFold(n_splits=5, shuffle=True)
        :return: cross-validation accuracy score
        """
        pair_id_to_objs = generate_pair_id_to_objs(
            corpus,
            self.obj_type,
            selector,
            self.pair_orientation_attribute_name,
            self.label_attribute_name,
            self.pair_id_attribute_name,
        )

        X, y = generate_paired_X_y(
            self.pred_feats, self.pair_orientation_attribute_name, pair_id_to_objs
        )
        return np.mean(cross_val_score(self.clf, X, y, cv=cv, error_score="raise"))

    def get_coefs(self, feature_names: List[str], coef_func=None):
        """
        Get dataframe of classifier coefficients.

        :param feature_names: list of feature names to get coefficients for
        :param coef_func: function for accessing the list of coefficients from the classifier model; by default, assumes it is a pipeline with a logistic regression component
        :return: DataFrame of features and coefficients, indexed by feature names
        """
        return get_coefs_helper(self.clf, feature_names, coef_func)
