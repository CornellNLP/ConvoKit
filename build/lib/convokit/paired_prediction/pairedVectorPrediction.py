from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from typing import List, Callable

from convokit import CorpusComponent, Corpus
from .util import *
from .pairedPrediction import PairedPrediction
from convokit.classifier.util import get_coefs_helper


class PairedVectorPrediction(PairedPrediction):
    """
    Transformer for doing a Paired Prediction with vectors.

    :param obj_type: corpus component type being used for analysis: 'utterance', 'speaker', or 'conversation'
    :param vector_name: name of the vector matrix containing the bag-of-words vectors
    :param clf: classifier to be used in the paired prediction; by default: standard-scaled logistic regression
    :param pair_id_attribute_name: metadata attribute name to use in annotating object with pair id, default: "pair_id"
    :param label_attribute_name: metadata attribute name to use in annotating object with predicted label,
        default: "label"
    :param pair_orientation_attribute_name: metadata attribute name to use in annotating object with pair orientation,
        default: "pair_orientation"

    """

    def __init__(
        self,
        obj_type: str,
        vector_name: str,
        clf=None,
        pair_id_attribute_name: str = "pair_id",
        label_attribute_name: str = "pair_obj_label",
        pair_orientation_attribute_name: str = "pair_orientation",
    ):
        assert obj_type in ["speaker", "utterance", "conversation"]
        self.obj_type = obj_type
        self.vector_name = vector_name

        clf = (
            Pipeline(
                [
                    ("standardScaler", StandardScaler(with_mean=False)),
                    ("logreg", LogisticRegression(solver="liblinear")),
                ]
            )
            if clf is None
            else clf
        )

        super().__init__(
            obj_type=obj_type,
            pred_feats=[],
            pair_id_attribute_name=pair_id_attribute_name,
            label_attribute_name=label_attribute_name,
            pair_orientation_attribute_name=pair_orientation_attribute_name,
            clf=clf,
        )

    def fit(
        self, corpus: Corpus, y=None, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ):
        """
        Fit the internal classifier to the Corpus component objects.

        :param corpus: the target Corpus
        :param selector: selector (lambda) function for which objects should be included in the analysis
        :return: this Transformer object with a fitted internal classifier
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
        X, y = generate_vectors_paired_X_y(
            corpus, self.vector_name, self.pair_orientation_attribute_name, pair_id_to_objs
        )
        self.clf.fit(X, y)
        return self

    def summarize(
        self,
        corpus: Corpus,
        selector: Callable[[CorpusComponent], bool] = lambda x: True,
        cv=KFold(n_splits=5, shuffle=True),
    ):
        """
        Run PairedPrediction on the corpus with cross-validation.

        :param corpus: annoted Corpus (with pair information from PairedPrediction.transform())
        :param selector: selector (lambda) function for which objects should be included in the analysis
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

        X, y = generate_vectors_paired_X_y(
            corpus, self.vector_name, self.pair_orientation_attribute_name, pair_id_to_objs
        )

        return np.mean(cross_val_score(self.clf, X, y, cv=cv, error_score="raise"))

    def get_coefs(self, feature_names: List[str], coef_func=None):
        """
        Get dataframe of classifier coefficients.
        By default, assumes it is a pipeline with a logistic regression component. For other setups, the user should
        define a custom `coef_func`.

        :param feature_names: list of feature names to get coefficients for
        :param coef_func: (optional) function for accessing the list of coefficients from the classifier model
        :return: DataFrame of features and coefficients, indexed by feature names
        """
        return get_coefs_helper(self.clf, feature_names, coef_func)
