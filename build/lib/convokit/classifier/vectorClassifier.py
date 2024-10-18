from convokit import Corpus, CorpusComponent
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from typing import Callable, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
from .classifier import Classifier
import numpy as np
from .util import extract_vector_feats_and_label


class VectorClassifier(Classifier):
    """
    Transformer that trains a classifier on the Corpus components' text vector representation
    (e.g. bag-of-words, TF-IDF, etc)

    Corpus must have a vector with the specified `vector_name`.

    Inherits from `Classifier` and has access to its methods.

    :param obj_type: "speaker", "utterance", or "conversation"
    :param vector_name: the metadata key where the Corpus object text vector is stored
    :param columns: list of column names of vector matrix to use; uses all columns by default.
    :param labeller: a (lambda) function that takes a Corpus object and returns True (y=1) or False (y=0) - i.e.
        labeller defines the y value of the object for fitting
    :param clf: a sklearn Classifier. By default, clf is a Pipeline with StandardScaler and LogisticRegression
    :param clf_attribute_name: the metadata attribute name to store the classifier prediction value under; default:
        "prediction"
    :param clf_prob_attribute_name: the metadata attribute name to store the classifier prediction score under;
        default: "pred_score"
    """

    def __init__(
        self,
        obj_type: str,
        vector_name: str,
        columns: List[str] = None,
        labeller: Callable[[CorpusComponent], bool] = lambda x: True,
        clf=None,
        clf_attribute_name: str = "prediction",
        clf_prob_attribute_name: str = "pred_score",
    ):
        if clf is None:
            clf = Pipeline(
                [
                    ("standardScaler", StandardScaler(with_mean=False)),
                    ("logreg", LogisticRegression(solver="liblinear")),
                ]
            )
            print("Initialized default classification model (standard scaled logistic regression).")

        super().__init__(
            obj_type=obj_type,
            pred_feats=[],
            labeller=labeller,
            clf=clf,
            clf_attribute_name=clf_attribute_name,
            clf_prob_attribute_name=clf_prob_attribute_name,
        )
        self.vector_name = vector_name
        self.columns = columns

    def fit(
        self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True, y=None
    ):
        """
        Fit the Transformer's internal classifier model on the vector matrix that represents one of
        the Corpus components, with an optional selector that selects for objects to be fit on.

        :param corpus: the target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False
            (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :return: the fitted VectorClassifier
        """
        # collect texts for vectorization
        obj_ids = []
        y = []
        for obj in corpus.iter_objs(self.obj_type, selector):
            obj_ids.append(obj.id)
            y.append(self.labeller(obj))
        X = corpus.get_vectors(self.vector_name, ids=obj_ids, columns=self.columns)
        y = np.array(y)
        # print(corpus.get_vector_matrix(self.vector_name).matrix.shape)
        # print(X.shape)
        # print(y.shape)
        self.clf.fit(X, y)
        return self

    def transform(
        self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ) -> Corpus:
        """
        Annotate the corpus components with the classifier prediction and prediction score, with an optional selector
        that selects for objects to be classified. Objects that are not selected will get a metadata value of 'None'
        instead of the classifier prediction.

        :param corpus: the target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False
            (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.

        :return: the target Corpus annotated
        """
        objs = []
        for obj in corpus.iter_objs(self.obj_type):
            if selector(obj):
                objs.append(obj)
            else:
                obj.add_meta(self.clf_attribute_name, None)
                obj.add_meta(self.clf_prob_attribute_name, None)

        obj_ids = [obj.id for obj in objs]
        X = corpus.get_vector_matrix(self.vector_name).get_vectors(obj_ids, self.columns)

        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_attribute_name, clf)
            obj.add_meta(self.clf_prob_attribute_name, clf_prob)
        return corpus

    def transform_objs(self, objs: List[CorpusComponent]) -> List[CorpusComponent]:
        """
        Not implemented for VectorClassifier.
        """
        # """
        # Run classifier on list of Corpus component objects and annotate them with their predictions and
        # prediction scores.
        #
        # :param objs: list of Corpus objects
        # :return: list of annotated Corpus objects
        # """
        raise NotImplementedError(
            "VectorClassifier can only be run on corpora, not arbitrary lists of corpus "
            "component objects."
        )

    def fit_transform(
        self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ) -> Corpus:
        """
        Runs the `fit()` and `transform()` steps in order, with the specified selector.

        :param corpus: the target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False
            (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.

        :return: the target Corpus annotated
        """
        self.fit(corpus, selector=selector)
        return self.transform(corpus, selector=selector)

    def summarize(
        self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ):
        """
        Generate a DataFrame indexed by object id with the classifier predictions and scores.

        :param corpus: the annotated Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False
            (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :return: a pandas DataFrame
        """
        objId_clf_prob = []

        for obj in corpus.iter_objs(self.obj_type, selector):
            objId_clf_prob.append(
                (obj.id, obj.meta[self.clf_attribute_name], obj.meta[self.clf_prob_attribute_name])
            )

        return (
            pd.DataFrame(
                list(objId_clf_prob),
                columns=["id", self.clf_attribute_name, self.clf_prob_attribute_name],
            )
            .set_index("id")
            .sort_values(self.clf_prob_attribute_name, ascending=False)
        )

    def summarize_objs(self, objs: List[CorpusComponent]):
        """
        Not implemented for VectorClassifier.
        """
        # Generate a pandas DataFrame (indexed by object id, with prediction and prediction score columns) of
        # classification results.
        #
        # Runs on a list of Corpus objects.
        #
        # :param objs: list of Corpus objects
        # :return: pandas DataFrame indexed by Corpus object id
        # objId_clf_prob = []
        # for obj in objs:
        #     objId_clf_prob.append((obj.id, obj.meta[self.clf_attribute_name], obj.meta[self.clf_prob_attribute_name]))
        #
        # return pd.DataFrame(list(objId_clf_prob),
        #                     columns=['id', self.clf_attribute_name, self.clf_prob_attribute_name]).set_index('id').sort_values(
        #                     self.clf_prob_attribute_name)
        raise NotImplementedError(
            "VectorClassifier can only be run on corpora, not arbitrary lists of corpus "
            "component objects."
        )

    def evaluate_with_train_test_split(
        self,
        corpus: Corpus,
        selector: Callable[[CorpusComponent], bool] = lambda x: True,
        test_size: float = 0.2,
    ):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label,
        using a train-test split.

        Run either on a Corpus (with Classifier labeller, selector, obj_type settings) or a list of Corpus objects

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include /
            exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :param test_size: size of test set
        :return: accuracy and confusion matrix
        """
        X, y = extract_vector_feats_and_label(
            corpus, self.obj_type, self.vector_name, self.columns, self.labeller, selector
        )

        print("Running a train-test-split evaluation...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        accuracy = np.mean(preds == y_test)
        print("Done.")
        return accuracy, confusion_matrix(y_true=y_test, y_pred=preds)

    def evaluate_with_cv(
        self,
        corpus: Corpus,
        cv=KFold(n_splits=5, shuffle=True),
        selector: Callable[[CorpusComponent], bool] = lambda x: True,
    ):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label,
        using cross-validation for data splitting.

        :param corpus: target Corpus
        :param cv: cross-validation model to use: KFold(n_splits=5, shuffle=True) by default.
        :param selector: if running on a Corpus, this is a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.

        :return: cross-validated accuracy score
        """

        X, y = extract_vector_feats_and_label(
            corpus, self.obj_type, self.vector_name, self.columns, self.labeller, selector
        )
        print("Running a cross-validated evaluation...", end="")
        score = cross_val_score(self.clf, X, y, cv=cv)
        print("Done.")
        return score
