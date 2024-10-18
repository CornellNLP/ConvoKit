from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from convokit import Transformer
from convokit.classifier.util import *


class Classifier(Transformer):
    """
    Transformer that trains a classifier on the specified features of a Corpus's objects.

    Runs on the Corpus's Speakers, Utterances, or Conversations (as specified by obj_type).

    :param obj_type: type of Corpus object to classify: 'conversation', 'speaker', or 'utterance'
    :param pred_feats: list of metadata attributes containing the features to be used in prediction.
        If the metadata attribute contains a dictionary, all the keys of the dictionary will be included in pred_feats.
        Each feature used should have a numeric/boolean type.
    :param labeller: a (lambda) function that takes a Corpus object and returns True (y=1) or False (y=0)
        - i.e. labeller defines the y value of the object for fitting
    :param clf: optional sklearn classifier model. By default, clf is a Pipeline with StandardScaler and LogisticRegression.
    :param clf_attribute_name: the metadata attribute name to store the classifier prediction value under; default: "prediction"
    :param clf_prob_attribute_name: the metadata attribute name to store the classifier prediction score under; default: "pred_score"

    """

    def __init__(
        self,
        obj_type: str,
        pred_feats: List[str],
        labeller: Callable[[CorpusComponent], bool] = lambda x: True,
        clf=None,
        clf_attribute_name: str = "prediction",
        clf_prob_attribute_name: str = "pred_score",
    ):
        self.pred_feats = pred_feats
        self.labeller = labeller
        self.obj_type = obj_type
        if clf is None:
            clf = Pipeline(
                [
                    ("standardScaler", StandardScaler(with_mean=False)),
                    ("logreg", LogisticRegression(solver="liblinear")),
                ]
            )
            print("Initialized default classification model (standard scaled logistic regression).")
        self.clf = clf
        self.clf_attribute_name = clf_attribute_name
        self.clf_prob_attribute_name = clf_prob_attribute_name

    def fit(
        self, corpus: Corpus, y=None, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ):
        """
        Trains the Transformer's classifier model, with an optional selector that filters for objects to be fit on.

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
            By default, the selector includes all objects of the specified type in the Corpus.
        :return: the fitted Classifier Transformer
        """
        X, y = extract_feats_and_label(
            corpus, self.obj_type, self.pred_feats, self.labeller, selector
        )
        self.clf.fit(X, y)
        return self

    def transform(
        self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ) -> Corpus:
        """
        Run classifier on given corpus's objects and annotate them with the predictions and prediction scores,
        with an optional selector that filters for objects to be classified. Objects that are not selected will get
        a metadata value of 'None' instead of the classifier prediction.

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
            By default, the selector includes all objects of the specified type in the Corpus.

        :return: annotated Corpus
        """
        obj_id_to_feats = extract_feats_dict(corpus, self.obj_type, self.pred_feats, selector)
        feats_df = pd.DataFrame.from_dict(obj_id_to_feats, orient="index").reindex(
            index=list(obj_id_to_feats)
        )
        X = csr_matrix(feats_df.values.astype("float64"))
        idx_to_id = {idx: obj_id for idx, obj_id in enumerate(list(obj_id_to_feats))}
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            corpus_obj = corpus.get_object(self.obj_type, idx_to_id[idx])
            corpus_obj.add_meta(self.clf_attribute_name, clf)
            corpus_obj.add_meta(self.clf_prob_attribute_name, clf_prob)

        for obj in corpus.iter_objs(self.obj_type, selector):
            if self.clf_attribute_name not in obj.meta:
                obj.meta[self.clf_attribute_name] = None
                obj.meta[self.clf_prob_attribute_name] = None

        return corpus

    def fit_transform(
        self, corpus: Corpus, y=None, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ) -> Corpus:
        self.fit(corpus, selector=selector)
        return self.transform(corpus, selector=selector)

    def transform_objs(self, objs: List[CorpusComponent]) -> List[CorpusComponent]:
        """
        Run classifier on list of Corpus objects and annotate them with the predictions and prediction scores

        :param objs: list of Corpus objects

        :return: list of annotated Corpus objects
        """
        X = np.array([list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs])
        # obj_ids = [obj.id for obj in objs]
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_attribute_name, clf)
            obj.add_meta(self.clf_prob_attribute_name, clf_prob)

        return objs

    def summarize(
        self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ):
        """
        Generate a pandas DataFrame (indexed by object id, with prediction and prediction score columns) of classification results.

        Run either on a target Corpus or a list of Corpus objects

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.

        :return: pandas DataFrame indexed by Corpus object id
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
            .sort_values(self.clf_prob_attribute_name)
        )

    def summarize_objs(self, objs: List[CorpusComponent]):
        """
        Generate a pandas DataFrame (indexed by object id, with prediction and prediction score columns) of classification results.

        Runs on a list of Corpus objects.

        :param objs: list of Corpus objects
        :return: pandas DataFrame indexed by Corpus object id
        """
        objId_clf_prob = []
        for obj in objs:
            objId_clf_prob.append(
                (obj.id, obj.meta[self.clf_attribute_name], obj.meta[self.clf_prob_attribute_name])
            )

        return (
            pd.DataFrame(
                list(objId_clf_prob),
                columns=["id", self.clf_attribute_name, self.clf_prob_attribute_name],
            )
            .set_index("id")
            .sort_values(self.clf_prob_attribute_name)
        )

    def evaluate_with_train_test_split(
        self,
        corpus: Corpus = None,
        objs: List[CorpusComponent] = None,
        selector: Callable[[CorpusComponent], bool] = lambda x: True,
        test_size: float = 0.2,
    ):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label,
        using a train-test split.

        Run either on a Corpus (with Classifier labeller, selector, obj_type settings) or a list of Corpus objects

        :param corpus: target Corpus
        :param objs: target list of Corpus objects
        :param selector: if running on a Corpus, this is a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :param test_size: size of test set
        :return: accuracy and confusion matrix
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError(
                "This function takes in either a Corpus or a list of speakers / utterances / conversations"
            )

        if corpus:
            print("Using corpus objects...")
            X, y = extract_feats_and_label(
                corpus, self.obj_type, self.pred_feats, self.labeller, selector
            )
        else:
            assert objs is not None
            print("Using input list of corpus objects...")
            X = np.array(
                [list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs]
            )
            y = np.array([self.labeller(obj) for obj in objs])

        print("Running a train-test-split evaluation...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        accuracy = np.mean(preds == y_test)
        print("Done.")
        return accuracy, confusion_matrix(y_true=y_test, y_pred=preds)

    # def evaluate_with_cv(self, corpus: Corpus = None,
    #                      objs: List[CorpusComponent] = None,
    #                      cv=KFold(n_splits=5),
    #                      selector: Callable[[CorpusComponent], bool] = lambda x: True
    #                      ):
    #     return
    def evaluate_with_cv(
        self,
        corpus: Corpus = None,
        objs: List[CorpusComponent] = None,
        cv=KFold(n_splits=5, shuffle=True),
        selector: Callable[[CorpusComponent], bool] = lambda x: True,
    ):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label,
        using cross-validation for data splitting.

        This method can be run on either a Corpus (passed in as the `corpus` parameter) or a list of Corpus
        component objects (passed in as the `objs` parameter). If run on a Corpus, the cross-validation will be run
        with the Classifier's `labeller` and `obj_type` settings, and the `selector` parameter of this function.

        :param corpus: target Corpus (do not pass in objs if using this)
        :param objs: target list of Corpus objects (do not pass in corpus if using this)
        :param cv: cross-validation model to use: KFold(n_splits=5, shuffle=True) by default.
        :param selector: if running on a Corpus, this is a (lambda) function that takes a Corpus object and returns
            True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type
            in the Corpus.

        :return: cross-validated accuracy score
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError(
                "This function takes in either a Corpus or a list of speakers / utterances / conversations"
            )

        if corpus:
            print("Using corpus objects...")
            X, y = extract_feats_and_label(
                corpus, self.obj_type, self.pred_feats, self.labeller, selector
            )
        else:
            assert objs is not None
            print("Using input list of corpus objects...")
            X = np.array(
                [list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs]
            )
            y = np.array([self.labeller(obj) for obj in objs])

        print("Running a cross-validated evaluation...")
        score = cross_val_score(self.clf, X, y, cv=cv)
        print("Done.")
        return score

    def confusion_matrix(
        self, corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ):
        """
        Generate confusion matrix for transformed corpus using labeller for y_true and clf_attribute_name as y_pred

        :param corpus: target Corpus
        :param selector: (lambda) function selecting objects to include in this confusion_matrix; uses all objects by default
        :return: sklearn confusion matrix
        """
        y_true = []
        y_pred = []
        for obj in corpus.iter_objs(self.obj_type, selector):
            y_true.append(self.labeller(obj))
            y_pred.append(obj.meta[self.clf_attribute_name])

        return confusion_matrix(y_true=y_true, y_pred=y_pred)

    def base_accuracy(self, corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True):
        """
        Get the base accuracy, i.e. the maximum of the percentages of results that are y=1 and y=0

        :param corpus: the classified Corpus
        :param selector: (lambda) function selecting objects to include in this accuracy calculation; uses all objects by default
        :return: float value
        """
        y_true, y_pred = self.get_y_true_pred(corpus, selector)
        all_true_accuracy = np.array(y_true).mean()
        return max(all_true_accuracy, 1 - all_true_accuracy)

    def accuracy(self, corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True):
        """
        Calculate the accuracy of the classification

        :param corpus: target Corpus
        :param selector: (lambda) function selecting objects to include in this accuracy calculation; uses all objects by default
        :return: float value
        """
        y_true, y_pred = self.get_y_true_pred(corpus, selector)
        return (np.array(y_true) == np.array(y_pred)).mean()

    def get_y_true_pred(self, corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True):
        """
        Get lists of true and predicted labels

        :param corpus: target Corpus
        :param selector: (lambda) function selecting objects to get labels for; uses all objects by default
        :return: list of true labels, and list of predicted labels
        """
        y_true = []
        y_pred = []
        for obj in corpus.iter_objs(self.obj_type, selector):
            y_true.append(self.labeller(obj))
            y_pred.append(obj.meta[self.clf_attribute_name])

        return y_true, y_pred

    def classification_report(
        self, corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ):
        """
        Generate classification report for transformed corpus using labeller for y_true and clf_attribute_name as y_pred

        :param corpus: target Corpus
        :param selector: (lambda) function selecting objects to include in this classification report
        :return: classification report
        """
        y_true = []
        y_pred = []
        for obj in corpus.iter_objs(self.obj_type, selector):
            y_true.append(self.labeller(obj))
            y_pred.append(obj.meta[self.clf_attribute_name])

        return classification_report(y_true=y_true, y_pred=y_pred)

    def get_coefs(self, feature_names: List[str], coef_func=None):
        """
        Get dataframe of classifier coefficients

        :param feature_names: list of feature names to get coefficients for
        :param coef_func: function for accessing the list of coefficients from the classifier model; by default,
                            assumes it is a pipeline with a logistic regression component
        :return: DataFrame of features and coefficients, indexed by feature names
        """
        return get_coefs_helper(self.clf, feature_names, coef_func)

    def get_model(self):
        """
        Gets the Classifier's internal model
        """
        return self.clf

    def set_model(self, clf):
        """
        Sets the Classifier's internal model
        """
        self.clf = clf
