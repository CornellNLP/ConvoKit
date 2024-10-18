from .forecasterModel import ForecasterModel
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import List
import pandas as pd


class CumulativeBoW(ForecasterModel):
    """
    A cumulative bag-of-words forecasting model.

    :param vectorizer: optional vectorizer; default CV (min_df=10, max_df=0.5, ngram_range=(1,1), max_features=15000)
    :param clf_model: optional classifier model; default standard-scaled logistic regression
    :param use_tokens: if using default vectorizer, set this to true if input is already tokenized
    :param forecast_attribute_name: name for DataFrame column containing predictions, default: "prediction"
    :param forecast_prob_attribute_name: name for column containing prediction scores, default: "score"
    """

    def __init__(
        self,
        vectorizer=None,
        clf_model=None,
        use_tokens=False,
        forecast_attribute_name: str = "prediction",
        forecast_prob_attribute_name: str = "score",
    ):
        super().__init__(
            forecast_attribute_name=forecast_attribute_name,
            forecast_prob_attribute_name=forecast_prob_attribute_name,
        )
        if vectorizer is None:
            print("Initializing default unigram CountVectorizer...")
            if use_tokens:
                self.vectorizer = CV(
                    decode_error="ignore",
                    min_df=10,
                    max_df=0.5,
                    ngram_range=(1, 1),
                    binary=False,
                    max_features=15000,
                    tokenizer=lambda x: x,
                    preprocessor=lambda x: x,
                )
            else:
                self.vectorizer = CV(
                    decode_error="ignore",
                    min_df=10,
                    max_df=0.5,
                    ngram_range=(1, 1),
                    binary=False,
                    max_features=15000,
                )
        else:
            self.vectorizer = vectorizer

        if clf_model is None:
            print("Initializing default classification model (standard scaled logistic regression)")
            self.clf_model = Pipeline(
                [
                    ("standardScaler", StandardScaler(with_mean=False)),
                    ("logreg", LogisticRegression(solver="liblinear")),
                ]
            )
        else:
            self.clf_model = clf_model

    @staticmethod
    def _combine_contexts(id_to_context_others):
        """
        Combines the context part of the dictionary into a single entity (list or string) - this mutates the dictionary

        :param id_to_context_others: dictionary of utterance id to (context, reply, label)
        :return: input dictionary with context as a single entity (list or string)
        """
        for comment_id, (context, *others) in id_to_context_others.items():
            if isinstance(context[0], str):
                context_all = " ".join(utt_text for utt_text in context)
            elif isinstance(context, List):
                context_all = []
                for utt_tokens in context:
                    context_all.extend(utt_tokens)
            else:
                raise TypeError(
                    "Context utterances are of an invalid type. "
                    "The utterance should be either a List of tokens or a string."
                )
            context = context_all
            id_to_context_others[comment_id] = (context, *others)
        return id_to_context_others

    def train(self, id_to_context_reply_label):
        CumulativeBoW._combine_contexts(id_to_context_reply_label)
        contexts = [context for id_, (context, reply, label) in id_to_context_reply_label.items()]
        y = [label for id_, (context, reply, label) in id_to_context_reply_label.items()]
        X = self.vectorizer.fit_transform(contexts)
        print("Fitting cumulative BoW classification model...")
        self.clf_model.fit(X, y)
        print("Done.")

    def forecast(self, id_to_context_reply_label):
        try:
            self.vectorizer.vocabulary_
        except AttributeError:
            raise ValueError("Forecaster model has not been fitted yet.")

        CumulativeBoW._combine_contexts(id_to_context_reply_label)
        contexts = [context for id_, (context, reply, label) in id_to_context_reply_label.items()]
        X = self.vectorizer.transform(contexts)
        ids = [id_ for id_, _ in id_to_context_reply_label.items()]
        preds, pred_probs = self.clf_model.predict(X), self.clf_model.predict_proba(X)[:, 1]
        return pd.DataFrame(
            data=list(zip(ids, preds, pred_probs)),
            columns=["id", self.forecast_attribute_name, self.forecast_prob_attribute_name],
        ).set_index("id")
