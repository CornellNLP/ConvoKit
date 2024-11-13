from convokit import Corpus, Conversation, Utterance, Transformer
from typing import Callable, Optional, Union, Any, List, Iterator
from collections import namedtuple
from .forecasterModel import ForecasterModel
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Define a namedtuple template to represent conversational context tuples
ContextTuple = namedtuple(
    "ContextTuple", ["context", "current_utterance", "future_context", "conversation_id"]
)


class Forecaster(Transformer):
    """
    A wrapper class that provides a consistent, Transformer-style interface to any conversational forecasting model.
    From a user perspective, this makes it easy to apply forecasting models to ConvoKit corpora and evaluate them without
    having to know a lot about the inner workings of conversational forecasting, and to swap between different kinds of
    models without having to change a lot of code. From a developer perspective, this provides a prebuilt foundation upon which
    new conversational forecasting models can be easily developed, as the Forecaster class handles to complicated work of
    iterating over conversational contexts in temporal fashion, allowing the developer to focus only on writing the code to handle each conversational context.

    :param forecaster_model: An instance of a ForecasterModel subclass that implements the conversational forecasting model you want to use. ConvoKit provides CRAFT and BERT implementations.
    :param labeler: A function that specifies where/how to find the label for any given conversation. Alternatively, a string can be provided, in which case it will be interpreted as the name of a Conversation metadata field containing the label.
    :param context_preprocessor: An optional function that allows simple preprocessing of conversational contexts. Note that this should NOT be used to perform any restructuring or feature engineering on the data (that work is considered the exclusive purview of the underlying ForecasterModel); instead, it is intended to perform simple Corpus-specific data cleaning steps (i.e., removing utterances that lack key metadata required by the model)
    :param forecast_attribute_name: metadata feature name to use in annotation for forecast result, default: "forecast"
    :param forecast_prob_attribute_name: metadata feature name to use in annotation for forecast result probability, default: "forecast_prob"
    """

    def __init__(
        self,
        forecaster_model: ForecasterModel,
        labeler: Union[Callable[[Conversation], int], str],
        context_preprocessor: Optional[Callable[[List[Utterance]], List[Utterance]]] = None,
        forecast_attribute_name: str = "forecast",
        forecast_prob_attribute_name: str = "forecast_prob",
    ):
        self.forecaster_model = forecaster_model
        if type(labeler) == str:
            # assume the string is the name of a conversation metadata field containing the label
            self.labeler = lambda c: int(c.meta[labeler])
        else:
            self.labeler = labeler
        self.context_preprocessor = context_preprocessor
        self.forecast_attribute_name = forecast_attribute_name
        self.forecast_prob_attribute_name = forecast_prob_attribute_name

        # also give the underlying ForecasterModel access to the labeler function
        self.forecaster_model.labeler = self.labeler

    def _create_context_iterator(
        self,
        corpus: Corpus,
        context_selector: Callable[[ContextTuple], bool],
        include_future_context: bool = False,
    ) -> Iterator[ContextTuple]:
        """
        Helper function that generates an iterator over conversational contexts that satisfy the provided context selector,
        across the entire corpus
        """
        for convo in corpus.iter_conversations():
            # contexts are iterated in chronological order, representing the idea that conversational forecasting models
            # must make an updated forecast every time a new utterance is posted
            chronological_utts = convo.get_chronological_utterance_list()
            for i in range(len(chronological_utts)):
                current_utt = chronological_utts[i]
                # context is all utterances up to and including the most recent utterance
                context = chronological_utts[: (i + 1)]
                # if a preprocessor is given, run it first to get the "clean" version of the context
                if self.context_preprocessor is not None:
                    context = self.context_preprocessor(context)
                if include_future_context:
                    if i == len(chronological_utts) - 1:
                        # not to be confused with future_context=None, which indicates that include_future_context was false;
                        # this special value indicates that include_future_context is true but there is no future context
                        # (because we are at the end of the conversation)
                        future_context = []
                    else:
                        future_context = [chronological_utts[(i + 1) :]]
                else:
                    future_context = None
                # pack the full context tuple
                context_tuple = ContextTuple(context, current_utt, future_context, convo.id)
                # the current context tuple should be skipped if it does not satisfy the given selector,
                # or the context is empty (which may happen as a result of preprocessing)
                if len(context_tuple.context) == 0 or not context_selector(context_tuple):
                    continue
                # if the current context was not skipped, it is next in the iterator
                yield context_tuple

    def fit(
        self,
        corpus: Corpus,
        context_selector: Callable[[ContextTuple], bool] = lambda context: True,
        val_context_selector: Optional[Callable[[ContextTuple], bool]] = None,
    ):
        """
        Wrapper method for training the underlying conversational forecasting model. Forecaster itself does not implement any actual training logic.
        Instead, it handles the job of selecting and iterating over context tuples. The resulting iterator is presented as a parameter to the fit
        method of the underlying model, which can process the tuples however it sees fit. Within each tuple, context is unstructured - it contains all
        utterances temporally preceding the most recent utterance, plus that most recent utterance itself, but does not impose any particular structure
        beyond that, allowing each conversational forecasting model to decide how it wants to define “context”.

        :param corpus: The Corpus containing the data to train on
        :param context_selector: A function that takes in a context tuple and returns a boolean indicator of whether it should be included in training data. This can be used to both select data based on splits (i.e. keep only those in the “train” split) and to specify special behavior of what contexts are looked at in training (i.e. in CRAFT where only the last context, directly preceding the toxic comment, is used in training).
        :param val_context_selector: An optional function that mirrors context_selector but is used to create a separate held-out validation set

        :return: fitted Forecaster Transformer
        """
        contexts = self._create_context_iterator(
            corpus, context_selector, include_future_context=True
        )
        val_contexts = None
        if val_context_selector is not None:
            val_contexts = self._create_context_iterator(
                corpus, val_context_selector, include_future_context=True
            )
        self.forecaster_model.fit(contexts, val_contexts)

        return self

    def transform(
        self,
        corpus: Corpus,
        context_selector: Callable[[ContextTuple], bool] = lambda context: True,
    ) -> Corpus:
        """
        Wrapper method for applying the underlying conversational forecasting model to make forecasts over the Conversations in a given Corpus.
        Like the fit method, this simply acts to create an iterator over context tuples to be transformed, and forwards the iterator to the
        underlying conversational forecasting model to do the actual forecasting.

        :param corpus: the Corpus containing the data to run on
        :param context_selector: A function that takes in a context tuple and returns a boolean indicator of whether it should be included. Excluded contexts will simply not have a forecast.

        :return: annotated Corpus
        """
        contexts = self._create_context_iterator(corpus, context_selector)
        forecast_df = self.forecaster_model.transform(
            contexts, self.forecast_attribute_name, self.forecast_prob_attribute_name
        )

        for utt in corpus.iter_utterances():
            if utt.id in forecast_df.index:
                utt.add_meta(
                    self.forecast_attribute_name,
                    forecast_df.loc[utt.id][self.forecast_attribute_name],
                )
                utt.add_meta(
                    self.forecast_prob_attribute_name,
                    forecast_df.loc[utt.id][self.forecast_prob_attribute_name],
                )
            else:
                utt.add_meta(self.forecast_attribute_name, None)
                utt.add_meta(self.forecast_prob_attribute_name, None)

        return corpus

    def fit_transform(
        self,
        corpus: Corpus,
        context_selector: Callable[[ContextTuple], bool] = lambda context: True,
    ) -> Corpus:
        """
        Convenience method for running fit and transform on the same data

        :param corpus: the Corpus containing the data to run on
        :param context_selector: A function that takes in a context tuple and returns a boolean indicator of whether it should be included. Excluded contexts will simply not have a forecast.

        :return: annotated Corpus
        """
        self.fit(corpus, context_selector)
        return self.transform(corpus, context_selector)

    def _draw_horizon_plot(
        self, corpus: Corpus, selector: Callable[[Conversation], bool] = lambda convo: True
    ):
        """
        Draw the "forecast horizon" plot showing how far before the end of the conversation the first forecast is made
        (for true positives). Note this is not always an especially meaningful plot, if the Corpus being used includes
        to-be-forecasted events earlier in the conversation and not at the end, but it works for datasets like
        CGA-CMV where the event is defined to be after the end of the included utterances.
        """
        comments_until_end = {}
        for convo in corpus.iter_conversations():
            if selector(convo) and self.labeler(convo) == 1:
                for i, utt in enumerate(convo.get_chronological_utterance_list()):
                    prediction = utt.meta.get(self.forecast_attribute_name)
                    if prediction is not None and prediction > 0:
                        comments_until_end[convo.id] = (
                            len(convo.get_chronological_utterance_list()) - i
                        )
                        break
        comments_until_end_vals = list(comments_until_end.values())
        plt.hist(
            comments_until_end_vals, bins=range(1, np.max(comments_until_end_vals)), density=True
        )
        plt.xlabel(
            "Number of comments between index of first positive forecast and end of conversation"
        )
        plt.ylabel("Percent of convesations")
        plt.show()
        return comments_until_end

    def summarize(
        self, corpus: Corpus, selector: Callable[[Conversation], bool] = lambda convo: True
    ):
        """
        Compute and display conversation-level performance metrics over a Corpus that has already been annotated by transform

        :param corpus: the Corpus containing the forecasts to evaluate
        :param selector: A filtering function to limit the conversations the metrics are computed over. Note that unlike the context_selectors used in fit and transform, this selector operates on conversations (since evaluation is conversation-level).
        """
        conversational_forecasts_df = {
            "conversation_id": [],
            "label": [],
            "score": [],
            "forecast": [],
        }
        for convo in corpus.iter_conversations():
            if selector(convo):
                conversational_forecasts_df["conversation_id"].append(convo.id)
                conversational_forecasts_df["label"].append(self.labeler(convo))
                forecasts = np.asarray(
                    [
                        utt.meta[self.forecast_attribute_name]
                        for utt in convo.iter_utterances()
                        if utt.meta.get(self.forecast_attribute_name, None) is not None
                    ]
                )
                forecast_scores = np.asarray(
                    [
                        utt.meta[self.forecast_prob_attribute_name]
                        for utt in convo.iter_utterances()
                        if utt.meta.get(self.forecast_prob_attribute_name, None) is not None
                    ]
                )
                conversational_forecasts_df["score"].append(np.max(forecast_scores))
                conversational_forecasts_df["forecast"].append(np.max(forecasts))
        conversational_forecasts_df = pd.DataFrame(conversational_forecasts_df).set_index(
            "conversation_id"
        )

        acc = (
            conversational_forecasts_df["label"] == conversational_forecasts_df["forecast"]
        ).mean()
        tp = (
            (conversational_forecasts_df["label"] == 1)
            & (conversational_forecasts_df["forecast"] == 1)
        ).sum()
        fp = (
            (conversational_forecasts_df["label"] == 0)
            & (conversational_forecasts_df["forecast"] == 1)
        ).sum()
        tn = (
            (conversational_forecasts_df["label"] == 0)
            & (conversational_forecasts_df["forecast"] == 0)
        ).sum()
        fn = (
            (conversational_forecasts_df["label"] == 1)
            & (conversational_forecasts_df["forecast"] == 0)
        ).sum()
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        fpr = fp / (fp + tn)
        f1 = 2 / (((tp + fp) / tp) + ((tp + fn) / tp))
        metrics = {"Accuracy": acc, "Precision": p, "Recall": r, "FPR": fpr, "F1": f1}

        print(pd.Series(metrics))

        comments_until_end = self._draw_horizon_plot(corpus, selector)
        comments_until_end_vals = list(comments_until_end.values())
        print(
            "Horizon statistics (# of comments between first positive forecast and conversation end):"
        )
        print(
            f"Mean = {np.mean(comments_until_end_vals)}, Median = {np.median(comments_until_end_vals)}"
        )

        return conversational_forecasts_df, metrics
