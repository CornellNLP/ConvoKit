from typing import List, Callable, Union

import pandas as pd

from convokit import Corpus, Transformer, CorpusComponent


class Ranker(Transformer):
    """
    Ranker sorts the objects in the Corpus by a given scoring function and annotates the objects with their rankings.

    :param obj_type: type of Corpus object to rank: 'conversation', 'speaker', or 'utterance'
    :param score_func: function for computing the score of a given object
    :param score_attribute_name: metadata attribute name to use in annotation for score value, default: "score"
    :param rank_attribute_name: metadata attribute name to use in annotation for the rank value, default: "rank"
    """

    def __init__(
        self,
        obj_type: str,
        score_func: Callable[[CorpusComponent], Union[int, float]],
        score_attribute_name: str = "score",
        rank_attribute_name: str = "rank",
    ):
        self.obj_type = obj_type
        self.score_func = score_func
        self.score_attribute_name = score_attribute_name
        self.rank_attribute_name = rank_attribute_name

    def transform(
        self, corpus: Corpus, y=None, selector: Callable[[CorpusComponent], bool] = lambda obj: True
    ) -> Corpus:
        """
        Annotate corpus objects with scores and rankings.

        :param corpus: target corpus
        :param selector: (lambda) function taking in a Corpus object and returning True / False; selects for Corpus objects to annotate.
        :return: annotated corpus
        """
        obj_iters = {
            "conversation": corpus.iter_conversations,
            "speaker": corpus.iter_speakers,
            "utterance": corpus.iter_utterances,
        }
        obj_scores = [(obj.id, self.score_func(obj)) for obj in obj_iters[self.obj_type](selector)]
        df = (
            pd.DataFrame(obj_scores, columns=["id", self.score_attribute_name])
            .set_index("id")
            .sort_values(self.score_attribute_name, ascending=False)
        )
        df[self.rank_attribute_name] = [idx + 1 for idx, _ in enumerate(df.index)]

        for obj in corpus.iter_objs(obj_type=self.obj_type):
            if obj.id in df.index:
                obj.add_meta(self.score_attribute_name, df.loc[obj.id][self.score_attribute_name])
                obj.add_meta(self.rank_attribute_name, df.loc[obj.id][self.rank_attribute_name])
            else:
                obj.add_meta(self.score_attribute_name, None)
                obj.add_meta(self.rank_attribute_name, None)
        return corpus

    def transform_objs(self, objs: List[CorpusComponent]):
        """
        Annotate list of Corpus objects with scores and rankings.

        :param objs: target list of Corpus objects
        :return: list of annotated COrpus objects
        """
        obj_scores = [(obj.id, self.score_func(obj)) for obj in objs]
        df = (
            pd.DataFrame(obj_scores, columns=["id", self.score_attribute_name])
            .set_index("id")
            .sort_values(self.score_attribute_name, ascending=False)
        )
        df[self.rank_attribute_name] = [idx + 1 for idx, _ in enumerate(df.index)]
        for obj in objs:
            obj.add_meta(self.score_attribute_name, df.loc[obj.id][self.score_attribute_name])
            obj.add_meta(self.rank_attribute_name, df.loc[obj.id][self.rank_attribute_name])
        return objs

    def summarize(
        self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda obj: True
    ):
        """
        Generate a dataframe indexed by object id, containing score + rank, and sorted by rank (in ascending order) of the objects in an annotated corpus, with an optional selector selecting which objects to be included in the dataframe

        :param corpus: annotated target corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :return: a pandas DataFrame
        """
        obj_iters = {
            "conversation": corpus.iter_conversations,
            "speaker": corpus.iter_speakers,
            "utterance": corpus.iter_utterances,
        }
        obj_scores_ranks = [
            (obj.id, obj.meta[self.score_attribute_name], obj.meta[self.rank_attribute_name])
            for obj in obj_iters[self.obj_type](selector)
        ]

        df = (
            pd.DataFrame(
                obj_scores_ranks,
                columns=["id", self.score_attribute_name, self.rank_attribute_name],
            )
            .set_index("id")
            .sort_values(self.rank_attribute_name, ascending=True)
        )

        return df

    def summarize_objs(self, objs: List[CorpusComponent]):
        """
        Generate a dataframe indexed by object id, containing score + rank, and sorted by rank (in ascending order) of the objects in an annotated corpus, or a list of corpus objects

        :param objs: list of annotated corpus objects
        :return: a pandas DataFrame
        """
        obj_scores_ranks = [
            (obj.id, obj.meta[self.score_attribute_name], obj.meta[self.rank_attribute_name])
            for obj in objs
        ]
        df = (
            pd.DataFrame(
                obj_scores_ranks,
                columns=["id", self.score_attribute_name, self.rank_attribute_name],
            )
            .set_index("id")
            .sort_values(self.rank_attribute_name, ascending=True)
        )

        return df
