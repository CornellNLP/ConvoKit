import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict

from convokit import Corpus, Speaker, Utterance
from convokit import TextParser

from pandas import DataFrame
from typing import List, Dict, Set

DATA_DIR = "Stanford_politeness_corpus/"
OUT_DIR = "/convokit_data"


def convert_df_to_corpus(df: DataFrame, id_col: str, text_col: str, meta_cols: List[str]) -> Corpus:
    """Helper function to convert data to Corpus format

    Arguments:
        df {DataFrame} -- Actual data, in a pandas Dataframe
        id_col {str} -- name of the column that corresponds to utterances ids
        text_col {str} -- name of the column that stores texts of the utterances
        meta_cols {List[str]} -- set of columns that stores relevant metadata

    Returns:
        Corpus -- the converted corpus
    """

    # in this particular case, speaker, reply_to, and timestamp information are all not applicable
    # and we will simply either create a placeholder entry, or leave it as None

    generic_speaker = Speaker(id="speaker")
    time = "NOT_RECORDED"

    utterance_list = []
    for index, row in tqdm(df.iterrows()):
        # extracting meta data
        metadata = {}
        for meta_col in meta_cols:
            metadata[meta_col] = row[meta_col]

        utterance_list.append(
            Utterance(
                id=str(row[id_col]),
                speaker=generic_speaker,
                conversation_id=str(row[id_col]),
                reply_to=None,
                timestamp=time,
                text=row[text_col],
                meta=metadata,
            )
        )

    return Corpus(utterances=utterance_list)


def prepare_corpus_df(filename, data_dir=DATA_DIR):
    df = pd.read_csv(os.path.join(data_dir, filename))

    # if Id is not uniquely identifiable, use df index
    if len(set(df["Id"])) < len(df):
        df["Id"] = df.index

    df["Annotations"] = [
        dict(
            zip(
                [df.iloc[i]["TurkId{}".format(j)] for j in range(1, 6)],
                [df.iloc[i]["Score{}".format(j)] for j in range(1, 6)],
            )
        )
        for i in tqdm(range(len(df)))
    ]

    top = np.percentile(df["Normalized Score"], 75)
    bottom = np.percentile(df["Normalized Score"], 25)

    df["Binary"] = [int(score >= top) - int(score <= bottom) for score in df["Normalized Score"]]

    return df


if __name__ == "__main__":
    parser = TextParser(verbosity=500)

    for name in ["wikipedia.annotated.csv", "stack-exchange.annotated.csv"]:
        corpus_name = "{}-politeness-corpus".format(name.split(".")[0])
        df = prepare_corpus_df(name)

        print("constructing corpus {}".format(corpus_name))
        corpus = convert_df_to_corpus(
            df, "Id", "Request", ["Normalized Score", "Binary", "Annotations"]
        )

        print("parsing corpus...")
        corpus = parser.transform(corpus)
        corpus.dump(corpus_name, base_path=OUT_DIR)
