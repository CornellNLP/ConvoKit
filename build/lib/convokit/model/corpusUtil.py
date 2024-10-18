"""
Contains various methods used by Corpus components

"""

import pandas as pd


def get_utterances_dataframe(obj, selector=lambda utt: True, exclude_meta: bool = False):
    """
    Get a DataFrame of the utterances of a given object with fields and metadata attributes,
    with an optional selector that filters for utterances that should be included.
    Edits to the DataFrame do not change the corpus in any way.

    :param exclude_meta: whether to exclude metadata
    :param selector: a (lambda) function that takes a Utterance and returns True or False (i.e. include / exclude).
        By default, the selector includes all Utterances that compose the object.
    :return: a pandas DataFrame
    """
    ds = dict()
    for utt in obj.iter_utterances(selector):
        d = utt.to_dict().copy()
        if not exclude_meta:
            for k, v in d["meta"].items():
                d["meta." + k] = v
        del d["meta"]
        ds[utt.id] = d

    df = pd.DataFrame(ds).T
    df = df.set_index("id")
    df["speaker"] = df["speaker"].map(lambda spkr: spkr.id)
    meta_columns = [k for k in df.columns if k.startswith("meta.")]
    return df[
        ["timestamp", "text", "speaker", "reply_to", "conversation_id"] + meta_columns + ["vectors"]
    ]


def get_conversations_dataframe(obj, selector=lambda convo: True, exclude_meta: bool = False):
    """
    Get a DataFrame of the conversations of a given object with fields and metadata attributes,
    with an optional selector that filters for conversations that should be included.
    Edits to the DataFrame do not change the corpus in any way.

    :param exclude_meta: whether to exclude metadata
    :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
        By default, the selector includes all Conversations in the Corpus.
    :return: a pandas DataFrame
    """
    ds = dict()
    for convo in obj.iter_conversations(selector):
        d = convo.to_dict().copy()
        if not exclude_meta:
            for k, v in d["meta"].items():
                d["meta." + k] = v
        del d["meta"]
        ds[convo.id] = d

    df = pd.DataFrame(ds).T
    return df.set_index("id")


def get_speakers_dataframe(obj, selector=lambda utt: True, exclude_meta: bool = False):
    """
    Get a DataFrame of the Speakers with fields and metadata attributes, with an optional selector that filters
    Speakers that should be included. Edits to the DataFrame do not change the corpus in any way.

    :param exclude_meta: whether to exclude metadata
    :param selector: selector: a (lambda) function that takes a Speaker and returns True or False
        (i.e. include / exclude). By default, the selector includes all Speakers in the Corpus.
    :return: a pandas DataFrame
    """
    ds = dict()
    for spkr in obj.iter_speakers(selector):
        d = spkr.to_dict().copy()
        if not exclude_meta:
            for k, v in d["meta"].items():
                d["meta." + k] = v
        del d["meta"]
        ds[spkr.id] = d

    df = pd.DataFrame(ds).T
    return df.set_index("id")
