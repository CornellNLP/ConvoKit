from typing import List

from convokit import Corpus, Speaker, Utterance


def construct_missing_convo_ids_corpus() -> Corpus:
    # test broken convo where there are multiple conversation_ids
    corpus = Corpus(
        utterances=[
            Utterance(
                id="0",
                reply_to=None,
                speaker=Speaker(id="alice"),
                timestamp=0,
            ),
            Utterance(
                id="1",
                reply_to="0",
                speaker=Speaker(id="bob"),
                timestamp=2,
            ),
            Utterance(
                id="2",
                reply_to="1",
                speaker=Speaker(id="charlie"),
                timestamp=1,
            ),
            Utterance(
                id="3",
                reply_to=None,
                speaker=Speaker(id="alice2"),
                timestamp=0,
            ),
        ]
    )
    return corpus


def get_new_utterances_without_convo_ids() -> List[Utterance]:
    return [
        Utterance(
            id="a",
            reply_to=None,
            speaker=Speaker(id="alice"),
            timestamp=0,
        ),
        Utterance(
            id="b",
            reply_to="a",
            speaker=Speaker(id="bob"),
            timestamp=0,
        ),
        Utterance(
            id="c",
            reply_to=None,
            speaker=Speaker(id="bob"),
            timestamp=0,
        ),
    ]


def get_new_utterances_without_existing_convo_ids():
    # i.e. they belong to existing convos
    # one responds to root utt, the other responds to leaf utt
    return [
        Utterance(
            id="z",
            reply_to="0",
            speaker=Speaker(id="alice"),
            timestamp=0,
        ),
        Utterance(
            id="zz",
            reply_to="2",
            speaker=Speaker(id="charlie"),
            timestamp=0,
        ),
    ]
