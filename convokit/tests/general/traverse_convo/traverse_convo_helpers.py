from convokit import Corpus, Speaker, Utterance


def construct_multiple_convo_id_corpus():
    # test broken convo where there are multiple conversation_ids
    corpus = Corpus(
        utterances=[
            Utterance(
                id="0",
                text="hello world",
                conversation_id="convo_id_0",
                reply_to=None,
                speaker=Speaker(id="alice"),
                timestamp=0,
            ),
            Utterance(
                id="1",
                text="my name is bob",
                conversation_id="convo_id_0",
                reply_to="0",
                speaker=Speaker(id="bob"),
                timestamp=2,
            ),
            Utterance(
                id="2",
                text="this is a test",
                conversation_id="convo_id_0",
                reply_to="1",
                speaker=Speaker(id="charlie"),
                timestamp=1,
            ),
            Utterance(
                id="3",
                text="hello world 2",
                conversation_id="convo_id_0",
                reply_to=None,
                speaker=Speaker(id="alice2"),
                timestamp=0,
            ),
        ]
    )
    return corpus


def construct_nonexistent_reply_to_corpus():
    corpus = Corpus(
        utterances=[
            Utterance(
                id="0",
                text="hello world",
                conversation_id="convo_id_0",
                reply_to=None,
                speaker=Speaker(id="alice"),
                timestamp=0,
            ),
            Utterance(
                id="1",
                text="my name is bob",
                conversation_id="convo_id_0",
                reply_to="0",
                speaker=Speaker(id="bob"),
                timestamp=2,
            ),
            Utterance(
                id="2",
                text="this is a test",
                conversation_id="convo_id_0",
                reply_to="1",
                speaker=Speaker(id="charlie"),
                timestamp=1,
            ),
            Utterance(
                id="3",
                text="hello world 2",
                conversation_id="convo_id_0",
                reply_to="9",
                speaker=Speaker(id="alice2"),
                timestamp=0,
            ),
        ]
    )
    return corpus


def construct_tree_corpus():
    corpus = Corpus(
        utterances=[
            Utterance(
                id="0",
                reply_to=None,
                conversation_id="0",
                speaker=Speaker(id="alice"),
                timestamp=0,
            ),
            Utterance(
                id="2", reply_to="0", conversation_id="0", speaker=Speaker(id="alice"), timestamp=2,
            ),
            Utterance(
                id="1", reply_to="0", conversation_id="0", speaker=Speaker(id="alice"), timestamp=1,
            ),
            Utterance(
                id="3", reply_to="0", conversation_id="0", speaker=Speaker(id="alice"), timestamp=3,
            ),
            Utterance(
                id="4", reply_to="1", conversation_id="0", speaker=Speaker(id="alice"), timestamp=4,
            ),
            Utterance(
                id="5", reply_to="1", conversation_id="0", speaker=Speaker(id="alice"), timestamp=5,
            ),
            Utterance(
                id="6", reply_to="1", conversation_id="0", speaker=Speaker(id="alice"), timestamp=6,
            ),
            Utterance(
                id="7", reply_to="2", conversation_id="0", speaker=Speaker(id="alice"), timestamp=4,
            ),
            Utterance(
                id="8", reply_to="2", conversation_id="0", speaker=Speaker(id="alice"), timestamp=5,
            ),
            Utterance(
                id="9", reply_to="3", conversation_id="0", speaker=Speaker(id="alice"), timestamp=4,
            ),
            Utterance(
                id="10",
                reply_to="4",
                conversation_id="0",
                speaker=Speaker(id="alice"),
                timestamp=5,
            ),
            Utterance(
                id="11",
                reply_to="9",
                conversation_id="0",
                speaker=Speaker(id="alice"),
                timestamp=10,
            ),
            Utterance(
                id="other",
                reply_to=None,
                conversation_id="other",
                speaker=Speaker(id="alice"),
                timestamp=99,
            ),
        ]
    )
    """
            Basic Conversation tree (left to right within subtree => earliest to latest)
                       0
                1      2      3
              4 5 6   7 8     9
            10                11
            """

    corpus.get_conversation("0").meta["hey"] = "jude"
    corpus.meta["foo"] = "bar"
    return corpus
