from convokit import Corpus, Utterance, Speaker


def construct_base_corpus():
    return Corpus(
        utterances=[
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ]
    )


def construct_base_corpus_with_convo_id():
    return Corpus(
        utterances=[
            Utterance(
                id="0", text="hello world", conversation_id="convo1", speaker=Speaker(id="alice")
            ),
            Utterance(
                id="1", text="my name is bob", conversation_id="convo1", speaker=Speaker(id="bob")
            ),
            Utterance(
                id="2",
                text="this is a test",
                conversation_id="convo1",
                speaker=Speaker(id="charlie"),
            ),
        ]
    )


def construct_non_overlapping_corpus():
    return Corpus(
        utterances=[
            Utterance(id="3", text="i like pie", speaker=Speaker(id="delta")),
            Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ]
    )


def construct_overlapping_corpus():
    return Corpus(
        utterances=[
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie"),),
            Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo"),),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ]
    )


def construct_overlapping_corpus_with_convo_id():
    return Corpus(
        utterances=[
            Utterance(
                id="2",
                conversation_id="convo1",
                text="this is a test",
                speaker=Speaker(id="charlie"),
            ),
            Utterance(
                id="4",
                conversation_id="convo1",
                text="this is a sentence",
                speaker=Speaker(id="echo"),
            ),
            Utterance(
                id="5", conversation_id="convo1", text="goodbye", speaker=Speaker(id="foxtrot")
            ),
        ]
    )
