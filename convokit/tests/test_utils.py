import os.path
import shutil
from uuid import uuid4

import spacy
from spacy.tokens import Doc

from convokit.model import Corpus, Utterance, Speaker

FOX_TEXT = "A quick brown fox jumps over the lazy dog."
BUFFALO_TEXT = "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo"
FOX_BUFFALO_TEXT = (
    "A quick brown fox jumps over the lazy dog. Buffalo buffalo Buffalo buffalo "
    "buffalo buffalo Buffalo buffalo"
)
BURR_SIR_TEXT_1 = "Pardon me. Are you Aaron Burr, sir?"
BURR_SIR_TEXT_2 = "That depends. Who's asking?"
BURR_SIR_SENTENCE_1 = "Pardon me."
BURR_SIR_SENTENCE_2 = "Are you Aaron Burr, sir?"
BURR_SIR_SENTENCE_3 = "That depends."
BURR_SIR_SENTENCE_4 = "Who's asking?"


def spacy_en_vocab():
    return spacy.load("en_core_web_sm").vocab


def fox_spacy_doc():
    return Doc(
        vocab=spacy_en_vocab(),
        words=["A", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        heads=[4, 4, 4, 4, 4, 4, 8, 8, 5],
        deps=["det", "amod", "amod", "NN", "ROOT", "prep", "det", "amod", "pobj"],
        tags=["DT", "JJ", "JJ", "NN", "NNS", "IN", "DT", "JJ", "NN"],
    )


def buffalo_spacy_doc():
    return Doc(
        vocab=spacy_en_vocab(),
        words=[
            "Buffalo",
            "buffalo",
            "Buffalo",
            "buffalo",
            "buffalo",
            "buffalo",
            "Buffalo",
            "buffalo",
        ],
        heads=[1, 1, 3, 4, 5, 5, 7, 7],
        deps=["compound", "ROOT", "compound", "compound", "nsubj", "ROOT", "compound", "ROOT"],
        tags=["NNP", "NNP", "NNP", "NNP", "NNP", "NNP", "NNP", "NNP"],
    )


def fox_buffalo_spacy_doc():
    return Doc(
        vocab=spacy_en_vocab(),
        words=[
            "A",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "the",
            "lazy",
            "dog",
            ".",
            "Buffalo",
            "buffalo",
            "Buffalo",
            "buffalo",
            "buffalo",
            "buffalo",
            "Buffalo",
            "buffalo",
        ],
        heads=[4, 4, 4, 4, 4, 4, 8, 8, 5, 4, 11, 11, 13, 14, 15, 15, 17, 17],
        deps=[
            "det",
            "amod",
            "amod",
            "compound",
            "ROOT",
            "prep",
            "det",
            "amod",
            "pobj",
            "punct",
            "compound",
            "ROOT",
            "compound",
            "compound",
            "nsubj",
            "ROOT",
            "compound",
            "ROOT",
        ],
        tags=[
            "DT",
            "JJ",
            "JJ",
            "NN",
            "NNS",
            "IN",
            "DT",
            "JJ",
            "NN",
            ".",
            "NNP",
            "NNP",
            "NNP",
            "NNP",
            "NNP",
            "NNP",
            "NNP",
            "NNP",
        ],
    )


def small_burr_corpus():
    hamilton = Speaker(id="hamilton")
    burr = Speaker(id="burr")
    utterances = [
        Utterance(id="0", text=BURR_SIR_TEXT_1, speaker=hamilton),
        Utterance(id="1", text=BURR_SIR_TEXT_2, speaker=burr),
    ]

    return Corpus(utterances=utterances)


def small_burr_conv_corpus():
    hamilton = Speaker(id="hamilton")
    burr = Speaker(id="burr")

    utterances = [
        Utterance(id="0", text=BURR_SIR_SENTENCE_1, conversation_id="0", speaker=hamilton),
        Utterance(
            id="1", text=BURR_SIR_SENTENCE_2, conversation_id="0", reply_to="0", speaker=burr
        ),
        Utterance(id="2", text=BURR_SIR_SENTENCE_3, conversation_id="1", speaker=burr),
        Utterance(
            id="3", text=BURR_SIR_SENTENCE_4, conversation_id="1", reply_to="2", speaker=hamilton
        ),
        Utterance(
            id="4", text=BURR_SIR_SENTENCE_1, conversation_id="1", reply_to="3", speaker=burr
        ),
        Utterance(id="5", text=BURR_SIR_SENTENCE_2, conversation_id="2", speaker=hamilton),
        Utterance(
            id="6", text=BURR_SIR_SENTENCE_3, conversation_id="2", reply_to="5", speaker=burr
        ),
    ]

    return Corpus(utterances=utterances)


def small_burr_corpus_parsed():
    corpus = small_burr_corpus()
    utterance_infos = [
        {
            "parsed": [
                {
                    "rt": 0,
                    "toks": [
                        {"tok": "Pardon", "tag": "VB", "dep": "ROOT", "dn": [1, 2]},
                        {"tok": "me", "tag": "PRP", "dep": "dobj", "up": 0, "dn": []},
                        {"tok": ".", "tag": ".", "dep": "punct", "up": 0, "dn": []},
                    ],
                },
                {
                    "rt": 0,
                    "toks": [
                        {"tok": "Are", "tag": "VBP", "dep": "ROOT", "dn": [1, 3, 4, 5, 6]},
                        {"tok": "you", "tag": "PRP", "dep": "nsubj", "up": 0, "dn": []},
                        {"tok": "Aaron", "tag": "NNP", "dep": "compound", "up": 3, "dn": []},
                        {"tok": "Burr", "tag": "NNP", "dep": "attr", "up": 0, "dn": [2]},
                        {"tok": ",", "tag": ",", "dep": "punct", "up": 0, "dn": []},
                        {"tok": "sir", "tag": "NN", "dep": "npadvmod", "up": 0, "dn": []},
                        {"tok": "?", "tag": ".", "dep": "punct", "up": 0, "dn": []},
                    ],
                },
            ]
        },
        {
            "parsed": [
                {
                    "rt": 1,
                    "toks": [
                        {"tok": "That", "tag": "DT", "dep": "nsubj", "up": 1, "dn": []},
                        {"tok": "depends", "tag": "VBZ", "dep": "ROOT", "dn": [0, 2]},
                        {"tok": ".", "tag": ".", "dep": "punct", "up": 1, "dn": []},
                    ],
                },
                {
                    "rt": 2,
                    "toks": [
                        {"tok": "Who", "tag": "WP", "dep": "nsubj", "up": 2, "dn": []},
                        {"tok": "'s", "tag": "VBZ", "dep": "aux", "up": 2, "dn": []},
                        {"tok": "asking", "tag": "VBG", "dep": "ROOT", "dn": [0, 1, 3]},
                        {"tok": "?", "tag": ".", "dep": "punct", "up": 2, "dn": []},
                    ],
                },
            ]
        },
    ]

    for info_dict, utterance in zip(utterance_infos, corpus.iter_utterances()):
        utterance.meta = info_dict

    sentence_lists = [
        [BURR_SIR_SENTENCE_1, BURR_SIR_SENTENCE_2],
        [BURR_SIR_SENTENCE_3, BURR_SIR_SENTENCE_4],
    ]
    for utterance, sentences in zip(corpus.iter_utterances(), sentence_lists):
        utterance.add_meta("sentences", sentences)

    return corpus


def burr_spacy_doc_1():
    return Doc(
        vocab=spacy_en_vocab(),
        words=["Pardon", "me", ".", "Are", "you", "Aaron", "Burr", ",", "sir", "?"],
        heads=[0, 0, 0, 3, 3, 6, 3, 3, 3, 3],
        deps=[
            "ROOT",
            "dobj",
            "punct",
            "ROOT",
            "nsubj",
            "compound",
            "attr",
            "punct",
            "npadvmod",
            "punct",
        ],
        tags=["VB", "PRP", ".", "VBP", "PRP", "NNP", "NNP", ",", "NN", "."],
    )


def burr_spacy_doc_2():
    return Doc(
        vocab=spacy_en_vocab(),
        words=["That", "depends", ".", "Who", "'s", "asking", "?"],
        heads=[1, 1, 1, 5, 5, 5, 5],
        deps=["nsubj", "ROOT", "punct", "nsubj", "aux", "ROOT", "punct"],
        tags=["DT", "VBZ", ".", "WP", "VBZ", "VBG", "."],
    )


def burr_spacy_sentence_doc_1():
    return Doc(
        vocab=spacy_en_vocab(),
        words=["Pardon", "me", "."],
        heads=[0, 0, 0],
        deps=["ROOT", "dobj", "punct"],
        tags=["VB", "PRP", "."],
    )


def burr_spacy_sentence_doc_2():
    return Doc(
        vocab=spacy_en_vocab(),
        words=["Are", "you", "Aaron", "Burr", ",", "sir", "?"],
        heads=[0, 0, 3, 0, 0, 0, 0],
        deps=["ROOT", "nsubj", "compound", "attr", "punct", "npadvmod", "punct"],
        tags=["VBP", "PRP", "NNP", "NNP", ",", "NN", "."],
    )


def burr_spacy_sentence_doc_3():
    return Doc(
        vocab=spacy_en_vocab(),
        words=["That", "depends", "."],
        heads=[1, 1, 1],
        deps=["nsubj", "ROOT", "punct"],
        tags=["DT", "VBZ", "."],
    )


def burr_spacy_sentence_doc_4():
    return Doc(
        vocab=spacy_en_vocab(),
        words=["Who", "'s", "asking", "?"],
        heads=[2, 2, 2, 2],
        deps=["nsubj", "aux", "ROOT", "punct"],
        tags=["WP", "VBZ", "VBG", "."],
    )


def reload_corpus_in_db_mode(corpus):
    corpus_id = "_" + uuid4().hex
    try:
        corpus.dump(corpus_id, base_path=".")
        db_corpus = Corpus(corpus_id, storage_type="db")
        return db_corpus
    finally:
        if os.path.exists(corpus_id):
            shutil.rmtree(corpus_id)
