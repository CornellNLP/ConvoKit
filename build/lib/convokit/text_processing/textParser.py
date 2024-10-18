import nltk
import spacy
import sys

import warnings
from spacy.pipeline import Sentencizer


from .textProcessor import TextProcessor


class TextParser(TextProcessor):
    """
    Transformer that dependency-parses each Utterance in a Corpus. This parsing step is a prerequisite for some of the models included in ConvoKit.

    By default, will perform the following:

            * tokenize words and sentences
            * POS-tags words
            * dependency-parses sentences

    However, also supports only tokenizing or only tokenizing-and-tagging. These are performed using SpaCy and nltk's sentence tokenizer (since SpaCy requires dependency parses in order to tokenize sentences).

    Parses are stored as json-serializable objects, consisting of a list of parses of each sentence, where each sentence-level parse is a dict containing:

            * toks: a list of tokens in the sentence.
            * rt: the index of the root of the dependency parse, in the list of tokens.

    Each token, in turn, is a dict containing:

            * tok: the text
            * tag: the POS tag (if tagging is on)
            * dep: the dependency between that token and its parent ('ROOT' if the token is the root). available if parsing is on.
            * up: the index of the parent of the token in the sentence. does not exist for root tokens.
            * dn: the indices of the children of the token in the sentence

    Note that in principle, this data structure is readily extensible -- arbitrary fields could be added to sentences and tokens (e.g., to support NER).

    :param output_field: name of attribute to write parse to, defaults to 'parsed'.
    :param input_field: name of the field to use as input. the field must point to a string, and defaults to utterance.text.
    :param mode: by default, is set to "parse", which indicates that the entire parsing pipeline is to be run. if set to "tag", only tokenizing and tagging will be run; if set to "tokenize", only tokenizing will be run.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`. parses will only be computed for utterances where `input_filter` returns `True`. By default, will always return `True`, meaning that parses will be computed for all utterances.
    :param spacy_nlp: if provided, will use this SpaCy object to do parsing; otherwise will initialize an object via `load('en')`.
    :param sent_tokenizer: if provided, will use this sentence tokenizer; otherwise will initialize nltk's sentence tokenizer.
    :param verbosity: frequency of status messages.
    """

    def __init__(
        self,
        output_field="parsed",
        input_field=None,
        mode="parse",
        input_filter=lambda utt, aux: True,
        spacy_nlp=None,
        sent_tokenizer=None,
        verbosity=0,
    ):
        self.mode = mode
        aux_input = {"mode": mode}

        if spacy_nlp is None:
            try:
                if mode == "parse":
                    aux_input["spacy_nlp"] = spacy.load("en_core_web_sm", disable=["ner"])
                elif mode == "tag":
                    aux_input["spacy_nlp"] = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                elif mode == "tokenize":
                    # also disable lemmatizer otherwise it triggers warning W108
                    aux_input["spacy_nlp"] = spacy.load(
                        "en_core_web_sm", disable=["ner", "parser", "tagger", "lemmatizer"]
                    )

            except OSError:
                print(
                    "Convokit requires a SpaCy model to be installed. Run `python -m spacy download MODEL_NAME` and retry."
                )
                sys.exit()
        else:
            aux_input["spacy_nlp"] = spacy_nlp

        if mode in ("tag", "tokenize"):
            if sent_tokenizer is None:
                # Use NLTK only when an English model is loaded
                if aux_input["spacy_nlp"].lang == "en":
                    try:
                        aux_input["sent_tokenizer"] = nltk.data.load(
                            "tokenizers/punkt/english.pickle"
                        )
                    except OSError:
                        print(
                            "Convokit requires nltk data to be downloaded. Run `python -m nltk.downloader all` and retry."
                        )
                        sys.exit()

                # For other languages, use spaCy's rule-based sentencizer
                # first check is for when the user provides a spacy_nlp with parser (despite the mode)
                elif (
                    "parser" not in aux_input["spacy_nlp"].pipe_names
                    and "sentencizer" not in aux_input["spacy_nlp"].pipe_names
                ):
                    aux_input["spacy_nlp"].add_pipe("sentencizer", first=True)

            else:
                aux_input["sent_tokenizer"] = sent_tokenizer

        TextProcessor.__init__(
            self,
            proc_fn=self._process_text_wrapper,
            output_field=output_field,
            input_field=input_field,
            input_filter=input_filter,
            aux_input=aux_input,
            verbosity=verbosity,
        )

    def _process_text_wrapper(self, text, aux_input={}):
        return process_text(
            text,
            aux_input.get("mode", "parse"),
            aux_input.get("sent_tokenizer", None),
            aux_input.get("spacy_nlp", None),
        )


# these could in principle live in a separate text_utils.py file.
def _process_token(token_obj, mode="parse", offset=0):
    if mode == "tokenize":
        token_info = {"tok": token_obj.text}
    else:
        token_info = {"tok": token_obj.text, "tag": token_obj.tag_}
    if mode == "parse":
        token_info["dep"] = token_obj.dep_
        if len([s for s in token_obj.ancestors]) > 0:
            token_info["up"] = next(token_obj.ancestors).i - offset
        token_info["dn"] = [x.i - offset for x in token_obj.children]
    return token_info


def _process_sentence(sent_obj, mode="parse", offset=0):
    tokens = []
    for token_obj in sent_obj:
        tokens.append(_process_token(token_obj, mode, offset))
    if mode == "parse":
        return {"rt": sent_obj.root.i - offset, "toks": tokens}
    else:
        return {"toks": tokens}


def process_text(text, mode="parse", sent_tokenizer=None, spacy_nlp=None):
    """
    Stand-alone function that computes the dependency parse of a string.

    :param text: string to parse
    :param mode: 'parse', 'tag', or 'tokenize'
    :param sent_tokenizer: if provided, use this sentence tokenizer
    :param spacy_nlp: if provided, use this spacy object
    :return: the parse, in json-serializable form.
    """
    if spacy_nlp is None:
        raise ValueError("spacy object required")

        # if sent_tokenizer is not provided
        # use spacy's rule-based sentencizer (only for these two modes)
    if mode in ("tag", "tokenize"):
        if sent_tokenizer is None:
            warnings.warn(
                "Sentence tokenizer is not provided. Spacy's rule-based sentencizer will be used."
            )

            # add sentencizing to spacy's pipeline
            if "parser" not in spacy_nlp.pipe_names and "sentencizer" not in spacy_nlp.pipe_names:
                spacy_nlp.add_pipe("sentencizer", first=True)

    if mode == "parse" or sent_tokenizer is None:
        sents = spacy_nlp(text.strip()).sents
    else:
        sents = [spacy_nlp(x) for x in (sent_tokenizer.tokenize(text.strip()))]

    sentences = []
    offset = 0
    for sent in sents:
        curr_sent = _process_sentence(sent, mode, offset)
        sentences.append(curr_sent)
        offset += len(curr_sent["toks"])

    return sentences
