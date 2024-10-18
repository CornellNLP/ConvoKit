from convokit.text_processing import TextProcessor

NP_LABELS = set(["nsubj", "nsubjpass", "dobj", "iobj", "pobj", "attr"])


class CensorNouns(TextProcessor):
    """
    Transformer that, given a parse (formatted as the output of a TextParser transformer) returns a parse where nouns and pronouns are replaced with "NN~". A rough heuristic for removing "content-related" tokens. This transformer also collapses constructions with Wh-determiners like What time [is it] into What [is it].

    :param output_field: name of attribute to output to.
    :param input_field: name of field to use as input. defaults to 'parsed', which stores dependency parses as returned by the TextParser transformer; otherwise expects similarly-formatted input.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`. parses will only be computed for utterances where `input_filter` returns `True`. By default, will always return `True`, meaning that arcs will be computed for all utterances.
    :param verbosity: frequency of status messages.
    """

    def __init__(self, output_field, input_field="parsed", input_filter=None, verbosity=0):
        TextProcessor.__init__(
            self,
            censor_nouns,
            output_field=output_field,
            input_field=input_field,
            input_filter=input_filter,
            verbosity=verbosity,
        )


def _is_noun_ish(tok):
    return (
        (tok["dep"] in NP_LABELS)
        or (tok["tag"].startswith("NN") or tok["tag"].startswith("PRP"))
        or (tok["tag"].endswith("DET") or tok["tag"].endswith("DT"))
    )


def _get_w_det(tok, sent):
    if tok["tag"].startswith("W"):
        return tok["tok"]
    if len(tok["dn"]) == 0:
        return False
    first_tok = sent["toks"][tok["dn"][0]]
    if first_tok["tag"].startswith("W"):
        return first_tok["tok"]
    return False


def _convert_noun(tok, sent):
    if _is_noun_ish(tok):
        has_w = _get_w_det(tok, sent)
        if has_w:
            return has_w.lower()
        else:
            return "NN~"
    return tok["tok"].lower()


def censor_nouns(text_entry):
    """
    Stand-alone function that removes nouns from parsed text.

    :param text_entry: parsed text
    :return: parse with nouns censored out.
    """

    sents = []
    for raw_sent in text_entry:
        sent = {"rt": raw_sent["rt"], "toks": []}
        for raw_tok in raw_sent["toks"]:
            tok = {k: raw_tok[k] for k in ["dep", "dn", "tag"]}
            if "up" in raw_tok:
                tok["up"] = raw_tok["up"]
            tok["tok"] = _convert_noun(raw_tok, raw_sent)
            sent["toks"].append(tok)
        sents.append(sent)
    return sents
