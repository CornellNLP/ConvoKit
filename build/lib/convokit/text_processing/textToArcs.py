from .textProcessor import TextProcessor


def _use_text(tok, sent):
    return tok["tok"].isalpha() or tok["tok"][1:].isalpha()


class TextToArcs(TextProcessor):
    """
    Transformer that outputs a collection of arcs in the dependency parses of each sentence of an utterance. The returned collection is a list where each element corresponds to a sentence in the utterance. Each sentence is represented in terms of its arcs, in a space-separated string.

    Each arc, in turn, can be read as follows:

            * `x_y` means that `x` is the parent and `y` is the child token (e.g., `agree_does` = `agree --> does`)
            * `x_*` means that `x` is a token with at least one descendant, which we do not resolve (this is analogous to bigrams backing off to unigrams)
            * `x>y` means that `x` and `y` are the first two tokens in the sentence
            * `x>*` means that `x` is the first token in the sentence.

    :param output_field: name of attribute to write arcs to.
    :param input_field: name of field to use as input. defaults to 'parsed', which stores dependency parses as returned by the TextParser transformer; otherwise expects similarly-formatted input.
    :param use_start: whether to also return the first and first two tokens of the sentence. defaults to `True`.
    :param root_only: whether to return only the arcs from the root of the dependency parse. defaults to `False`.
    :param follow_deps: if root_only is set to `True`, will nonetheless examine subtrees coming out of a dependency listed in follow_deps; by default will follow 'conj' dependencies (hence examining the parts of a sentence following conjunctions like "and").
    :param filter_fn: a boolean function determining which tokens to use. arcs will only be included if filter_fn returns True for all tokens in the arc. the function is of signature filter_fn(token, sent) where tokens and sents are formatted according to the output of TextParser. by default, will use tokens which only contain alphabet letters, or only contain letters after the first character (allowing for contractions like you 're): i.e.:  `tok['tok'].isalpha() or tok['tok'][1:].isalpha()`.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`. parses will only be computed for utterances where `input_filter` returns `True`. By default, will always return `True`, meaning that arcs will be computed for all utterances.
    :param verbosity: frequency of status messages.
    """

    def __init__(
        self,
        output_field,
        input_field="parsed",
        use_start=True,
        root_only=False,
        follow_deps=("conj",),
        filter_fn=_use_text,
        input_filter=lambda utt, aux: True,
        verbosity=0,
    ):
        aux_input = {
            "root_only": root_only,
            "use_start": use_start,
            "follow_deps": follow_deps,
            "filter_fn": filter_fn,
        }
        TextProcessor.__init__(
            self,
            proc_fn=self._get_arcs_per_message_wrapper,
            output_field=output_field,
            input_field=input_field,
            aux_input=aux_input,
            input_filter=input_filter,
            verbosity=verbosity,
        )

    def _get_arcs_per_message_wrapper(self, text_entry, aux_input={}):
        return get_arcs_per_message(
            text_entry,
            aux_input["use_start"],
            aux_input["root_only"],
            aux_input["follow_deps"],
            aux_input["filter_fn"],
        )


def _get_arcs_at_root(
    root, sent, use_start=True, root_only=False, follow_deps=("conj",), filter_fn=_use_text
):
    arcs = set()
    if not filter_fn(root, sent):
        return arcs
    arcs.add(root["tok"].lower() + "_*")

    next_elems = []
    for kid_idx in root["dn"]:
        kid = sent["toks"][kid_idx]
        if kid["dep"] in ["cc"]:
            continue

        if filter_fn(kid, sent):
            if (kid["dep"] not in follow_deps) and (root["tok"].lower() != kid["tok"].lower()):
                arcs.add(root["tok"].lower() + "_" + kid["tok"].lower())
            if (not root_only) or (kid["dep"] in follow_deps):
                next_elems.append(kid)
    if use_start:
        first_elem = sent["toks"][0]
        if filter_fn(first_elem, sent):
            arcs.add(first_elem["tok"].lower() + ">*")
            if (1 not in first_elem["dn"]) and (len(sent["toks"]) >= 2):
                second_elem = sent["toks"][1]
                if 0 not in second_elem["dn"]:
                    if filter_fn(second_elem, sent) and (
                        first_elem["tok"].lower() != second_elem["tok"].lower()
                    ):
                        arcs.add(first_elem["tok"].lower() + ">" + second_elem["tok"].lower())
    for next_elem in next_elems:
        arcs.update(
            _get_arcs_at_root(
                next_elem,
                sent,
                use_start=False,
                root_only=root_only,
                follow_deps=follow_deps,
                filter_fn=filter_fn,
            )
        )
    return arcs


def get_arcs_per_message(
    message, use_start=True, root_only=False, follow_deps=("conj",), filter_fn=_use_text
):
    """
    Stand-alone function that returns the arcs of parsed text.

    :param message: parse to extract arcs from
    :param use_start: whether to also return the first and first two tokens of the sentence. defaults to `True`.
    :param root_only: whether to return only the arcs from the root of the dependency parse. defaults to `False`.
    :param follow_deps: if root_only is set to `True`, will nonetheless examine subtrees coming out of a dependency listed in follow_deps; by default will follow 'conj' dependencies (hence examining the parts of a sentence following conjunctions like "and").
    :param filter_fn: a boolean function determining which tokens to use. arcs will only be included if filter_fn returns True for all tokens in the arc. the function is of signature filter_fn(token, sent) where tokens and sents are formatted according to the output of TextParser. by default, will use tokens which only contain alphabet letters, or only contain letters after the first character (allowing for contractions like you 're): i.e.:  `tok['tok'].isalpha() or tok['tok'][1:].isalpha()`.
    :return: a list where each element corresponds to a sentence in the input message. Each sentence is represented in terms of its arcs, in a space-separated string.
    """

    return [
        " ".join(
            sorted(
                _get_arcs_at_root(
                    sent["toks"][sent["rt"]],
                    sent,
                    use_start=use_start,
                    root_only=root_only,
                    follow_deps=follow_deps,
                    filter_fn=filter_fn,
                )
            )
        )
        for sent in message
    ]
