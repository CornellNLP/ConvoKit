from convokit.text_processing import TextProcessor


class QuestionSentences(TextProcessor):
    """
    Transformer that, given a list of sentences, returns a list containing only sentences which are questions (determined, as a rough heuristic, by whether they end in question marks). Returns an empty list if there are no questions.

    :param output_field: name of attribute to output to.
    :param input_field: name of field to use as input. expects a list where each sentence corresponds to a sentence in filter_field.
    :param use_caps: whether to only use sentences which start in capital letters. defaults to True.
    :param filter_field: name of field to check for question marks in, defaults to the output of the TextParser transformer. the entries of input_field and filter_field should exactly correspond.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`. parses will only be computed for utterances where `input_filter` returns `True`. By default, will always return `True`, meaning that arcs will be computed for all utterances.
    :param verbosity: frequency of status messages.
    """

    def __init__(
        self,
        output_field,
        input_field,
        use_caps=True,
        filter_field="parsed",
        input_filter=None,
        verbosity=0,
    ):
        aux_input = {"input_field": input_field, "filter_field": filter_field, "use_caps": use_caps}

        TextProcessor.__init__(
            self,
            self._get_question_sentences,
            output_field=output_field,
            input_field=[input_field, filter_field],
            input_filter=input_filter,
            aux_input=aux_input,
            verbosity=verbosity,
        )

    def _get_question_sentences(self, text_entry, aux_input):
        text = text_entry[aux_input["input_field"]]
        parse = text_entry[aux_input["filter_field"]]
        sents = []
        for input_sent, filter_sent in zip(text, parse):
            if isinstance(filter_sent, dict):
                if filter_sent["toks"][-1]["tok"] != "?":
                    continue
                if aux_input["use_caps"] and not filter_sent["toks"][0]["tok"][0].isupper():
                    continue
            else:
                if "?" not in filter_sent:
                    continue
                if aux_input["use_caps"] and not filter_sent[0].isupper():
                    continue
            sents.append(input_sent)
        return sents
