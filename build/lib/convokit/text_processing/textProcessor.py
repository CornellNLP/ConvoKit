from inspect import signature

from convokit.model import Corpus, Utterance, Speaker
from convokit.transformer import Transformer


class TextProcessor(Transformer):
    """
    A base class for Transformers that perform per-utterance computations, i.e., computing  utterance-by-utterance features or representations.

    :param proc_fn: function to compute per utterance. Supports one of two function signatures: `proc_fn(input)` and `proc_fn(input, auxiliary_info)`.
    :param input_field: If set to a string, the attribute of the utterance that `proc_fn` will take as input. If set to `None`, will default to reading `utt.text`. If set to a list of attributes, `proc_fn` will expect a dict of {attribute name: attribute value}.
    :param output_field: If set to a string, the name of the attribute that the output of `proc_fn` will be written to. If set to a list, `proc_fn` will return a tuple where each entry in the tuple corresponds to a field in the list.
    :param aux_input: any auxiliary input that `proc_fn` needs (e.g., a pre-loaded model); passed in as a dict.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`. attributes will only be computed for utterances where `input_filter` returns `True`. By default, will always return `True`, meaning that attributes will be computed for all utterances.
    :param verbosity: frequency at which to print status messages when computing attributes.
    """

    def __init__(
        self,
        proc_fn,
        output_field,
        input_field=None,
        aux_input=None,
        input_filter=None,
        verbosity=0,
    ):
        self.proc_fn = proc_fn
        self.aux_input = aux_input if aux_input is not None else {}
        # self.input_filter = input_filter if input_filter is not None else lambda utt, aux: True
        # temporary fix to deal with aux_input argument to input_filter
        if input_filter:
            if len(signature(input_filter).parameters) == 1:
                self.input_filter = lambda utt, aux: input_filter(utt)
            else:
                self.input_filter = input_filter
        else:
            self.input_filter = lambda utt, aux: True
        self.input_field = input_field
        self.output_field = output_field
        self.verbosity = verbosity
        self.multi_outputs = isinstance(output_field, list)

    def _print_output(self, i):
        return (self.verbosity > 0) and (i > 0) and (i % self.verbosity == 0)

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Computes per-utterance attributes for each utterance in the Corpus, storing these values in the `output_field` of each utterance as specified in the constructor. For utterances which do not contain all of the `input_field` attributes as specified in the constructor, or for utterances which return `False` on `input_filter`, this call will not annotate the utterance.

        :param corpus: Corpus
        :return: the corpus
        """

        total_utts = len(list(corpus.iter_utterances()))

        for idx, utterance in enumerate(corpus.iter_utterances()):
            if self._print_output(idx):
                print("%03d/%03d utterances processed" % (idx, total_utts))
            if not self.input_filter(utterance, self.aux_input):
                continue
            if self.input_field is None:
                text_entry = utterance.text
            elif isinstance(self.input_field, str):
                text_entry = utterance.retrieve_meta(self.input_field)

            elif isinstance(self.input_field, list):
                text_entry = {field: utterance.retrieve_meta(field) for field in self.input_field}
                if sum(x is None for x in text_entry.values()) > 0:
                    text_entry = None
            if text_entry is None:
                continue
            if len(self.aux_input) == 0:
                result = self.proc_fn(text_entry)
            else:
                result = self.proc_fn(text_entry, self.aux_input)
            if self.multi_outputs:
                for res, out in zip(result, self.output_field):
                    utterance.add_meta(out, res)
            else:
                utterance.add_meta(self.output_field, result)
        if self.verbosity > 0:
            print("%03d/%03d utterances processed" % (total_utts, total_utts))
        return corpus

    def transform_utterance(self, utt, override_input_filter=False):
        """
        Computes per-utterance attributes of an individual utterance or string. For utterances which do not contain all of the `input_field` attributes as specified in the constructor, or for utterances which return `False` on `input_filter`, this call will not annotate the utterance. For strings, will convert the string to an utterance and return the utterance, annotating it if `input_field` is not set to `None` at initialization.

        :param utt: utterance or a string
        :param override_input_filter: ignore `input_filter` and compute attribute for all utterances
        :return: the utterance
        """

        if isinstance(utt, str):
            utt = Utterance(text=utt, speaker=Speaker(id="speaker"))
        if self.input_field is None:
            text_entry = utt.text
        else:
            if not override_input_filter:
                if not self.input_filter(utt, self.aux_input):
                    return utt
            if isinstance(self.input_field, str):
                text_entry = utt.retrieve_meta(self.input_field)
            elif isinstance(self.input_field, list):
                text_entry = {field: utt.retrieve_meta(field) for field in self.input_field}
                if sum(x is None for x in text_entry.values()) > 0:
                    return utt
        if text_entry is None:
            return utt
        if len(self.aux_input) == 0:
            result = self.proc_fn(text_entry)
        else:
            result = self.proc_fn(text_entry, self.aux_input)
        if self.multi_outputs:
            for res, out in zip(result, self.output_field):
                utt.add_meta(out, res)
        else:
            utt.add_meta(self.output_field, result)
        return utt
