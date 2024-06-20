import convokit
from convokit import Corpus, download, FightingWords
from convokit.transformer import Transformer
from inspect import signature
from collections import defaultdict
from itertools import permutations
from nltk.tokenize import word_tokenize
from convokit import Corpus, download
import matplotlib.pyplot as plt
import numpy as np
import random
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import language_tool_python
import os


class OpennessScoreBERT(Transformer):
    """
    A transformer to calculate openness score for all utterance

    :param obj_type: type of Corpus object to calculate: 'conversation', 'speaker', or 'utterance', default to be 'utterance'
    :param input_field: Input fields from every utterance object. Will default to reading 'utt.text'. If a string is provided, than consider metadata with field name input_field.
    :param output_field: field for writing the computed output in metadata. Will default to write to utterance metadata with name 'capitalization'.
    :param input_filter: a boolean function of signature `input_filter(utterance, aux_input)`. attributes will only be computed for utterances where `input_filter` returns `True`. By default, will always return `True`, meaning that attributes will be computed for all utterances.
    :param verbosity: frequency at which to print status messages when computing attributes.
    """

    def __init__(
        self,
        obj_type="utterance",
        output_field="openness_score",
        input_field=None,
        input_filter=None,
        model_name="bert-base-cased",
        verbosity=1000,
    ):
        if input_filter:
            if len(signature(input_filter).parameters) == 1:
                self.input_filter = lambda utt: input_filter(utt)
            else:
                self.input_filter = input_filter
        else:
            self.input_filter = lambda utt: True
        self.obj_type = obj_type
        self.input_field = input_field
        self.output_field = output_field
        self.verbosity = verbosity
        self.grammar_tool = language_tool_python.LanguageToolPublicAPI("en")
        self.answer_sample = ["Mhm", "Okay", "I see", "Yup"]
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _print_output(self, i):
        return (self.verbosity > 0) and (i > 0) and (i % self.verbosity == 0)

    def bert_score(self, question, answer):
        """
        Outputs the perplexitty score for predicting the answer, given the question

        :param question: str
        :param answer: str
        :return: perplexity
        """
        sentence = question + " " + answer
        tensor_input = self.tokenizer.encode(sentence, return_tensors="pt")
        question_tok_len = len(self.tokenizer.encode(question)) - 2
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2 - question_tok_len, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2][question_tok_len:]
        masked_input = repeat_input.masked_fill(mask == 1, self.tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != self.tokenizer.mask_token_id, -100)
        with torch.inference_mode():
            loss = self.model(masked_input, labels=labels).loss
        return np.exp(loss.item())

    def find_last_question(self, text):
        """
        Finds the last sentence that ended with a question mark

        :param text: str
        :return: text
        """
        end_sent = set([".", "?", "!"])
        last_q = text.rfind("?")
        for i in range(last_q - 1, -1, -1):
            if text[i] in end_sent:
                return text[i + 1 : last_q + 1].strip()
        return text[: last_q + 1].strip()

    def bert_opennes_score(self, question):
        scores = []
        question = self.find_last_question(question)
        question = self.grammar_tool.correct(question)

        for ans in self.answer_sample:
            ans_text = ans
            perp = self.bert_score(question, ans_text)
            scores.append(perp)
        return np.mean(scores)

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Score the given utterance on their openness and store it to the corresponding object metadata fields.

        :param corpus: Corpus
        :return: the corpus
        """
        if self.obj_type == "utterance":
            total = len(list(corpus.iter_utterances()))

            for idx, utterance in enumerate(corpus.iter_utterances()):
                if self._print_output(idx):
                    print(f"%03d/%03d {self.obj_type} processed" % (idx, total))

                if not self.input_filter(utterance):
                    continue

                if self.input_field is None:
                    text_entry = utterance.text
                elif isinstance(self.input_field, str):
                    text_entry = utterance.meta(self.input_field)
                if text_entry is None:
                    continue

                # do the catching and add to output_field
                catch = self.bert_opennes_score(text_entry)

                utterance.add_meta(self.output_field, catch)

        elif self.obj_type == "conversation":
            total = len(list(corpus.iter_conversations()))
            for idx, convo in enumerate(corpus.iter_conversations()):
                if self._print_output(idx):
                    print(f"%03d/%03d {self.obj_type} processed" % (idx, total))

                if not self.input_filter(convo):
                    continue

                if self.input_field is None:
                    utt_lst = convo.get_utterance_ids()
                    text_entry = " ".join([corpus.get_utterance(x).text for x in utt_lst])
                elif isinstance(self.input_field, str):
                    text_entry = convo.meta(self.input_field)
                if text_entry is None:
                    continue

                # do the catching and add to output_field
                catch = self.bert_opennes_score(text_entry)

                convo.add_meta(self.output_field, catch)

        elif self.obj_type == "speaker":
            total = len(list(corpus.iter_speakers()))
            for idx, sp in enumerate(corpus.iter_speakers()):
                if self._print_output(idx):
                    print(f"%03d/%03d {self.obj_type} processed" % (idx, total))

                if not self.input_filter(sp):
                    continue

                if self.input_field is None:
                    utt_lst = sp.get_utterance_ids()
                    text_entry = " ".join([corpus.get_utterance(x).text for x in utt_lst])
                elif isinstance(self.input_field, str):
                    text_entry = sp.meta(self.input_field)
                if text_entry is None:
                    continue

                # do the catching and add to output_field
                catch = self.bert_opennes_score(text_entry)

                sp.add_meta(self.output_field, catch)

        else:
            raise KeyError("obj_type must be utterance, conversation, or speaker")

        if self.verbosity > 0:
            print(f"%03d/%03d {self.obj_type} processed" % (total, total))
        return corpus
