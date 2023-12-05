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
from sentence_transformers import SentenceTransformer, util
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import DocumentSearchPipeline
import language_tool_python


class OpennessScoreSimilarity(Transformer):
    """
    A transformer that uses BERT similarity to calculate openness score

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
        self.document_store = InMemoryDocumentStore(use_bm25=True)
        self.model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device="cpu")  # SBERT model

    def fit(self, corpus: Corpus, y=None):
        """Learn context information for the given corpus."""
        self.corpus = corpus
        self._load_questions(corpus)

    def _print_output(self, i):
        return (self.verbosity > 0) and (i > 0) and (i % self.verbosity == 0)

    def generated_openness_score_similarity(self, text):
        if len(text) > 500 and len(word_tokenize(text)) > 100:
            text_token = word_tokenize(self.find_last_question(text))
            text = ""
            for token in text:
                text = text + " " + token
        prediction = self.pipe.run(query=text, params={"Retriever": {"top_k": 10}})
        answers = [prediction["documents"][i].meta["answer"] for i in range(10)]
        return self._avg_bert_sim(answers)

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
                catch = self.generated_openness_score_similarity(text_entry)

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
                catch = self.generated_openness_score_similarity(text_entry)

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
                catch = self.generated_openness_score_similarity(text_entry)

                sp.add_meta(self.output_field, catch)

        else:
            raise KeyError("obj_type must be utterance, conversation, or speaker")

        if self.verbosity > 0:
            print(f"%03d/%03d {self.obj_type} processed" % (total, total))
        return corpus

    # helper function
    def _load_questions(self, corpus):
        """"""
        docs = []
        convo_ids = corpus.get_conversation_ids()
        for idx in convo_ids:
            convo = corpus.get_conversation(idx)
            utts = convo.get_chronological_utterance_list()
            had_question = False
            before_text = ""
            for utt in utts:
                if had_question:
                    dic_transf = {
                        "content": before_text,
                        "meta": {"convo_id": idx, "answer": utt.text},
                    }
                    docs.append(dic_transf)
                    had_question = False
                if utt.meta["questions"] > 0:
                    had_question = True
                    before_text = utt.text
        self.document_store.write_documents(docs)
        self.retriever = BM25Retriever(document_store=self.document_store)
        self.pipe = DocumentSearchPipeline(retriever=self.retriever)

    def _sbert_embedd_sim(self, embedding1, embedding2):
        return float(util.cos_sim(embedding1, embedding2))

    def _avg_bert_sim(self, texts):
        embeddings = []
        for text in texts:
            embeddings.append(self.model.encode(text))

        scores = []
        for i, embedding1 in enumerate(embeddings):
            for j, embedding2 in enumerate(embeddings):
                if i >= j:
                    continue
                scores.append(self._sbert_embedd_sim(embedding1, embedding2))
        return np.mean(scores)

    def _find_last_question(text):
        end_sent = set([".", "?", "!"])
        last_q = text.rfind("?")
        for i in range(last_q - 1, -1, -1):
            if text[i] in end_sent:
                return text[i + 1 : last_q + 1].strip()
        return text[: last_q + 1].strip()
