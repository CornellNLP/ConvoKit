from __future__ import unicode_literals
import os
import pickle
import string
import nltk
import spacy
from itertools import chain
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
import json
import re

# local import
from .politeness_strategies import get_politeness_strategy_features

# Will need access to local dir
# for support files
LOCAL_DIR = os.path.split(__file__)[0]


def get_unigrams_and_bigrams(document):
    """
    Grabs unigrams and bigrams from document
    sentences. NLTK does the work.
    """
    # Get unigram list per sentence:
    unigram_lists = [
        [y for y in t] for t in map(lambda x: nltk.word_tokenize(x), document["sentences"])
    ]
    # Generate bigrams from all sentences:
    bigrams = [
        tuple([y for y in t]) for l in map(lambda x: nltk.bigrams(x), unigram_lists) for t in l
    ]
    # Chain unigram lists
    unigrams = [x for l in unigram_lists for x in l]
    return unigrams, bigrams


class PolitenessFeatureVectorizer:
    """
    Returns document features based on-
        - unigrams and bigrams
        - politeness strategies
            (inspired by B&L, modeled using dependency parses)
    """

    UNIGRAMS_FILENAME = os.path.join(LOCAL_DIR, "featunigrams.p")
    BIGRAMS_FILENAME = os.path.join(LOCAL_DIR, "featbigrams.p")

    def __init__(self):
        """
        Load pickled lists of unigram and bigram features
        These lists can be generated using the training set
        and PolitenessFeatureVectorizer.generate_bow_features
        """
        self.unigrams = pickle.load(open(self.UNIGRAMS_FILENAME, "rb"))
        self.bigrams = pickle.load(open(self.BIGRAMS_FILENAME, "rb"))

    def features(self, document):
        """
        document must be a dict of the following format--
            {
                'text': "text str",
            }
        """
        feature_dict = {}
        # Add unigram, bigram features:
        feature_dict.update(self._get_term_features(document))
        # Add politeness strategy features:
        feature_dict.update(get_politeness_strategy_features(document))
        return feature_dict

    def _get_term_features(self, document):
        # One binary feature per ngram in
        # in self.unigrams and self.bigrams
        unigrams, bigrams = document["unigrams"], document["bigrams"]
        # Add unigrams to document for later use
        unigrams, bigrams = set(unigrams), set(bigrams)
        f = {}
        f.update(
            dict(map(lambda x: ("UNIGRAM_" + str(x), 1 if x in unigrams else 0), self.unigrams))
        )
        f.update(dict(map(lambda x: ("BIGRAM_" + str(x), 1 if x in bigrams else 0), self.bigrams)))
        return f

    @staticmethod
    def preprocess(documents):
        nlp = spacy.load("en_core_web_sm")

        for document in documents:
            document["sentences"] = nltk.sent_tokenize(document["text"])
            document["parses"] = []

            for s in document["sentences"]:
                # Spacy inclues punctuation in dependency parsing, which would lead to errors in feature extraction
                bak = s
                s = ""
                for x in bak:
                    if x in string.punctuation:
                        s += " "
                    else:
                        s += x
                s = " ".join(s.split())
                doc = nlp(s)  # unicode(s, "utf-8"))
                cur = []
                for sent in doc.sents:
                    pos = sent.start
                    for tok in sent:
                        ele = "%s(%s-%d, %s-%d)" % (
                            tok.dep_,
                            tok.head.text,
                            tok.head.i + 1 - pos,
                            tok.text,
                            tok.i + 1 - pos,
                        )
                        cur.append(ele)
                    document["parses"].append(cur)
            document["unigrams"], document["bigrams"] = get_unigrams_and_bigrams(document)
        return documents

    @staticmethod
    def generate_bow_features(documents, min_unigram_count=20, min_bigram_count=20):
        """
        Given a list of documents, compute and store list of unigrams and bigrams
        with a frequency > min_unigram_count and min_bigram_count, respectively.
        This method must be called prior to the first vectorizer instantiation.

        documents -

            each document must be a dict
            {
                'text': 'text'
            }
        """
        unigram_counts, bigram_counts = defaultdict(int), defaultdict(int)
        # Count unigrams and bigrams:
        for d in documents:
            unigrams = set(d["unigrams"])
            bigrams = set(d["bigrams"])
            # Count
            for w in unigrams:
                unigram_counts[w] += 1
            for w in bigrams:
                bigram_counts[w] += 1
        # Keep only ngrams that pass frequency threshold:
        unigram_features = list(
            filter(lambda x: unigram_counts[x] > min_unigram_count, unigram_counts.keys())
        )
        bigram_features = list(
            filter(lambda x: bigram_counts[x] > min_bigram_count, bigram_counts.keys())
        )
        # Save results:
        pickle.dump(unigram_features, open(PolitenessFeatureVectorizer.UNIGRAMS_FILENAME, "wb"))
        pickle.dump(bigram_features, open(PolitenessFeatureVectorizer.BIGRAMS_FILENAME, "wb"))


def alphas(s):
    bak = s
    s = ""
    for x in bak:
        if x.isalpha():
            s += x
    return s


if __name__ == "__main__":
    """
    Extract features from test documents
    """

    from test_documents import TEST_DOCUMENTS

    vectorizer = PolitenessFeatureVectorizer()
    documents = TEST_DOCUMENTS
    documents = PolitenessFeatureVectorizer.preprocess(documents)

    for doc in documents:
        f = vectorizer.features(doc)

        # Print summary of features that are present
        print("\n====================")
        print("Text: ", doc["text"])
        print(
            "\tUnigrams, Bigrams: %d"
            % len(filter(lambda x: f[x] > 0 and ("UNIGRAM_" in x or "BIGRAM_" in x), f.iterkeys()))
        )
        print(
            "\tPoliteness Strategies: \n\t\t%s"
            % "\n\t\t".join(filter(lambda x: f[x] > 0 and "feature_politeness_" in x, f.iterkeys()))
        )
        print("\n")
