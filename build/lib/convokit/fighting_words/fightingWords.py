import warnings
from collections import defaultdict
from typing import List, Callable, Tuple

import numpy as np
import pandas as pd
from cleantext import clean
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer as CV

from convokit import Transformer
from convokit.model import Corpus, CorpusComponent

clean_str = lambda s: clean(
    s,
    fix_unicode=True,  # fix various unicode errors
    to_ascii=True,  # transliterate to closest ASCII representation
    lower=True,  # lowercase text
    no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
    no_urls=True,  # replace all URLs with a special token
    no_emails=True,  # replace all email addresses with a special token
    no_phone_numbers=True,  # replace all phone numbers with a special token
    no_numbers=False,  # replace all numbers with a special token
    no_digits=False,  # replace all digits with a special token
    no_currency_symbols=True,  # replace all currency symbols with a special token
    no_punct=False,  # fully remove punctuation
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en",
)


class FightingWords(Transformer):
    """
    Based on Monroe et al.'s "Fightin’ Words: Lexical Feature Selection and Evaluation for Identifying the Content of
    Political Conflict"

    Implementation adapted from Jack Hessel's https://github.com/jmhessel/FightingWords

    Identifies the fighting words of two groups of corpus components (e.g. two groups of utterances),
    which we define as the groups: 'class1' and 'class2'

    Runs on the Corpus's Speakers, Utterances, or Conversations (as specified by `obj_type`).
    By default, the text used for the different object types:

    - For utterances, this would be the utterance text.
    - For conversations, this would be joined texts of all the utterances in the conversation
    - For speakers, this would be the joined texts of all the utterances by the speaker

    Other custom text configurations can be configured using the `text_func` argument

    :param obj_type: 'utterance', 'conversation', or 'speaker'
    :param text_func: function for getting text from the Corpus component object. By default, this is configured
        based on the `obj_type`.
    :param cv: optional CountVectorizer. default: an sklearn CV with min_df=10, max_df=.5, and ngram_range=(1,3)
        with max 15000 features
    :param ngram_range: range of ngrams to use if using default cv
    :param prior: either a float describing a uniform prior, or a vector describing a prior over vocabulary items.
        If using a predefined vocabulary, make sure to specify that when you make your CountVectorizer object.
    :param class1_attribute_name: metadata attribute name to store class1 ngrams under during the `transform()` step.
        Default is 'fighting_words_class1'.
    :param class2_attribute_name: metadata attribute name to store class2 ngrams under during the `transform()` step.
        Default is 'fighting_words_class2'.

    :ivar cv: modifiable countvectorizer

    """

    def __init__(
        self,
        obj_type="utterance",
        text_func=None,
        cv=None,
        ngram_range=None,
        prior=0.1,
        class1_attribute_name="fighting_words_class1",
        class2_attribute_name="fighting_words_class2",
    ):
        assert obj_type in ["speaker", "utterance", "conversation"]
        self.obj_type = obj_type

        if text_func is None:
            if obj_type == "utterance":
                self.text_func = lambda utt: FightingWords.clean_text(utt.text)
            elif obj_type == "conversation":
                self.text_func = lambda convo: FightingWords.clean_text(
                    " ".join([utt.text for utt in convo.iter_utterances()])
                )
            else:
                self.text_func = lambda spkr: FightingWords.clean_text(
                    " ".join([utt.text for utt in spkr.iter_utterances()])
                )
        else:
            self.text_func = text_func

        self.ngram_range = ngram_range
        self.prior = prior
        self.cv = cv
        self.ngram_zscores = None
        self._count_matrix = None
        if self.cv is None and type(self.prior) is not float:
            raise ValueError(
                "If using a non-uniform prior, you must pass a count vectorizer with "
                "the vocabulary parameter set."
            )
        if self.cv is None:
            print("Initializing default CountVectorizer", end=" ")
            if self.ngram_range is None:
                self.ngram_range = (1, 3)
            print("with ngram_range {}...".format(self.ngram_range), end=" ")
            self.cv = CV(
                decode_error="ignore",
                min_df=10,
                max_df=0.5,
                ngram_range=self.ngram_range,
                binary=False,
                max_features=15000,
            )
            print("Done.")

        self.class1_attribute_name = class1_attribute_name
        self.class2_attribute_name = class2_attribute_name

    @staticmethod
    def clean_text(in_string):
        """
        Cleans the text using Python clean-text package: fixes unicode, transliterates all characters to closest ASCII, lowercases text, removes line breaks and punctuation, replaces (urls, emails, phone numbers, numbers, currency) with corresponding <TOKEN>

        :param in_string: input string
        :return: cleaned string
        """
        return clean_str(in_string)

    def _bayes_compare_language(self, class1: List[CorpusComponent], class2: List[CorpusComponent]):
        """
        Arguments:
        - class1, class2; a list of strings from each language sample

        Returns:
        - A dict of length |Vocab| with (n-gram, zscore) pairs.
        """

        class1 = [self.text_func(obj) for obj in class1]
        class2 = [self.text_func(obj) for obj in class2]

        counts_mat = self.cv.fit_transform(class1 + class2).toarray()
        # Now sum over languages...
        vocab_size = len(self.cv.vocabulary_)
        print("Vocab size is {}".format(vocab_size))
        if type(self.prior) is float:
            priors = np.array([self.prior for _ in range(vocab_size)])
        else:
            priors = self.prior
        z_scores = np.empty(priors.shape[0])
        count_matrix = np.empty([2, vocab_size], dtype=np.float32)
        count_matrix[0, :] = np.sum(counts_mat[: len(class1), :], axis=0)
        count_matrix[1, :] = np.sum(counts_mat[len(class1) :, :], axis=0)
        self._count_matrix = count_matrix
        a0 = np.sum(priors)
        n1 = 1.0 * np.sum(count_matrix[0, :])
        n2 = 1.0 * np.sum(count_matrix[1, :])
        print("Comparing language...")
        for i in range(vocab_size):
            # compute delta
            term1 = np.log(
                (count_matrix[0, i] + priors[i]) / (n1 + a0 - count_matrix[0, i] - priors[i])
            )
            term2 = np.log(
                (count_matrix[1, i] + priors[i]) / (n2 + a0 - count_matrix[1, i] - priors[i])
            )
            delta = term1 - term2
            # compute variance on delta
            var = 1.0 / (count_matrix[0, i] + priors[i]) + 1.0 / (count_matrix[1, i] + priors[i])
            # store final score
            z_scores[i] = delta / np.sqrt(var)
        index_to_term = {v: k for k, v in self.cv.vocabulary_.items()}
        sorted_indices = np.argsort(z_scores)
        return {index_to_term[i]: z_scores[i] for i in sorted_indices}

    def fit(
        self,
        corpus: Corpus,
        class1_func: Callable[[CorpusComponent], bool],
        class2_func: Callable[[CorpusComponent], bool],
        y=None,
        selector: Callable[[CorpusComponent], bool] = lambda utt: True,
    ):
        """
        Learn the fighting words from a corpus, with an optional selector that selects for corpus components prior to
            grouping the corpus components into `class1` / `class2`.

        A warning will be printed if there are components that appear in both `class1` and `class2`, as FightingWords
            is typically used for disjoint sets of texts.

        :param corpus: target Corpus
        :param class1_func: selector function for identifying corpus components that belong to class 1
        :param class2_func: selector function for identifying corpus components that belong to class 2
        :param selector: a (lambda) function that takes a CorpusComponent and returns True/False; this selects for
            Corpus components that should be considered in this fitting step
        :return: fitted FightingWords Transformer

        """
        class1, class2 = [], []
        for obj in corpus.iter_objs(self.obj_type, selector):
            if class1_func(obj):
                class1.append(obj)
            if class2_func(obj):
                class2.append(obj)

        if len(class1) == 0:
            raise ValueError("class1_func returned 0 valid corpus components.")
        if len(class2) == 0:
            raise ValueError("class2_func returned 0 valid corpus components.")

        overlaps_found = set([obj.id for obj in class1]) & set([obj.id for obj in class2])
        if len(overlaps_found):
            warnings.warn(
                "There are components that appear in both classes. "
                "Note that FightingWords is typically used to compare two disjoint sets of texts."
            )

        print(
            f"class1_func returned {len(class1)} valid corpus components. "
            f"class2_func returned {len(class2)} valid corpus components."
        )

        self.ngram_zscores = self._bayes_compare_language(class1, class2)
        print("ngram zscores computed.")
        return self

    def get_ngram_zscores(self, class1_name="class1", class2_name="class2"):
        """
        Get a DataFrame of ngrams and their corresponding zscores and class labels.

        :param class1_name: readable name for objects in class1
        :param class2_name: readable name for objects in class2
        :return: a DataFrame of ngrams with zscores and classes, indexed by the ngrams
        """

        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")
        df = pd.DataFrame(list(self.ngram_zscores.items()), columns=["ngram", "z-score"]).set_index(
            "ngram"
        )
        df["class"] = (df["z-score"] >= 0).apply(lambda x: [class2_name, class1_name][int(x)])
        return df

    def get_top_k_ngrams(self, top_k: int = 10) -> Tuple[List[str], List[str]]:
        """
        Returns the (ordered) top k ngrams for both classes.

        :param top_k: by default, k = 10
        :return: two ordered lists of ngrams (with descending z-score): first list is for class 1,
            second list is for class 2.
        """
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")

        ngram_zscores_list = list(
            zip(self.get_ngram_zscores().index, self.get_ngram_zscores()["z-score"])
        )
        top_k_class1 = list(reversed([x[0] for x in ngram_zscores_list[-top_k:]]))
        top_k_class2 = [x[0] for x in ngram_zscores_list[:top_k]]
        return top_k_class1, top_k_class2

    def get_ngrams_past_threshold(self, threshold: float = 1.0) -> Tuple[List[str], List[str]]:
        """
        Returns the (ordered) ngrams that have absolute z-scores that exceed a specified threshold, for both classes

        :param threshold: by default, threshold z-score = 1
        :return: two ordered lists of ngrams (with descending z-score):
                first list is for class 1, second list is for class 2
        """
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")

        class1_ngrams = []
        class2_ngrams = []
        for ngram, zscore in self.ngram_zscores.items():
            if zscore > threshold:
                class1_ngrams.append(ngram)
            elif zscore < -threshold:
                class2_ngrams.append(ngram)
        return class1_ngrams[::-1], class2_ngrams

    def transform(
        self,
        corpus: Corpus,
        selector: Callable[[CorpusComponent], bool] = lambda x: True,
        config=None,
    ) -> Corpus:
        """
        Annotates the corpus component objects with the lists of fighting words that the object contains.

        The relevant fighting words to use are specified by the config parameter. By default, the annotation method
        is to annotate the corpus components with the top 10 fighting words of each class.

        Lists are stored under the metadata attributes defined when initializing the FightingWords Transformer.

        :param corpus: corpus to annotate
        :param selector: a (lambda) function that takes a CorpusComponent and returns True/False; this selects for
            corpus components that should be annotated with the fighting words
        :param config: a dictionary of configuration parameters for setting which fighting words are significant enough
            to annotate. The dictionary should hold the keys: annot_method ('top_k' or 'threshold'), and either
            'threshold' (a float for the min absolute z-score to be considered significant) or 'top_k' (an int to set
            the value of k). By default, config is {'annot_method': 'top_k', 'top_k': 10}.

        :return: annotated corpus
        """
        config = {"top_k": 10, "annot_method": "top_k"} if config is None else config

        class1_ngrams, class2_ngrams = (
            self.get_top_k_ngrams(top_k=config["top_k"])
            if config["annot_method"] == "top_k"
            else self.get_ngrams_past_threshold(threshold=config["threshold"])
        )

        for obj in corpus.iter_objs(
            self.obj_type
        ):  # improve the efficiency of this; tricky because ngrams #TODO
            if selector(obj):
                obj_text = self.text_func(obj)
                obj.meta[self.class1_attribute_name] = [
                    ngram for ngram in class1_ngrams if ngram in obj_text
                ]
                obj.meta[self.class2_attribute_name] = [
                    ngram for ngram in class2_ngrams if ngram in obj_text
                ]
            else:
                obj.meta[self.class1_attribute_name] = None
                obj.meta[self.class2_attribute_name] = None

        return corpus

    def get_zscore(self, ngram):
        """
        Get z-score of a given ngram.

        :param ngram: ngram of interest
        :return: z-score value, None if ngram not in vocabulary
        """
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")
        return self.ngram_zscores.get(ngram, None)

    def get_class(self, ngram):
        """
        Get the class that ngram more belongs to.

        :param ngram: ngram of interest
        :return: "class1" if the ngram has non-negative z-score, "class2" if ngram has positive z-score, None if
            ngram not in vocabulary
        """
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")
        zscore = self.ngram_zscores.get(ngram, None)
        if zscore is None:
            return zscore
        if zscore >= 0:
            return "class1"
        else:
            return "class2"

    def summarize(
        self, corpus: Corpus, plot: bool = False, class1_name="class1", class2_name="class2"
    ):
        """
        Returns a DataFrame of ngram with zscores and classes, and optionally plots the fighting words distribution.
        FightingWords Transformer must be fitted prior to running this.

        :param corpus: corpus to learn fighting words from if not already fitted
        :param plot: if True, generates a plot for the fighting words distribution
        :param class1_name: descriptive name for class1 corpus component objects
        :param class2_name: descriptive name for class2 corpus component objects
        :return: DataFrame of ngrams with zscores and classes, indexed by the ngrams (plot is optionally generated)
        """
        if plot:
            self.plot_fighting_words(class1_name=class1_name, class2_name=class2_name)
        return self.get_ngram_zscores(class1_name=class1_name, class2_name=class2_name)

    def plot_fighting_words(
        self, max_label_size=15, class1_name="class1", class2_name="class2", config=None
    ):
        """
        Plots the distribution of fighting words.

        Adapted from Xanda Schofield's https://gist.github.com/xandaschofield/3c4070b2f232b185ce6a09e47b4e7473

        Specifically, the weighted log-odds ratio is plotted against frequency of word within topic.

        Only the most significant ngrams will have text labels. The most significant ngrams are specified by the config
        parameter. By default, the annotation method is to annotate the corpus components with the top 10 fighting words
        of each class.

        :param max_label_size: For the text labels, set the largest possible size for any text label
            (the rest will be scaled accordingly)
        :param class1_name: descriptive name for class1 corpus component objects
        :param class2_name: descriptive name for class2 corpus component objects
        :param config: a dictionary of configuration parameters for setting which fighting words are significant enough
            to annotate. The dictionary should hold the keys: annot_method ('top_k' or 'threshold'), and either
            'threshold' (a float for the min absolute z-score to be considered significant) or 'top_k' (an int to set
            the value of k). By default, config is {'annot_method': 'top_k', 'top_k': 10}.
        :return: None (plot is generated)
        """
        config = {"top_k": 10, "annot_method": "top_k"} if config is None else config

        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")

        x_vals = self._count_matrix.sum(axis=0)
        y_vals = list(self.get_ngram_zscores()["z-score"])
        sizes = abs(np.array(y_vals))
        scale_factor = max_label_size / max(sizes)
        sizes *= scale_factor
        neg_color, pos_color, insig_color = ("orange", "purple", "grey")
        annots = []

        class1_sig_ngrams, class2_sig_ngrams = (
            self.get_top_k_ngrams(top_k=config["top_k"])
            if config["annot_method"] == "top_k"
            else self.get_ngrams_past_threshold(threshold=config["threshold"])
        )
        class1_sig_ngrams = set(class1_sig_ngrams)
        class2_sig_ngrams = set(class2_sig_ngrams)

        terms = list(self.get_ngram_zscores().index)

        class1, class2, class_insig = defaultdict(list), defaultdict(list), defaultdict(list)

        for i in range(len(terms)):
            if terms[i] in class1_sig_ngrams:
                class1["x"].append(x_vals[i])
                class1["y"].append(y_vals[i])
                class1["size"].append(sizes[i])
                annots.append(terms[i])
            elif terms[i] in class2_sig_ngrams:
                class2["x"].append(x_vals[i])
                class2["y"].append(y_vals[i])
                class2["size"].append(sizes[i])
                annots.append(terms[i])
            else:
                class_insig["x"].append(x_vals[i])
                class_insig["y"].append(y_vals[i])
                class_insig["size"].append(sizes[i])
                annots.append(None)

        fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

        ax.scatter(class1["x"], class1["y"], c=pos_color, s=class1["size"], label=class1_name)
        ax.scatter(class2["x"], class2["y"], c=neg_color, s=class2["size"], label=class2_name)
        ax.scatter(class_insig["x"], class_insig["y"], c=insig_color, s=class_insig["size"])

        for i, annot in enumerate(annots):
            if annot is not None:
                ax.annotate(annot, (x_vals[i], y_vals[i]))

        ax.legend()
        ax.set_xscale("log")
        ax.set_title("Weighted log-odds ratio vs. Frequency of word within class")
        plt.show()

    def get_model(self):
        """
        Get the FightingWords CountVectorizer model
        """
        return self.cv

    def set_model(self, cv):
        """
        Set the FightingWords CountVectorizer model
        """
        self.cv = cv
