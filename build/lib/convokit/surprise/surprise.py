import numpy as np
from collections import defaultdict, Counter
from convokit import Transformer
from convokit.model import Corpus, CorpusComponent, Utterance
from itertools import chain
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import Callable, List, Tuple, Union


def _cross_entropy(target: List[str], context: List[str], smooth=True):
    """
    Calculates H(P,Q) = -sum_{x\in X}(P(x) * log(Q(x)))

    :param target: list of tokens that make up the target text (P)
    :param context: list of tokens that make up the context (Q)
    :param smooth: whether to use add 1 smoothing for OOV tokens

    :return: cross entropy
    """
    N_target, N_context = len(target), len(context)
    if min(N_target, N_context) == 0:
        return np.nan
    context_counts = Counter(context)
    V = len(context_counts) + 1 if smooth else 0
    k = 1 if smooth else 0
    val = 0 if smooth else 1
    return (
        sum(-np.log((context_counts.get(tok, val) + k) / (N_context + V)) for tok in target)
        / N_target
    )


def sample(tokens: List[Union[np.ndarray, List[str]]], sample_size: int, n_samples=50, p=None):
    """
    Generates random samples from a list of lists of tokens.

    :param toks: a list of lists of tokens to sample from.
    :param sample_size: the number of tokens to include in each sample.
    :param n_samples: the number of samples to take.

    :return: numpy array where each row is a sample of tokens
    """
    if not sample_size:
        assert len(tokens) == 1
        return np.tile(tokens[0], (n_samples, 1))
    tokens_list = np.array([toks for toks in tokens if len(toks) >= sample_size])
    if tokens_list.shape[0] == 0:
        return None
    rng = np.random.default_rng()
    sample_idxes = rng.integers(0, tokens_list.shape[0], size=(n_samples))
    return np.array([rng.choice(tokens_list[i], sample_size) for i in sample_idxes])


class Surprise(Transformer):
    """
    Computes how surprising a target (an utterance or group of utterances) is based on some context.
    The measure for surprise used is cross entropy. Uses fixed size samples from target and context text
    to mitigate effects of length on cross entropy.

    :param model_key_selector: function that defines how utterances should be mapped to models.
        Takes in an utterance and returns the key to use for mapping the utterance to a corresponding model.
    :param tokenize: optional function that takes in a string and returns a list of tokens in that string.
        default: nltk's word_tokenize
    :param surprise_attr_name: the name for the metadata attribute to add to objects.
        default: surprise
    :param target_sample_size: number of tokens to sample from each target (test text). If `None`, then the entire target will be used.
    :param context_sample_size: number of tokens to sample from each context (training text). If `None`, then the entire context will be used.
    :param n_samples: number of samples to take for each target-context pair.
    :param sampling_fn: function for generating samples of tokens.
    :param smooth: whether to use laplace smoothing when calculating surprise.
    """

    def __init__(
        self,
        model_key_selector: Callable[[Utterance], str],
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        surprise_attr_name="surprise",
        target_sample_size=100,
        context_sample_size=100,
        n_samples=50,
        sampling_fn: Callable[[np.ndarray, int], np.ndarray] = sample,
        smooth: bool = True,
    ):
        self.model_key_selector = model_key_selector
        self.tokenizer = tokenizer
        self.surprise_attr_name = surprise_attr_name
        self.target_sample_size = target_sample_size
        self.context_sample_size = context_sample_size
        self.n_samples = n_samples
        self.sampling_fn = sampling_fn
        self.smooth = smooth

    def fit(
        self,
        corpus: Corpus,
        text_func: Callable[[Utterance], List[str]] = None,
        selector: Callable[[Utterance], bool] = lambda utt: True,
    ):
        """
        Fits a model for each group of utterances in a corpus. The group that an
        utterance belongs to is determined by the `model_key_selector` parameter in
        the transformer's constructor.

        :param corpus: corpus to create models from.
        :param text_func: optional function to define how the text a model is trained
            on should be selected. Takes an utterance as input and returns a list of
            strings to train the model corresponding to that utterance on. The model
            corresponding to the utterance is determined by `self.model_key_selector`.
            For every utterance corresponding to the same model key, this function
            should return the same result.
            If `text_func` is `None`, a model will be trained on the text from all
            the utterances that belong to its group.
        :param selector: determines which utterances in the corpus to train models for.
        """
        self.model_groups = defaultdict(list)
        for utt in tqdm(corpus.iter_utterances(selector=selector), desc="fit1"):
            key = self.model_key_selector(utt)
            if text_func:
                if key not in self.model_groups:
                    self.model_groups[key] = text_func(utt)
            else:
                self.model_groups[key].append(utt.text)
        for key in tqdm(self.model_groups, desc="fit2"):
            if not text_func:
                self.model_groups[key] = [" ".join(self.model_groups[key])]
            self.model_groups[key] = list(map(lambda x: self.tokenizer(x), self.model_groups[key]))
        return self

    def transform(
        self,
        corpus: Corpus,
        obj_type: str,
        group_and_models: Callable[[Utterance], Tuple[str, List[str]]] = None,
        group_model_attr_key: Callable[[str, str], str] = None,
        selector: Callable[[CorpusComponent], bool] = lambda _: True,
        target_text_func: Callable[[Utterance], List[str]] = None,
    ):
        """
        Annotates `obj_type` components in a corpus with surprise scores. Should be
        called after fit().

        :param corpus: corpus to compute surprise for.
        :param obj_type: the type of corpus components to annotate. Should be either
            'utterance', 'speaker', 'conversation', or 'corpus'.
        :param group_and_models: optional function that defines how an utterance should
            be grouped to form a target text and what models (contexts) the group should
            be compared to when calculating surprise. Takes in an utterance and returns
            a tuple containing the name of the group the utterance belongs to and a
            list of models to calculate how surprising that group is against. Objects
            will be annotated with a metadata field `self.surprise_attr_name` that is
            maps a key corresponding to the `groupname` and `modelkey` to the surprise
            score for utterances in the group when compared to the model. The key used
            is defined by the `group_model_attr_key` parameter.
            If `group_and_models` is `None`, `self.model_key_selector` will be used
            to select the group that an utterance belongs to. The surprise score will
            be calculated for each group of utterances compared to the model in
            `self.models` corresponding to the group.
        :param group_model_attr_key: optional function to define what key should be used
            for a given `groupname` and `modelkey`.
            If `group_model_attr_key` is `None`, the default key used will be
            "GROUP_groupname_MODEL_modelkey" unless `groupname` and `modelkey` are equal
            in which case just "modelkey" will be used as the key.
        :param selector: function to select objects to annotate. if function returns true, object will be annotated.
        :param target_text_func: optional function to define what the target text corresponding to an utterance should be.
            takes in an utterance and returns a list of string tokens
        """
        if obj_type == "corpus":
            utt_groups = defaultdict(list)
            group_models = defaultdict(set)
            for utt in corpus.iter_utterances():
                if group_and_models:
                    group_name, models = group_and_models(utt)
                else:
                    group_name = self.model_key_selector(utt)
                    models = {group_name}
                if target_text_func:
                    if group_name not in utt_groups:
                        utt_groups[group_name] = [target_text_func(utt)]
                else:
                    utt_groups[group_name].append(self.tokenizer(utt.text))
                group_models[group_name].update(models)
            surprise_scores = {}
            for group_name in tqdm(utt_groups, desc="transform"):
                for model_key in group_models[group_name]:
                    context = self.model_groups[model_key]
                    target = list(chain(*utt_groups[group_name]))
                    surprise_scores[
                        Surprise._format_attr_key(group_name, model_key, group_model_attr_key)
                    ] = self._compute_surprise(target, context)
            corpus.add_meta(self.surprise_attr_name, surprise_scores)
        elif obj_type == "utterance":
            for utt in tqdm(corpus.iter_utterances(selector=selector), desc="transform"):
                if group_and_models:
                    group_name, models = group_and_models(utt)
                    surprise_scores = {}
                    for model_key in models:
                        context = self.model_groups[model_key]
                        target = (
                            target_text_func(utt) if target_text_func else self.tokenizer(utt.text)
                        )
                        surprise_scores[
                            Surprise._format_attr_key(group_name, model_key, group_model_attr_key)
                        ] = self._compute_surprise(target, context)
                    utt.add_meta(self.surprise_attr_name, surprise_scores)
                else:
                    group_name = self.model_key_selector(utt)
                    context = self.model_groups[group_name]
                    target = target_text_func(utt) if target_text_func else self.tokenizer(utt.text)
                    utt.add_meta(self.surprise_attr_name, self._compute_surprise(target, context))
        else:
            for obj in tqdm(corpus.iter_objs(obj_type, selector=selector), desc="transform"):
                utt_groups = defaultdict(list)
                group_models = defaultdict(set)
                for utt in obj.iter_utterances():
                    if group_and_models:
                        group_name, models = group_and_models(utt)
                    else:
                        group_name = self.model_key_selector(utt)
                        models = {group_name}
                    if target_text_func:
                        if group_name not in utt_groups:
                            utt_groups[group_name] = [target_text_func(utt)]
                    else:
                        utt_groups[group_name].append(self.tokenizer(utt.text))
                    group_models[group_name].update(models)
                surprise_scores = {}
                for group_name in utt_groups:
                    for model_key in group_models[group_name]:
                        assert model_key in self.model_groups, "invalid model key"
                        if not self.model_groups[model_key]:
                            continue
                        context = self.model_groups[model_key]
                        target = list(chain(*utt_groups[group_name]))
                        surprise_scores[
                            Surprise._format_attr_key(group_name, model_key, group_model_attr_key)
                        ] = self._compute_surprise(target, context)
                obj.add_meta(self.surprise_attr_name, surprise_scores)
        return corpus

    def _compute_surprise(self, target: List[str], context: List[List[str]]):
        """
        Computes how surprising a target text is based on a context. Surprise scores are calculated using cross entropy.
        To mitigate length based effects on cross entropy, several random sample of fixed sizes are taken from the traget and context.
        Returns the average of the cross entropies for all pairs of samples.

        :param target: a list of tokens in the target
        :param context: a list of lists of tokens in each group of the context

        :return: surprise score
        """
        target_tokens = np.array(target)
        context_tokens = [np.array(text) for text in context]
        target_samples = self.sampling_fn([target_tokens], self.target_sample_size, self.n_samples)
        context_samples = self.sampling_fn(context_tokens, self.context_sample_size, self.n_samples)
        if target_samples is None or context_samples is None:
            return np.nan
        return np.nanmean(
            [
                _cross_entropy(target_sample, context_sample, self.smooth)
                for target_sample, context_sample in zip(target_samples, context_samples)
            ]
        )

    @staticmethod
    def _format_attr_key(group_name, model_key, format_fn=None):
        if format_fn:
            return format_fn(group_name, model_key)
        if group_name == model_key:
            return model_key
        return f"GROUP_{group_name}__MODEL_{model_key}"
