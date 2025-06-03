from convokit.model import Corpus
from convokit.transformer import Transformer
from tqdm import tqdm
from typing import Callable
from convokit.model.conversation import Conversation
import re

from .balance_util import (
    _get_ps,
    _convo_balance_score,
    _convo_balance_lst,
    _plot_individual_conversation_floors,
    _plot_multi_conversation_floors,
)


def plot_single_conversation_balance(
    corpus,
    convo_id,
    window_ps_threshold,
    window_size,
    sliding_size,
    remove_first_last_utt,
    min_utt_words,
    plot_name=None,
    window_ss_threshold=None,
):
    if window_ss_threshold is None:
        window_ss_threshold = window_ps_threshold
    _plot_individual_conversation_floors(
        corpus,
        convo_id,
        window_ps_threshold,
        window_ss_threshold,
        window_size,
        sliding_size,
        remove_first_last_utt,
        min_utt_words,
        plot_name=plot_name,
    )


def plot_multi_conversation_balance(
    corpus,
    convo_id_lst,
    window_ps_threshold,
    window_ss_threshold,
    window_size,
    sliding_size,
    remove_first_last_utt,
    min_utt_words,
    plot_name=None,
):
    if window_ss_threshold is None:
        window_ss_threshold = window_ps_threshold
    _plot_multi_conversation_floors(
        corpus,
        convo_id_lst,
        window_ps_threshold,
        window_ss_threshold,
        window_size,
        sliding_size,
        remove_first_last_utt,
        min_utt_words,
        plot_name=plot_name,
    )


class Balance(Transformer):
    """
    The Balance transformer quantifies and annotates conversations' talk-time sharing dynamics
    between predefined speaker groups within a corpus.

    It assigns each conversation a primary speaker group (more talkative), a secondary
    speaker group (less talkative), and a scalar imbalance score. It also computes a
    list of windowed imbalance scores over a sliding windows of the conversation.

    Each utterance is expected to have a speaker group label under `utt.meta['utt_group']`,
    which can be precomputed or inferred from `convo.meta['speaker_groups']`.
    Annotation of speaker groups for each utterance is required before using the Balance transformer.
    The transform() function assumes either `convo.meta['speaker_groups']`  or `utt.meta['utt_group']`
    is already presented in the corpus for correct computation.

    :param primary_threshold: Minimum talk-time share to label a group as the primary speaker.
    :param window_ps_threshold: Talk-time share threshold for identifying dominance in a time window for primary speaker group.
    :param window_ss_threshold: Talk-time share threshold for identifying dominance in a time window for secondary speaker group. If not provided, defaults to `window_ps_threshold`.
    :param window_size: Length (in minutes) of each analysis window.
    :param sliding_size: Step size (in seconds) to slide the window forward.
    :param min_utt_words: Exclude utterances shorter than this number of words from the analysis.
    :param remove_first_last_utt: Whether to exclude the first and last utterance.
    """

    def __init__(
        self,
        primary_threshold=0.50001,
        window_ps_threshold=0.6,
        window_ss_threshold=None,
        window_size=2.5,
        sliding_size=30,
        min_utt_words=0,
        remove_first_last_utt=True,
    ):
        self.primary_threshold = primary_threshold
        self.window_ps_threshold = window_ps_threshold
        self.window_ss_threshold = (
            window_ss_threshold if window_ss_threshold else window_ps_threshold
        )
        self.window_size = window_size
        self.sliding_size = sliding_size
        self.min_utt_words = min_utt_words
        self.remove_first_last_utt = remove_first_last_utt

    def transform(
        self, corpus: Corpus, selector: Callable[[Conversation], bool] = lambda convo: True
    ):
        """
        Computes talk-time balance metrics for each conversation in the corpus.

        Annotates the corpus with speaker group labels and if utterances `utt_group` metadata is missing, the data
        is assumed to be labeled in `convo.meta['speaker_groups']`.
        Each conversation is then annotated with its primary and secondary speaker groups, an overall conversation level
        imbalance score, and a list of windowed imbalance score computed via sliding window analysis.

        :param corpus: Corpus to transform
        :param selector: (lambda) function selecting conversations to include in this accuracy calculation;

        :return: The input corpus where selected data is annotated with talk-time sharing dynamics information
        """
        ### Annotate utterances with speaker group information
        if "utt_group" not in corpus.random_utterance().meta.keys():
            for convo in tqdm(
                corpus.iter_conversations(),
                desc="Annotating speaker groups based on `speaker_groups` from conversation metadata",
            ):
                if selector(convo):
                    if "speaker_groups" not in convo.meta:
                        raise ValueError(
                            f"Missing 'speaker_groups' metadata in conversation {convo.id}, which is required for annotating utterances."
                        )
                    speaker_groups_dict = convo.meta["speaker_groups"]
                    for utt in convo.iter_utterances():
                        utt.meta["utt_group"] = speaker_groups_dict[utt.speaker.id]

        ### Annotate conversations with Balance information
        for convo in tqdm(corpus.iter_conversations(), desc="Annotating conversation balance"):
            if selector(convo):
                convo.meta["primary_speaker"] = _get_ps(
                    corpus,
                    convo,
                    self.remove_first_last_utt,
                    self.min_utt_words,
                    self.primary_threshold,
                )
                if convo.meta["primary_speaker"] is not None:
                    convo.meta["secondary_speaker"] = (
                        "groupA" if convo.meta["primary_speaker"] == "groupB" else "groupB"
                    )
                else:
                    convo.meta["secondary_speaker"] = None
                convo.meta["balance_score"] = _convo_balance_score(
                    corpus, convo.id, self.remove_first_last_utt, self.min_utt_words
                )
                convo.meta["balance_lst"] = _convo_balance_lst(
                    corpus,
                    convo.id,
                    self.window_ps_threshold,
                    self.window_ss_threshold,
                    self.window_size,
                    self.sliding_size,
                    self.remove_first_last_utt,
                    self.min_utt_words,
                )

    def fit_transform(
        self, corpus: Corpus, selector: Callable[[Conversation], bool] = lambda convo: True
    ):
        """
        Same as transform.

        :param corpus: Corpus to transform
        :param selector: (lambda) function selecting conversations to include in this accuracy calculation;
        """
        return self.transform(corpus, selector=selector)
