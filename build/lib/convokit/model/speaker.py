from functools import total_ordering
from typing import Dict, List, Optional

from .corpusComponent import CorpusComponent
from .corpusUtil import *


@total_ordering
class Speaker(CorpusComponent):
    """
    Represents a single speaker in a dataset.

    :param id: id of the speaker.
    :type id: str
    :param utts: dictionary of utterances by the speaker, where key is utterance id
    :param convos: dictionary of conversations started by the speaker, where key is conversation id
    :param meta: arbitrary dictionary of attributes associated
        with the speaker.
    :type meta: dict

    :ivar id: id of the speaker.
    :ivar meta: A dictionary-like view object providing read-write access to
        speaker-level metadata.
    """

    def __init__(
        self,
        owner=None,
        id: str = None,
        utts=None,
        convos=None,
        meta: Optional[Dict] = None,
    ):
        super().__init__(obj_type="speaker", owner=owner, id=id, meta=meta)
        self.utterances = utts if utts is not None else dict()
        self.conversations = convos if convos is not None else dict()
        # self._split_attribs = set()
        # self._update_uid()

    # def identify_by_attribs(self, attribs: Collection) -> None:
    #     """Identify a speaker by a list of attributes. Sets which speaker info
    #     attributes should distinguish speakers of the same name in equality tests.
    #     For example, in the Supreme Court dataset, speakers are labeled with the
    #     current case id. Call this method with attribs = ["case"] to count
    #     the same person across different cases as different speakers.
    #
    #     By default, if this function is not called, speakers are identified by name only.
    #
    #     :param attribs: Collection of attribute names.
    #     :type attribs: Collection
    #     """
    #
    #     self._split_attribs = set(attribs)
    #     # self._update_uid()

    def _add_utterance(self, utt):
        self.utterances[utt.id] = utt

    def _add_conversation(self, convo):
        self.conversations[convo.id] = convo

    def get_utterance(self, ut_id: str):  # -> Utterance:
        """
        Get the Utterance with the specified Utterance id

        :param ut_id: The id of the Utterance
        :return: An Utterance object
        """
        return self.utterances[ut_id]

    def iter_utterances(self, selector=lambda utt: True):  # -> Generator[Utterance, None, None]:
        """
        Get utterances made by the Speaker, with an optional selector that selects for Utterances that
        should be included.

                :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
                        By default, the selector includes all Utterances in the Corpus.
        :return: An iterator of the Utterances made by the speaker
        """
        for v in self.utterances.values():
            if selector(v):
                yield v

    def get_utterances_dataframe(self, selector=lambda utt: True, exclude_meta: bool = False):
        """
        Get a DataFrame of the Utterances made by the Speaker with fields and metadata attributes.
        Set an optional selector that filters for Utterances that should be included.
        Edits to the DataFrame do not change the corpus in any way.

        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Utterance and returns True or False (i.e. include / exclude).
                By default, the selector includes all Utterances in the Corpus.
        :return: a pandas DataFrame
        """
        return get_utterances_dataframe(self, selector, exclude_meta)

    def get_utterance_ids(self, selector=lambda utt: True) -> List[str]:
        """

        :return: a List of the ids of Utterances made by the speaker
        """
        return list([utt.id for utt in self.iter_utterances(selector)])

    def get_conversation(self, cid: str):  # -> Conversation:
        """
        Get the Conversation with the specified Conversation id

        :param cid: The id of the Conversation
        :return: A Conversation object
        """
        return self.conversations[cid]

    def iter_conversations(
        self, selector=lambda convo: True
    ):  # -> Generator[Conversation, None, None]:
        """

        :return: An iterator of the Conversations that the speaker has participated in
        """
        for v in self.conversations.values():
            if selector(v):
                yield v

    def get_conversations_dataframe(self, selector=lambda convo: True, exclude_meta: bool = False):
        """
        Get a DataFrame of the Conversations the Speaker has participated in, with fields and metadata attributes.
        Set an optional selector that filters for Conversations that should be included. Edits to the DataFrame do not
        change the corpus in any way.

        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
            By default, the selector includes all Conversations in the Corpus.
        :return: a pandas DataFrame
        """
        return get_conversations_dataframe(self, selector, exclude_meta)

    def get_conversation_ids(self, selector=lambda convo: True) -> List[str]:
        """

        :return: a List of the ids of Conversations started by the speaker
        """
        return [convo.id for convo in self.iter_conversations(selector)]

    def print_speaker_stats(self):
        """
        Helper function for printing the number of Utterances made and Conversations participated in by the Speaker.

        :return: None (prints output)
        """
        print("Number of Utterances: {}".format(len(list(self.iter_utterances()))))
        print("Number of Conversations: {}".format(len(list(self.iter_conversations()))))

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Speaker):
            return False
        try:
            return self.id == other.id
        except AttributeError:
            return self.__dict__["_name"] == other.__dict__["_name"]

    def __str__(self):
        return "Speaker(id: {}, vectors: {}, meta: {})".format(
            repr(self.id), self.vectors, self.meta
        )
