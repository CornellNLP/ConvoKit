from collections import defaultdict
from typing import Dict, List, Callable, Generator, Optional

from convokit.util import warn
from .corpusComponent import CorpusComponent
from .corpusUtil import *
from .speaker import Speaker
from .utterance import Utterance
from .utteranceNode import UtteranceNode


class Conversation(CorpusComponent):
    """
    Represents a discrete subset of utterances in the dataset, connected by a reply-to chain.

    :param owner: The Corpus that this Conversation belongs to
    :param id: The unique ID of this Conversation
    :param utterances: A list of the IDs of the Utterances in this Conversation
    :param meta: Table of initial values for conversation-level metadata

    :ivar id: the ID of the Conversation
    :ivar meta: A dictionary-like view object providing read-write access to conversation-level metadata.
    """

    def __init__(
        self,
        owner,
        id: Optional[str] = None,
        utterances: Optional[List[str]] = None,
        meta: Optional[Dict] = None,
    ):
        super().__init__(obj_type="conversation", owner=owner, id=id, meta=meta)
        self._owner = owner
        self._utterance_ids: List[str] = utterances
        self._speaker_ids = None
        self.tree: Optional[UtteranceNode] = None

    def _add_utterance(self, utt: Utterance):
        self._utterance_ids.append(utt.id)
        self._speaker_ids = None
        self.tree = None

    def get_utterance_ids(self) -> List[str]:
        """Produces a list of the unique IDs of all utterances in the
        Conversation, which can be used in calls to get_utterance() to retrieve
        specific utterances. Provides no ordering guarantees for the list.

        :return: a list of IDs of Utterances in the Conversation
        """
        # pass a copy of the list
        return self._utterance_ids[:]

    def get_utterance(self, ut_id: str) -> Utterance:
        """Looks up the Utterance associated with the given ID. Raises a
        KeyError if no utterance by that ID exists.

        :return: the Utterance with the given ID
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_utterance(ut_id)

    def iter_utterances(
        self, selector: Callable[[Utterance], bool] = lambda utt: True
    ) -> Generator[Utterance, None, None]:
        """
        Get utterances in the Corpus, with an optional selector that filters for Utterances that should be included.

        :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude). By default, the selector includes all Utterances in the Conversation.
        :return: a generator of Utterances
        """
        for ut_id in self._utterance_ids:
            utt = self._owner.get_utterance(ut_id)
            if selector(utt):
                yield utt

    def get_utterances_dataframe(
        self, selector: Callable[[Utterance], bool] = lambda utt: True, exclude_meta: bool = False
    ):
        """
        Get a DataFrame of the Utterances in the Conversation with fields and metadata attributes.
                Set an optional selector that filters for Utterances that should be included.
                Edits to the DataFrame do not change the corpus in any way.

        :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
                        By default, the selector includes all Utterances in the Conversation.
        :param exclude_meta: whether to exclude metadata
        :return: a pandas DataFrame
        """
        return get_utterances_dataframe(self, selector, exclude_meta)

    def get_speaker_ids(self) -> List[str]:
        """
        Produces a list of ids of all speakers in the Conversation, which can be used in calls to get_speaker()
        to retrieve specific speakers. Provides no ordering guarantees for the list.

        :return: a list of speaker ids
        """
        if self._speaker_ids is None:
            # first call to get_speaker_ids or iter_speakers; precompute cached list of speaker ids
            self._speaker_ids = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._speaker_ids.add(ut.speaker.id)
        return list(self._speaker_ids)

    def get_speaker(self, speaker_id: str) -> Speaker:
        """
        Looks up the Speaker with the given name. Raises a KeyError if no speaker
        with that name exists.

        :return: the Speaker with the given speaker_id
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_speaker(speaker_id)

    def iter_speakers(
        self, selector: Callable[[Speaker], bool] = lambda speaker: True
    ) -> Generator[Speaker, None, None]:
        """
        Get Speakers that have participated in the Conversation, with an optional selector that filters for Speakers
        that should be included.

                :param selector: a (lambda) function that takes a Speaker and returns True or False (i.e. include / exclude).
                        By default, the selector includes all Speakers in the Conversation.
                :return: a generator of Speakers
        """
        if self._speaker_ids is None:
            # first call to get_ids or iter_speakers; precompute cached list of speaker ids
            self._speaker_ids = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._speaker_ids.add(ut.speaker.id)
        for speaker_id in self._speaker_ids:
            speaker = self._owner.get_speaker(speaker_id)
            if selector(speaker):
                yield speaker

    def get_speakers_dataframe(
        self,
        selector: Optional[Callable[[Speaker], bool]] = lambda utt: True,
        exclude_meta: bool = False,
    ):
        """
        Get a DataFrame of the Speakers that have participated in the Conversation with fields and metadata attributes,
        with an optional selector that filters Speakers that should be included.
        Edits to the DataFrame do not change the corpus in any way.

                :param exclude_meta: whether to exclude metadata
                :param selector: selector: a (lambda) function that takes a Speaker and returns True or False
                        (i.e. include / exclude). By default, the selector includes all Speakers in the Conversation.
                :return: a pandas DataFrame
        """
        return get_speakers_dataframe(self, selector, exclude_meta)

    def print_conversation_stats(self):
        """
        Helper function for printing the number of Utterances and Spekaers in the Conversation.

        :return: None (prints output)
        """
        print("Number of Utterances: {}".format(len(list(self.iter_utterances()))))
        print("Number of Speakers: {}".format(len(list(self.iter_speakers()))))

    def get_chronological_speaker_list(
        self, selector: Callable[[Speaker], bool] = lambda speaker: True
    ):
        """
        Get the speakers in the conversation sorted in chronological order (speakers may appear more than once)

        :param selector: (lambda) function for which speakers should be included; all speakers are included by default
        :return: list of speakers for each chronological utterance
        """
        try:
            chrono_utts = sorted(list(self.iter_utterances()), key=lambda utt: utt.timestamp)
            return [utt.speaker for utt in chrono_utts if selector(utt.speaker)]
        except TypeError as e:
            raise ValueError(str(e) + "\nUtterance timestamps may not have been set correctly.")

    def check_integrity(self, verbose: bool = True) -> bool:
        """
        Check the integrity of this Conversation; i.e. do the constituent utterances form a complete reply-to chain?

        :param verbose: whether to print errors indicating the problems with the Conversation
        :return: True if the conversation structure is complete else False
        """
        if verbose:
            print("Checking reply-to chain of Conversation", self.id)
        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utt_ids = set(list(utt_reply_tos.values()))
        speaker_utt_ids = set(list(utt_reply_tos.keys()))
        root_utt_id = target_utt_ids - speaker_utt_ids  # There should only be 1 root_utt_id: None

        if len(root_utt_id) != 1:
            if verbose:
                for utt_id in root_utt_id:
                    if utt_id is not None:
                        warn("ERROR: Missing utterance {}".format(utt_id))
            return False
        else:
            root_id = list(root_utt_id)[0]
            if root_id is not None:
                if verbose:
                    warn("ERROR: Missing utterance {}".format(root_id))
                return False

        # sanity check
        utts_replying_to_none = 0
        for utt in self.iter_utterances():
            if utt.reply_to is None:
                utts_replying_to_none += 1

        if utts_replying_to_none > 1:
            if verbose:
                warn("ERROR: Found more than one Utterance replying to None.")
            return False

        circular = [
            utt_id for utt_id, utt_reply_to in utt_reply_tos.items() if utt_id == utt_reply_to
        ]
        if len(circular) > 0:
            if verbose:
                warn(
                    "ERROR: Found utterances with .reply_to pointing to themselves: {}".format(
                        circular
                    )
                )
            return False

        if verbose:
            print("No issues found.\n")
        return True

    def initialize_tree_structure(self):
        if not self.check_integrity(verbose=False):
            raise ValueError(
                "Conversation {} reply-to chain does not form a valid tree.".format(self.id)
            )

        root_node_id = None
        # Find root node
        for utt in self.iter_utterances():
            if utt.reply_to is None:
                root_node_id = utt.id

        parent_to_children_ids = defaultdict(list)
        for utt in self.iter_utterances():
            parent_to_children_ids[utt.reply_to].append(utt.id)

        wrapped_utts = {utt.id: UtteranceNode(utt) for utt in self.iter_utterances()}

        for parent_id, wrapped_utt in wrapped_utts.items():
            wrapped_utt.set_children(
                [wrapped_utts[child_id] for child_id in parent_to_children_ids[parent_id]]
            )

        self.tree = wrapped_utts[root_node_id]

    def traverse(self, traversal_type: str, as_utterance: bool = True):
        """
        Traverse through the Conversation tree structure in a breadth-first search ('bfs'), depth-first search (dfs),
        pre-order ('preorder'), or post-order ('postorder') way.

        :param traversal_type: dfs, bfs, preorder, or postorder
        :param as_utterance: whether the iterator should yield the utterance (True) or the utterance node (False)
        :return: an iterator of the utterances or utterance nodes
        """
        if self.tree is None:
            self.initialize_tree_structure()
            if self.tree is None:
                raise ValueError(
                    "Failed to traverse because Conversation reply-to chain does not form a valid tree."
                )

        traversals = {
            "bfs": self.tree.bfs_traversal,
            "dfs": self.tree.dfs_traversal,
            "preorder": self.tree.pre_order,
            "postorder": self.tree.post_order,
        }

        for utt_node in traversals[traversal_type]():
            yield utt_node.utt if as_utterance else utt_node

    def get_subtree(self, root_utt_id):
        """
        Get the utterance node of the specified input id

        :param root_utt_id: id of the root node that the subtree starts from
        :return: UtteranceNode object
        """
        if self.tree is None:
            self.initialize_tree_structure()
            if self.tree is None:
                raise ValueError(
                    "Failed to traverse because Conversation reply-to chain does not form a valid tree."
                )

        for utt_node in self.tree.bfs_traversal():
            if utt_node.utt.id == root_utt_id:
                return utt_node

    def get_longest_paths(self) -> List[List[Utterance]]:
        """
        Finds the Utterances form the longest path (i.e. root to leaf) in the Conversation tree.
        If there are multiple paths with tied lengths, returns all of them as a list of lists. If only one such path
        exists, a list containing a single list of Utterances is returned.

        :return: a list of lists of Utterances
        """
        if self.tree is None:
            self.initialize_tree_structure()
            if self.tree is None:
                raise ValueError(
                    "Failed to traverse because Conversation reply-to chain does not form a valid tree."
                )

        paths = self.get_root_to_leaf_paths()
        max_len = max(len(path) for path in paths)

        return [p for p in paths if len(p) == max_len]

    def _print_convo_helper(
        self,
        root: str,
        indent: int,
        reply_to_dict: Dict[str, str],
        utt_info_func: Callable[[Utterance], str],
        limit=None,
    ) -> None:
        """
        Helper function for print_conversation_structure()
        """
        if limit is not None:
            if self.get_utterance(root).meta["order"] > limit:
                return
        print(" " * indent + utt_info_func(self.get_utterance(root)))
        children_utt_ids = [k for k, v in reply_to_dict.items() if v == root]
        for child_utt_id in children_utt_ids:
            self._print_convo_helper(
                root=child_utt_id,
                indent=indent + 4,
                reply_to_dict=reply_to_dict,
                utt_info_func=utt_info_func,
                limit=limit,
            )

    def print_conversation_structure(
        self,
        utt_info_func: Callable[[Utterance], str] = lambda utt: utt.speaker.id,
        limit: int = None,
    ) -> None:
        """
        Prints an indented representation of utterances in the Conversation with conversation reply-to structure
        determining the indented level. The details of each utterance to be printed can be configured.

        If limit is set to a value other than None, this will annotate utterances with an 'order' metadata indicating
        their temporal order in the conversation, where the first utterance in the conversation is annotated with 1.

        :param utt_info_func: callable function taking an utterance as input and returning a string of the desired
            utterance information. By default, this is a lambda function returning the utterance's speaker's id
        :param limit: maximum number of utterances to print out. if k, this includes the first k utterances.
        :return: None. Prints to stdout.
        """
        if not self.check_integrity(verbose=False):
            raise ValueError(
                "Could not print conversation structure: The utterance reply-to chain is broken. "
                "Try check_integrity() to diagnose the problem."
            )

        if limit is not None:
            assert isinstance(limit, int)
            for idx, utt in enumerate(self.get_chronological_utterance_list()):
                utt.meta["order"] = idx + 1

        root_utt_id = [utt for utt in self.iter_utterances() if utt.reply_to is None][0].id
        reply_to_dict = {utt.id: utt.reply_to for utt in self.iter_utterances()}

        self._print_convo_helper(
            root=root_utt_id,
            indent=0,
            reply_to_dict=reply_to_dict,
            utt_info_func=utt_info_func,
            limit=limit,
        )

    def get_utterances_dataframe(self, selector=lambda utt: True, exclude_meta: bool = False):
        """
        Get a DataFrame of the Utterances in the COnversation with fields and metadata attributes.
        Set an optional selector that filters Utterances that should be included.
        Edits to the DataFrame do not change the corpus in any way.

        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Utterance and returns True or False (i.e. include / exclude).
                By default, the selector includes all Utterances in the Conversation.
        :return: a pandas DataFrame
        """
        return get_utterances_dataframe(self, selector, exclude_meta)

    def get_chronological_utterance_list(
        self, selector: Callable[[Utterance], bool] = lambda utt: True
    ):
        """
        Get the utterances in the conversation sorted in increasing order of timestamp

        :param selector: function for which utterances should be included; all utterances are included by default
        :return: list of utterances, sorted by timestamp
        """
        try:
            return sorted(
                [utt for utt in self.iter_utterances(selector)], key=lambda utt: utt.timestamp
            )
        except TypeError as e:
            raise ValueError(str(e) + "\nUtterance timestamps may not have been set correctly.")

    def _get_path_from_leaf_to_root(
        self, leaf_utt: Utterance, root_utt: Utterance
    ) -> List[Utterance]:
        """
        Helper function for get_root_to_leaf_paths, which returns the path for a given leaf_utt and root_utt
        """
        if leaf_utt == root_utt:
            return [leaf_utt]
        path = [leaf_utt]
        root_id = root_utt.id
        while leaf_utt.reply_to != root_id:
            path.append(self.get_utterance(leaf_utt.reply_to))
            leaf_utt = path[-1]
        path.append(root_utt)
        return path[::-1]

    def get_root_to_leaf_paths(self) -> List[List[Utterance]]:
        """
        Get the paths (stored as a list of lists of utterances) from the root to each of the leaves
        in the conversational tree

        :return: List of lists of Utterances
        """
        if not self.check_integrity(verbose=False):
            raise ValueError(
                "Conversation failed integrity check. "
                "It is either missing an utterance in the reply-to chain and/or has multiple root nodes. "
                "Run check_integrity() to diagnose issues."
            )

        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utt_ids = set(list(utt_reply_tos.values()))
        speaker_utt_ids = set(list(utt_reply_tos.keys()))
        root_utt_id = target_utt_ids - speaker_utt_ids  # There should only be 1 root_utt_id: None
        assert len(root_utt_id) == 1
        root_utt = [utt for utt in self.iter_utterances() if utt.reply_to is None][0]
        leaf_utt_ids = speaker_utt_ids - target_utt_ids

        paths = [
            self._get_path_from_leaf_to_root(self.get_utterance(leaf_utt_id), root_utt)
            for leaf_utt_id in leaf_utt_ids
        ]
        return paths

    @staticmethod
    def generate_default_conversation_id(utterance_id):
        return f"__default_conversation__{utterance_id}"

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.id == other.id and set(self._utterance_ids) == set(other._utterance_ids)

    def __str__(self):
        return "Conversation('id': {}, 'utterances': {}, 'meta': {})".format(
            repr(self.id), self._utterance_ids, self.meta
        )
