from pandas import DataFrame
from tqdm import tqdm
from typing import List, Collection, Callable, Set, Generator, Tuple, Optional, ValuesView, Union
from .corpusHelper import *
from convokit.util import deprecation, warn
from .corpusUtil import *
from .convoKitIndex import ConvoKitIndex
import random
from .convoKitMeta import ConvoKitMeta
from .convoKitMatrix import ConvoKitMatrix
import shutil


class Corpus:
    """
    Represents a dataset, which can be loaded from a folder or constructed from a list of utterances.

    :param filename: Path to a folder containing a Corpus or to an utterances.jsonl / utterances.json file to load
    :param utterances: list of utterances to initialize Corpus from
    :param preload_vectors: list of names of vectors to be preloaded from directory; by default,
        no vectors are loaded but can be loaded any time after corpus initialization (i.e. vectors are lazy-loaded).
    :param utterance_start_index: if loading from directory and the corpus folder contains utterances.jsonl, specify the
        line number (zero-indexed) to begin parsing utterances from
    :param utterance_end_index: if loading from directory and the corpus folder contains utterances.jsonl, specify the
        line number (zero-indexed) of the last utterance to be parsed.
    :param merge_lines: whether to merge adjacent lines from same speaker if multiple consecutive utterances belong to
        the same conversation.
    :param exclude_utterance_meta: utterance metadata to be ignored
    :param exclude_conversation_meta: conversation metadata to be ignored
    :param exclude_speaker_meta: speaker metadata to be ignored
    :param exclude_overall_meta: overall metadata to be ignored
    :param disable_type_check: whether to do type checking when loading the Corpus from a directory.
        Type-checking ensures that the ConvoKitIndex is initialized correctly. However, it may be unnecessary if the
        index.json is already accurate and disabling it will allow for a faster corpus load. This parameter is set to
        True by default, i.e. type-checking is not carried out.

    :ivar meta_index: index of Corpus metadata
    :ivar vectors: the vectors stored in the Corpus
    :ivar corpus_dirpath: path to the directory the corpus was loaded from
    """

    def __init__(self, filename: Optional[str] = None, utterances: Optional[List[Utterance]] = None,
                 preload_vectors: List[str] = None, utterance_start_index: int = None,
                 utterance_end_index: int = None, merge_lines: bool = False,
                 exclude_utterance_meta: Optional[List[str]] = None,
                 exclude_conversation_meta: Optional[List[str]] = None,
                 exclude_speaker_meta: Optional[List[str]] = None,
                 exclude_overall_meta: Optional[List[str]] = None,
                 disable_type_check=True):

        if filename is None:
            self.corpus_dirpath = None
        elif os.path.isdir(filename):
            self.corpus_dirpath = filename
        else:
            self.corpus_dirpath = os.path.dirname(filename)

        self.meta_index = ConvoKitIndex(self)
        self.meta = ConvoKitMeta(self.meta_index, 'corpus')

        # private storage
        self._vector_matrices = dict()

        convos_data = defaultdict(dict)
        if exclude_utterance_meta is None: exclude_utterance_meta = []
        if exclude_conversation_meta is None: exclude_conversation_meta = []
        if exclude_speaker_meta is None: exclude_speaker_meta = []
        if exclude_overall_meta is None: exclude_overall_meta = []

        # Construct corpus from file or directory
        if filename is not None:
            if disable_type_check: self.meta_index.disable_type_check()
            if os.path.isdir(filename):
                utterances = load_uttinfo_from_dir(filename, utterance_start_index,
                                                   utterance_end_index, exclude_utterance_meta)

                speakers_data = load_speakers_data_from_dir(filename, exclude_speaker_meta)
                convos_data = load_convos_data_from_dir(filename, exclude_conversation_meta)
                load_corpus_meta_from_dir(filename, self.meta, exclude_overall_meta)

                with open(os.path.join(filename, "index.json"), "r") as f:
                    idx_dict = json.load(f)
                    self.meta_index.update_from_dict(idx_dict)

                # load all processed text information, but don't load actual text.
                # also checks if the index file exists.
                # try:
                #     with open(os.path.join(filename, "processed_text.index.json"), "r") as f:
                #         self.processed_text = {k: {} for k in json.load(f)}
                # except:
                #     pass

                # unpack binary data for utterances
                unpack_binary_data_for_utts(utterances, filename, self.meta_index.utterances_index,
                                            exclude_utterance_meta, KeyMeta)
                # unpack binary data for speakers
                unpack_binary_data(filename, speakers_data, self.meta_index.speakers_index, "speaker",
                                   exclude_speaker_meta)

                # unpack binary data for conversations
                unpack_binary_data(filename, convos_data, self.meta_index.conversations_index, "convo",
                                   exclude_conversation_meta)

                # unpack binary data for overall corpus
                unpack_binary_data(filename, self.meta, self.meta_index.overall_index, "overall", exclude_overall_meta)

            else:
                speakers_data = defaultdict(dict)
                convos_data = defaultdict(dict)
                utterances = load_from_utterance_file(filename, utterance_start_index, utterance_end_index)

            self.utterances = dict()
            self.speakers = dict()

            initialize_speakers_and_utterances_objects(self, self.utterances, utterances, self.speakers, speakers_data)

            self.meta_index.enable_type_check()

            # load preload_vectors
            if preload_vectors is not None:
                for vector_name in preload_vectors:
                    matrix = ConvoKitMatrix.from_dir(self.corpus_dirpath, vector_name)
                    if matrix is not None:
                        self._vector_matrices[vector_name] = matrix

        elif utterances is not None:  # Construct corpus from utterances list
            self.speakers = {u.speaker.id: u.speaker for u in utterances}
            self.utterances = {u.id: u for u in utterances}
            for _, speaker in self.speakers.items():
                speaker.owner = self
            for _, utt in self.utterances.items():
                utt.owner = self

        if merge_lines:
            self.utterances = merge_utterance_lines(self.utterances)

        if disable_type_check: self.meta_index.disable_type_check()
        self.conversations = initialize_conversations(self, self.utterances, convos_data)
        self.meta_index.enable_type_check()
        self.update_speakers_data()

    @property
    def vectors(self):
        return self.meta_index.vectors

    @vectors.setter
    def vectors(self, new_vectors):
        if not isinstance(new_vectors, type(['stringlist'])):
            raise ValueError("The preload_vectors being set should be a list of strings, "
                             "where each string is the name of a vector matrix.")
        self.meta_index.vectors = new_vectors

    def dump(self, name: str, base_path: Optional[str] = None,
             exclude_vectors: List[str] = None,
             force_version: int = None,
             overwrite_existing_corpus: bool = False,
             fields_to_skip=None) -> None:
        """
        Dumps the corpus and its metadata to disk. Optionally, set `force_version` to a desired integer version number,
        otherwise the version number is automatically incremented.

        :param name: name of corpus
        :param base_path: base directory to save corpus in (None to save to a default directory)
        :param exclude_vectors: list of names of vector matrices to exclude from the dumping step. By default; all
            vector matrices that belong to the Corpus (whether loaded or not) are dumped.
        :param force_version: version number to set for the dumped corpus
        :param overwrite_existing_corpus: if True, save to the path you loaded the corpus from, overriding the original corpus. 
        :param fields_to_skip: a dictionary of {object type: list of metadata attributes to omit when writing to disk}. object types can be one of "speaker", "utterance", "conversation", "corpus".
        """
        if fields_to_skip is None:
            fields_to_skip = dict()
        dir_name = name
        if base_path is not None and overwrite_existing_corpus:
            raise ValueError("Not allowed to specify both base_path and overwrite_existing_corpus!")
        if overwrite_existing_corpus and self.corpus_dirpath is None:
            raise ValueError("Cannot use save to existing path on Corpus generated from utterance list!")
        if not overwrite_existing_corpus:
            if base_path is None:
                base_path = os.path.expanduser("~/.convokit/")
                if not os.path.exists(base_path):
                    os.mkdir(base_path)
                base_path = os.path.join(base_path, "saved-corpora/")
                if not os.path.exists(base_path):
                    os.mkdir(base_path)
            dir_name = os.path.join(base_path, dir_name)
        else:
            dir_name = os.path.join(self.corpus_dirpath)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # dump speakers, conversations, utterances
        dump_corpus_component(self, dir_name, "speakers.json", "speaker", "speaker", exclude_vectors, fields_to_skip)
        dump_corpus_component(self, dir_name, "conversations.json", "conversation", "convo",
                              exclude_vectors, fields_to_skip)
        dump_utterances(self, dir_name, exclude_vectors, fields_to_skip)

        # dump corpus
        with open(os.path.join(dir_name, "corpus.json"), "w") as f:
            d_bin = defaultdict(list)
            meta_up = dump_helper_bin(self.meta, d_bin, fields_to_skip.get('corpus', None))

            json.dump(meta_up, f)
            for name, l_bin in d_bin.items():
                with open(os.path.join(dir_name, name + "-overall-bin.p"), "wb") as f_pk:
                    pickle.dump(l_bin, f_pk)

        # dump index
        with open(os.path.join(dir_name, "index.json"), "w") as f:
            json.dump(self.meta_index.to_dict(exclude_vectors=exclude_vectors, force_version=force_version), f)

        # dump vectors
        if exclude_vectors is not None:
            vectors_to_dump = [v for v in self.vectors if v not in set(exclude_vectors)]
        else:
            vectors_to_dump = self.vectors
        for vector_name in vectors_to_dump:
            if vector_name in self._vector_matrices:
                self._vector_matrices[vector_name].dump(dir_name)
            else:
                src = os.path.join(self.corpus_dirpath, 'vectors.{}.p'.format(vector_name))
                dest = os.path.join(dir_name, 'vectors.{}.p'.format(vector_name))
                shutil.copy(src, dest)

    # with open(os.path.join(dir_name, "processed_text.index.json"), "w") as f:
    #     json.dump(list(self.processed_text.keys()), f)

    def get_utterance(self, utt_id: str) -> Utterance:
        """
        Gets Utterance of the specified id from the corpus

        :param utt_id: id of Utterance
        :return: Utterance
        """
        return self.utterances[utt_id]

    def get_conversation(self, convo_id: str) -> Conversation:
        """
        Gets Conversation of the specified id from the corpus

        :param convo_id: id of Conversation
        :return: Conversation
        """
        return self.conversations[convo_id]

    def get_speaker(self, speaker_id: str) -> Speaker:
        """
        Gets Speaker of the specified id from the corpus

        :param speaker_id: id of Speaker
        :return: Speaker
        """
        return self.speakers[speaker_id]

    def get_user(self, user_id: str) -> Speaker:
        deprecation("get_user()", "get_speaker()")
        return self.get_speaker(user_id)

    def get_object(self, obj_type: str, oid: str):
        """
        General Corpus object getter. Gets Speaker / Utterance / Conversation of specified id from the Corpus

        :param obj_type: "speaker", "utterance", or "conversation"
        :param oid: object id
        :return: Corpus object of specified object type with specified object id
        """
        assert obj_type in ["speaker", "utterance", "conversation"]
        if obj_type == "speaker":
            return self.get_speaker(oid)
        elif obj_type == "utterance":
            return self.get_utterance(oid)
        else:
            return self.get_conversation(oid)

    def has_utterance(self, utt_id: str) -> bool:
        """
        Checks if an Utterance of the specified id exists in the Corpus

        :param utt_id: id of Utterance
        :return: True if Utterance of specified id is present, False otherwise
        """
        return utt_id in self.utterances

    def has_conversation(self, convo_id: str) -> bool:
        """
        Checks if a Conversation of the specified id exists in the Corpus

        :param convo_id: id of Conversation
        :return: True if Conversation of specified id is present, False otherwise
        """
        return convo_id in self.conversations

    def has_speaker(self, speaker_id: str) -> bool:
        """
        Checks if a Speaker of the specified id exists in the Corpus

        :param speaker_id: id of Speaker
        :return: True if Speaker of specified id is present, False otherwise
        """
        return speaker_id in self.speakers

    def has_user(self, speaker_id):
        deprecation("has_user()", "has_speaker()")
        return self.has_speaker(speaker_id)

    def random_utterance(self) -> Utterance:
        """
        Get a random Utterance from the Corpus

        :return: a random Utterance
        """
        return random.choice(list(self.utterances.values()))

    def random_conversation(self) -> Conversation:
        """
        Get a random Conversation from the Corpus

        :return: a random Conversation
        """
        return random.choice(list(self.conversations.values()))

    def random_speaker(self) -> Speaker:
        """
        Get a random Speaker from the Corpus

        :return: a random Speaker
        """
        return random.choice(list(self.speakers.values()))

    def random_user(self) -> Speaker:
        deprecation("random_user()", "random_speaker()")
        return self.random_speaker()

    def iter_utterances(self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True) -> \
            Generator[Utterance, None, None]:
        """
        Get utterances in the Corpus, with an optional selector that filters for Utterances that should be included.

        :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
            By default, the selector includes all Utterances in the Corpus.
        :return: a generator of Utterances
        """
        for v in self.utterances.values():
            if selector(v):
                yield v

    def get_utterances_dataframe(self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True,
                                 exclude_meta: bool = False):
        """
        Get a DataFrame of the utterances with fields and metadata attributes, with an optional selector that filters
        utterances that should be included. Edits to the DataFrame do not change the corpus in any way.

        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Utterance and returns True or False (i.e. include / exclude).
            By default, the selector includes all Utterances in the Corpus.
        :return: a pandas DataFrame
        """
        return get_utterances_dataframe(self, selector, exclude_meta)

    def iter_conversations(self, selector: Optional[Callable[[Conversation], bool]] = lambda convo: True) -> Generator[
                           Conversation, None, None]:
        """
        Get conversations in the Corpus, with an optional selector that filters for Conversations that should be included

        :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
            By default, the selector includes all Conversations in the Corpus.
        :return: a generator of Conversations
        """
        for v in self.conversations.values():
            if selector(v):
                yield v

    def get_conversations_dataframe(self, selector: Optional[Callable[[Conversation], bool]] = lambda convo: True,
                                    exclude_meta: bool = False):
        """
        Get a DataFrame of the conversations with fields and metadata attributes, with an optional selector that filters
        for conversations that should be included. Edits to the DataFrame do not change the corpus in any way.

        :param exclude_meta: whether to exclude metadata
        :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
            By default, the selector includes all Conversations in the Corpus.
        :return: a pandas DataFrame
        """
        return get_conversations_dataframe(self, selector, exclude_meta)

    def iter_speakers(self, selector: Optional[Callable[[Speaker], bool]] = lambda speaker: True) -> \
            Generator[Speaker, None, None]:
        """
        Get Speakers in the Corpus, with an optional selector that filters for Speakers that should be included

        :param selector: a (lambda) function that takes a Speaker and returns True or False (i.e. include / exclude).
            By default, the selector includes all Speakers in the Corpus.
        :return: a generator of Speakers
        """

        for speaker in self.speakers.values():
            if selector(speaker):
                yield speaker

    def get_speakers_dataframe(self, selector: Optional[Callable[[Speaker], bool]] = lambda utt: True,
                               exclude_meta: bool = False):
        """
        Get a DataFrame of the Speakers with fields and metadata attributes, with an optional selector that filters
        Speakers that should be included. Edits to the DataFrame do not change the corpus in any way.

        :param exclude_meta: whether to exclude metadata
        :param selector: selector: a (lambda) function that takes a Speaker and returns True or False
            (i.e. include / exclude). By default, the selector includes all Speakers in the Corpus.
        :return: a pandas DataFrame
        """
        return get_speakers_dataframe(self, selector, exclude_meta)

    def iter_users(self, selector=lambda speaker: True):
        deprecation("iter_users()", "iter_speakers()")
        return self.iter_speakers(selector)

    def iter_objs(self, obj_type: str,
                  selector: Callable[[Union[Speaker, Utterance, Conversation]], bool] = lambda obj: True):
        """
        Get Corpus objects of specified type from the Corpus, with an optional selector that filters for Corpus object that should be included

        :param obj_type: "speaker", "utterance", or "conversation"
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
            By default, the selector includes all objects of the specified type in the Corpus.
        :return: a generator of Speakers
        """

        assert obj_type in ["speaker", "utterance", "conversation"]
        obj_iters = {"conversation": self.iter_conversations,
                     "speaker": self.iter_speakers,
                     "utterance": self.iter_utterances}

        return obj_iters[obj_type](selector)

    def get_utterance_ids(self, selector: Optional[Callable[[Utterance], bool]] = lambda utt: True) -> List[str]:
        """
        Get a list of ids of Utterances in the Corpus, with an optional selector that filters for Utterances that should be included

        :param selector: a (lambda) function that takes an Utterance and returns True or False (i.e. include / exclude).
            By default, the selector includes all Utterances in the Corpus.
        :return: list of Utterance ids
        """
        return [utt.id for utt in self.iter_utterances(selector)]

    def get_conversation_ids(self, selector: Optional[Callable[[Conversation], bool]] = lambda convo: True) -> List[
        str]:
        """
        Get a list of ids of Conversations in the Corpus, with an optional selector that filters for Conversations that should be included

        :param selector: a (lambda) function that takes a Conversation and returns True or False (i.e. include / exclude).
            By default, the selector includes all Conversations in the Corpus.
        :return: list of Conversation ids
        """
        return [convo.id for convo in self.iter_conversations(selector)]

    def get_speaker_ids(self, selector: Optional[Callable[[Speaker], bool]] = lambda speaker: True) -> List[
        str]:
        """
        Get a list of ids of Speakers in the Corpus, with an optional selector that filters for Speakers that should be included

        :param selector: a (lambda) function that takes a Speaker and returns True or False (i.e. include / exclude).
            By default, the selector includes all Speakers in the Corpus.
        :return: list of Speaker ids
        """
        return [speaker.id for speaker in self.iter_speakers(selector)]

    def get_object_ids(self, obj_type: str,
                       selector: Callable[[Union[Speaker, Utterance, Conversation]], bool] = lambda obj: True):
        """
        Get a list of ids of Corpus objects of the specified type in the Corpus, with an optional selector that filters for objects that should be included

        :param obj_type: "speaker", "utterance", or "conversation"
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude).
            By default, the selector includes all objects of the specified type in the Corpus.
        :return: list of Corpus object ids
        """
        assert obj_type in ["speaker", "utterance", "conversation"]
        return [obj.id for obj in self.iter_objs(obj_type, selector)]

    def get_usernames(self, selector: Optional[Callable[[Speaker], bool]] = lambda user: True) -> Set[str]:
        """Get names of speakers in the dataset.

        This function will be deprecated and replaced by get_speaker_ids()

        :param selector: optional function that takes in a
            `Speaker` and returns True to include the speaker's name in the
            resulting list, or False otherwise.

        :return: Set containing all speaker names selected by the selector
            function, or all speaker names in the dataset if no selector function
            was used.

        """
        deprecation("get_usernames()", "get_speaker_ids()")
        return set([u.id for u in self.iter_speakers(selector)])

    def filter_conversations_by(self, selector: Callable[[Conversation], bool]):
        """
        Mutate the corpus by filtering for a subset of Conversations within the Corpus.

        :param selector: function for selecting which Conversations to keep
        :return: the mutated Corpus
        """

        self.conversations = {convo_id: convo for convo_id, convo in self.conversations.items() if selector(convo)}
        utt_ids = set([utt for convo in self.conversations.values() for utt in convo.get_utterance_ids()])
        self.utterances = {utt.id: utt for utt in self.utterances.values() if utt.id in utt_ids}
        speaker_ids = set([utt.speaker.id for utt in self.utterances.values()])
        self.speakers = {speaker.id: speaker for speaker in self.speakers.values() if speaker.id in speaker_ids}
        self.update_speakers_data()
        self.reinitialize_index()
        return self

    def filter_utterances_by(self, selector: Callable[[Utterance], bool]):
        """
        Returns a new corpus that includes only a subset of Utterances within this Corpus. This filtering provides no
        guarantees with regard to maintaining conversational integrity and should be used with care.

        Vectors are not preserved.

        :param selector: function for selecting which
        :return: a new Corpus with a subset of the Utterances
        """
        utts = list(self.iter_utterances(selector))
        new_corpus = Corpus(utterances=utts)
        for convo in new_corpus.iter_conversations():
            convo.meta.update(self.get_conversation(convo.id).meta)
        return new_corpus

    def reindex_conversations(self, new_convo_roots: List[str], preserve_corpus_meta: bool = True,
                              preserve_convo_meta: bool = True, verbose=True) -> 'Corpus':
        """
        Generates a new Corpus from current Corpus with specified list of utterance ids to use as conversation ids.

        The subtrees denoted by these utterance ids should be distinct and should not overlap, otherwise there may be unexpected behavior.

        Vectors are not preserved. The original Corpus will be mutated.

        :param new_convo_roots: List of utterance ids to use as conversation ids
        :param preserve_corpus_meta: set as True to copy original Corpus metadata to new Corpus
        :param preserve_convo_meta: set as True to copy original Conversation metadata to new Conversation metadata
            (For each new conversation, use the metadata of the conversation that the utterance belonged to.)
        :param verbose: whether to print a warning when
        :return: new Corpus with reindexed Conversations
        """""
        new_convo_roots = set(new_convo_roots)
        for convo in self.iter_conversations():
            try:
                convo.initialize_tree_structure()
            except ValueError as e:
                if verbose:
                    warn(str(e))

        new_corpus_utts = []
        original_utt_to_convo_id = dict()

        for utt_id in new_convo_roots:
            orig_convo = self.get_conversation(self.get_utterance(utt_id).conversation_id)
            original_utt_to_convo_id[utt_id] = orig_convo.id
            try:
                subtree = orig_convo.get_subtree(utt_id)
                new_root_utt = subtree.utt
                new_root_utt.reply_to = None
                subtree_utts = [node.utt for node in subtree.bfs_traversal()]
                for utt in subtree_utts:
                    utt.conversation_id = utt_id
                new_corpus_utts.extend(subtree_utts)
            except ValueError:
                continue

        new_corpus = Corpus(utterances=new_corpus_utts)

        if preserve_corpus_meta:
            new_corpus.meta.update(self.meta)

        if preserve_convo_meta:
            for convo in new_corpus.iter_conversations():
                convo.meta['original_convo_meta'] = self.get_conversation(original_utt_to_convo_id[convo.id]).meta
                convo.meta['original_convo_id'] = original_utt_to_convo_id[convo.id]
        if verbose:
            missing_convo_roots = list(set(new_convo_roots) - set(new_corpus.get_conversation_ids()))
            if len(missing_convo_roots) > 0:
                warn("Failed to find some of the specified new convo roots:\n")
                print(missing_convo_roots)

        return new_corpus

    def get_meta(self) -> Dict:
        return self.meta

    def add_meta(self, key: str, value) -> None:
        self.meta[key] = value

    def speaking_pairs(self, selector: Optional[Callable[[Speaker, Speaker], bool]] = lambda speaker1, speaker2: True,
                       speaker_ids_only: bool = False) -> Set[Tuple[str, str]]:
        """
        Get all directed speaking pairs (a, b) of speakers such that a replies to b at least once in the dataset.

        :param selector: optional function that takes in a Speaker and a replied-to Speaker and returns True to include
            the pair in the result, or False otherwise.
        :param speaker_ids_only: if True, return just pairs of speaker names rather than speaker objects.
        :type speaker_ids_only: bool

        :return: Set containing all speaking pairs selected by the selector function, or all speaking pairs in the
            dataset if no selector function was used.
        """
        pairs = set()
        for utt2 in self.iter_utterances():
            if utt2.speaker is not None and utt2.reply_to is not None and self.has_utterance(utt2.reply_to):
                utt1 = self.get_utterance(utt2.reply_to)
                if utt1.speaker is not None:
                    if selector(utt2.speaker, utt1.speaker):
                        pairs.add((utt2.speaker.id, utt1.speaker.id) if
                                  speaker_ids_only else (utt2.speaker, utt1.speaker))
        return pairs

    def directed_pairwise_exchanges(self, selector: Optional[Callable[[Speaker, Speaker], bool]] = lambda speaker1, speaker2: True,
                                    speaker_ids_only: bool = False) -> Dict[Tuple, List[Utterance]]:
        """
        Get all directed pairwise exchanges in the dataset.

        :param selector: optional function that takes in a speaking speaker and a replied-to speaker and 
            returns True to include the pair in the result, or False otherwise.
        :param speaker_ids_only: if True, index conversations
            by speaker ids rather than Speaker objects.
        :type speaker_ids_only: bool

        :return: Dictionary mapping (speaker, target) tuples to a list of
            utterances given by the speaker in reply to the target.
        """
        pairs = defaultdict(list)
        for u2 in self.iter_utterances():
            if u2.speaker is not None and u2.reply_to is not None:
                u1 = self.get_utterance(u2.reply_to)
                if u1.speaker is not None:
                    if selector(u2.speaker, u1.speaker):
                        key = (u2.speaker.id, u1.speaker.id) if speaker_ids_only else (u2.speaker, u1.speaker)
                        pairs[key].append(u2)

        return pairs

    @staticmethod
    def _merge_utterances(utts1: List[Utterance], utts2: List[Utterance], warnings: bool) -> ValuesView[Utterance]:
        """
        Helper function for merge().

        Combine two collections of utterances into a single dictionary of Utterance id -> Utterance.

        If metadata of utterances in the two collections share the same key, but different values,
        the second collections' utterance metadata will be used.

        May mutate original collections of Utterances.

        Prints warnings when:
        1) Utterances with same id from this and other collection do not share the same data
        2) Utterance metadata has different values for the same key, and overwriting occurs

        :param utts1: First collection of Utterances
        :param utts2: Second collection of Utterances
        :param warnings: whether to print warnings when conflicting data is found.
        :return: ValuesView for merged set of utterances
        """
        seen_utts = dict()

        # Merge UTTERANCE metadata
        # Add all the utterances from this corpus
        for utt in utts1:
            seen_utts[utt.id] = utt

        # Add all the utterances from the other corpus, checking for data sameness and updating metadata as appropriate
        for utt in utts2:
            if utt.id in seen_utts:
                prev_utt = seen_utts[utt.id]
                if prev_utt == utt:
                    # other utterance metadata is ignored if data is not matched
                    for key, val in utt.meta.items():
                        if key in prev_utt.meta and prev_utt.meta[key] != val:
                            if warnings:
                                warn("Found conflicting values for Utterance {} for metadata key: {}. "
                                     "Overwriting with other corpus's Utterance metadata.".format(repr(utt.id),
                                                                                                  repr(key)))
                        prev_utt.meta[key] = val
                else:
                    if warnings:
                        warn("Utterances with same id do not share the same data:\n" +
                             str(prev_utt) + "\n" +
                             str(utt) + "\n" +
                             "Ignoring second corpus's utterance."
                             )
            else:
                seen_utts[utt.id] = utt

        return seen_utts.values()

    @staticmethod
    def _collect_speaker_data(utt_sets: Collection[Collection[Utterance]]) -> Tuple[
        Dict[str, Dict[str, str]], Dict[str, Dict[str, bool]]]:
        """
        Helper function for merge().

        Iterates through the input set of utterances, to collect Speaker data and metadata.

        Collect Speaker metadata in another Dictionary indexed by Speaker ID

        Track if conflicting speaker metadata is found in another dictionary

        :param utt_sets: Collections of collections of Utterances to extract Speakers from
        :return: speaker metadata and the corresponding tracker
        """
        # Collect SPEAKER data and metadata
        speakers_meta = defaultdict(lambda: defaultdict(str))
        speakers_meta_conflict = defaultdict(lambda: defaultdict(bool))
        for utt_set in utt_sets:
            for utt in utt_set:
                for meta_key, meta_val in utt.speaker.meta.items():
                    curr = speakers_meta[utt.speaker][meta_key]
                    if curr != meta_val:
                        if curr != "":
                            speakers_meta_conflict[utt.speaker][meta_key] = True
                        speakers_meta[utt.speaker][meta_key] = meta_val

        return speakers_meta, speakers_meta_conflict

    @staticmethod
    def _update_corpus_speaker_data(new_corpus, speakers_meta: Dict, speakers_meta_conflict: Dict,
                                    warnings: bool) -> None:
        """
        Helper function for merge().

        Update new_corpus's Speakers' data (utterance and conversation lists) and metadata

        Prints a warning if multiple values are found for any speaker's metadata key; latest speaker metadata is used

        :param speakers_meta: Dictionary indexed by Speaker ID, containing the collected Speaker metadata
        :param speakers_meta_conflict: Dictionary indexed by Speaker ID, indicating if there were value conflicts for the associated meta keys
        :return: None (mutates the new_corpus's Speakers)
        """
        # Update SPEAKER data and metadata with merged versions
        for speaker in new_corpus.iter_speakers():
            for meta_key, meta_val in speakers_meta[speaker].items():
                if speakers_meta_conflict[speaker][meta_key]:
                    if warnings:
                        warn("Multiple values found for {} for metadata key: {}. "
                             "Taking the latest one found".format(speaker, repr(meta_key)))
                speaker.meta[meta_key] = meta_val

    def _reinitialize_index_helper(self, new_index, obj_type):
        """
        Helper for reinitializing the index of the different Corpus object types
        :param new_index: new ConvoKitIndex object
        :param obj_type: utterance, speaker, or conversation
        :return: None (mutates new_index)
        """
        for obj in self.iter_objs(obj_type):
            for key, value in obj.meta.items():
                ConvoKitMeta._check_type_and_update_index(new_index, obj_type, key, value)
            obj.meta.index = new_index

    def reinitialize_index(self):
        """
        Reinitialize the Corpus Index from scratch.

        :return: None (sets the .meta_index of Corpus and of the corpus component objects)
        """
        old_index = self.meta_index
        new_index = ConvoKitIndex(self)

        self._reinitialize_index_helper(new_index, "utterance")
        self._reinitialize_index_helper(new_index, "speaker")
        self._reinitialize_index_helper(new_index, "conversation")

        for key, value in self.meta.items():  # overall
            new_index.update_index('corpus', key, str(type(value)))

        new_index.version = old_index.version
        self.meta_index = new_index

    def merge(self, other_corpus, warnings: bool = True):
        """
        Merges this corpus with another corpus.

        Utterances with the same id must share the same data, otherwise the other corpus utterance data & metadata
        will be ignored. A warning is printed when this happens.

        If metadata of this corpus (or its conversations / utterances) shares a key with the metadata of the
        other corpus, the other corpus's metadata (or its conversations / utterances) values will be used. A warning
        is printed when this happens.

        May mutate original and other corpus in the process.

        (Updates internal ConvoKit Index to match post-merge state and uses this Corpus's version number.)

        :param other_corpus: Corpus
        :param warnings: print warnings when data conflicts are encountered
        :return: new Corpus constructed from combined lists of utterances
        """
        utts1 = list(self.iter_utterances())
        utts2 = list(other_corpus.iter_utterances())
        combined_utts = self._merge_utterances(utts1, utts2, warnings=warnings)
        new_corpus = Corpus(utterances=list(combined_utts))
        # Note that we collect Speakers from the utt sets directly instead of the combined utts, otherwise
        # differences in Speaker meta will not be registered for duplicate Utterances (because utts would be discarded
        # during merging)
        speakers_meta, speakers_meta_conflict = self._collect_speaker_data([utts1, utts2])
        Corpus._update_corpus_speaker_data(new_corpus, speakers_meta, speakers_meta_conflict, warnings=warnings)

        # Merge CORPUS metadata
        new_corpus.meta = self.meta
        for key, val in other_corpus.meta.items():
            if key in new_corpus.meta and new_corpus.meta[key] != val:
                if warnings:
                    warn("Found conflicting values for Corpus metadata key: {}. "
                         "Overwriting with other Corpus's metadata.".format(repr(key)))
            new_corpus.meta[key] = val

        # Merge CONVERSATION metadata
        convos1 = self.iter_conversations()
        convos2 = other_corpus.iter_conversations()

        for convo in convos1:
            new_corpus.get_conversation(convo.id).meta = convo.meta

        for convo in convos2:
            for key, val in convo.meta.items():
                curr_meta = new_corpus.get_conversation(convo.id).meta
                if key in curr_meta and curr_meta[key] != val:
                    if warnings:
                        warn("Found conflicting values for Conversation {} for metadata key: {}. "
                             "Overwriting with other corpus's Conversation metadata.".format(repr(convo.id), repr(key)))
                curr_meta[key] = val

        new_corpus.update_speakers_data()
        new_corpus.reinitialize_index()

        return new_corpus

    def add_utterances(self, utterances=List[Utterance], warnings: bool = False, with_checks=True):
        """
        Add utterances to the Corpus.

        If the corpus has utterances that share an id with an utterance in the input utterance list,

        Optional warnings will be printed:
        - if the utterances with same id do not share the same data (added utterance is ignored)
        - added utterances' metadata have the same key but different values (added utterance's metadata will overwrite)

        :param utterances: Utterances to be added to the Corpus
        :param warnings: set to True for warnings to be printed
        :param with_checks: set to True if checks on utterance and metadata overlaps are desired. Set to False if newly added utterances are guaranteed to be new and share the same set of metadata keys.
        :return: a Corpus with the utterances from this Corpus and the input utterances combined
        """
        if with_checks:
            helper_corpus = Corpus(utterances=utterances)
            return self.merge(helper_corpus, warnings=warnings)
        else:
            new_speakers = {u.speaker.id: u.speaker for u in utterances}
            new_utterances = {u.id: u for u in utterances}
            for speaker in new_speakers.values():
                speaker.owner = self
            for utt in new_utterances.values():
                utt.owner = self

            # update corpus speakers
            for new_speaker_id, new_speaker in new_speakers.items():
                if new_speaker_id not in self.speakers:
                    self.speakers[new_speaker_id] = new_speaker

            # update corpus utterances + (link speaker -> utt)
            for new_utt_id, new_utt in new_utterances.items():
                if new_utt_id not in self.utterances:
                    self.utterances[new_utt_id] = new_utt
                    self.speakers[new_utt.speaker.id]._add_utterance(new_utt)

            # update corpus conversations + (link convo <-> utt)
            new_convos = defaultdict(list)
            for utt in new_utterances.values():
                if utt.conversation_id in self.conversations:
                    self.conversations[utt.conversation_id]._add_utterance(utt)
                else:
                    new_convos[utt.conversation_id].append(utt.id)
            for convo_id, convo_utts in new_convos.items():
                new_convo = Conversation(owner=self, id=convo_id,
                                         utterances=convo_utts,
                                         meta=None)
                self.conversations[convo_id] = new_convo
                # (link speaker -> convo)
                new_convo_speaker = self.speakers[new_convo.get_utterance(convo_id).speaker.id]
                new_convo_speaker._add_conversation(new_convo)

        return self

    def update_speakers_data(self) -> None:
        """
        Updates the conversation and utterance lists of every Speaker in the Corpus

        :return: None
        """
        speakers_utts = defaultdict(list)
        speakers_convos = defaultdict(list)

        for utt in self.iter_utterances():
            speakers_utts[utt.speaker.id].append(utt)

        for convo in self.iter_conversations():
            for utt in convo.iter_utterances():
                speakers_convos[utt.speaker.id].append(convo)

        for speaker in self.iter_speakers():
            speaker.utterances = {utt.id: utt for utt in speakers_utts[speaker.id]}
            speaker.conversations = {convo.id: convo for convo in speakers_convos[speaker.id]}

    def print_summary_stats(self) -> None:
        """
        Helper function for printing the number of Speakers, Utterances, and Conversations in this Corpus

        :return: None
        """
        print("Number of Speakers: {}".format(len(self.speakers)))
        print("Number of Utterances: {}".format(len(self.utterances)))
        print("Number of Conversations: {}".format(len(self.conversations)))

    def delete_metadata(self, obj_type: str, attribute: str):
        """
        Delete a specified metadata attribute from all Corpus components of the specified object type.

        Note that cancelling this method before it runs to completion may lead to errors in the Corpus.

        :param obj_type: 'utterance', 'conversation', 'speaker'
        :param attribute: name of metadata attribute
        :return: None
        """
        self.meta_index.lock_metadata_deletion[obj_type] = False
        for obj in self.iter_objs(obj_type):
            if attribute in obj.meta:
                del obj.meta[attribute]
        self.meta_index.del_from_index(obj_type, attribute)
        self.meta_index.lock_metadata_deletion[obj_type] = True

    def set_vector_matrix(self, name: str, matrix, ids: List[str]=None, columns: List[str] = None):
        """
        Adds a vector matrix to the Corpus, where the matrix is an array of vector representations of some
        set of Corpus components (i.e. Utterances, Conversations, Speakers).

        A ConvoKitMatrix object is initialized from the arguments and stored in the Corpus.

        :param name: descriptive name for the matrix
        :param matrix: numpy or scipy array matrix
        :param ids: optional list of Corpus component object ids, where each id corresponds to each row of the matrix
        :param columns: optional list of names for the columns of the matrix
        :return: None
        """

        matrix = ConvoKitMatrix(name=name, matrix=matrix, ids=ids, columns=columns)
        if name in self.meta_index.vectors:
            warn('Vector matrix "{}" already exists. Overwriting it with newly set vector matrix.'.format(name))
        self.meta_index.add_vector(name)
        self._vector_matrices[name] = matrix

    def append_vector_matrix(self, matrix: ConvoKitMatrix):
        """
        Adds an already constructed ConvoKitMatrix to the Corpus.

        :param matrix: a ConvoKitMatrix object
        :return: None
        """
        if matrix.name in self.meta_index.vectors:
            warn('Vector matrix "{}" already exists. '
                 "Overwriting it with newly appended vector matrix that has name: '{}'.".format(matrix.name, matrix.name))
        self.meta_index.add_vector(matrix.name)
        self._vector_matrices[matrix.name] = matrix

    def get_vector_matrix(self, name):
        """
        Gets the ConvoKitMatrix stored in the corpus as `name`.
        Returns None if no such matrix exists.

        :param name: name of the vector matrix
        :return: a ConvoKitMatrix object
        """
        # This is the lazy load step
        if name in self.vectors and name not in self._vector_matrices:
            matrix = ConvoKitMatrix.from_dir(self.corpus_dirpath, name)
            if matrix is not None:
                self._vector_matrices[name] = matrix
        return self._vector_matrices[name]

    def get_vectors(self, name, ids: Optional[List[str]] = None, columns: Optional[List[str]] = None,
                    as_dataframe: bool = False):
        """
        Get the vectors for some corpus component objects.

        :param name: name of the vector matrix
        :param ids: optional list of object ids to get vectors for; all by default
        :param columns: optional list of named columns of the vector to include; all by default
        :param as_dataframe: whether to return the vector as a dataframe (True) or in its raw array form (False). False
            by default.
        :return: a vector matrix (either np.ndarray or csr_matrix) or a pandas dataframe
        """
        return self.get_vector_matrix(name).get_vectors(ids=ids, columns=columns, as_dataframe=as_dataframe)

    def delete_vector_matrix(self, name):
        """
        Deletes the vector matrix stored under `name`.

        :param name: name of the vector mtrix
        :return: None
        """
        self.meta_index.vectors.remove(name)
        if name in self._vector_matrices:
            del self._vector_matrices[name]

    def dump_vectors(self, name, dir_name=None):

        if (self.corpus_dirpath is None) and (dir_name is None):
            raise ValueError('Must specify a directory to read from.')
        if dir_name is None:
            dir_name = self.corpus_dirpath

        self.get_vector_matrix(name).dump(dir_name)

    def load_info(self, obj_type, fields=None, dir_name=None):
        """
        loads attributes of objects in a corpus from disk.
        This function, along with dump_info, supports cases where a particular attribute is to be stored separately from
        the other corpus files, for organization or efficiency. These attributes will not be read when the corpus is
        initialized; rather, they can be loaded on-demand using this function.

        For each attribute with name <NAME>, will read from a file called info.<NAME>.jsonl, and load each attribute
        value into the respective object's .meta field.

        :param obj_type: type of object the attribute is associated with. can be one of "utterance", "speaker", "conversation".
        :param fields: a list of names of attributes to load. if empty, will load all attributes stored in the specified directory dir_name.
        :param dir_name: the directory to read attributes from. by default, or if set to None, will read from the directory that the Corpus was loaded from.
        :return: None
        """
        if fields is None:
            fields = []

        if (self.corpus_dirpath is None) and (dir_name is None):
            raise ValueError('Must specify a directory to read from.')
        if dir_name is None:
            dir_name = self.corpus_dirpath

        if len(fields) == 0:
            fields = [x.replace('info.', '').replace('.jsonl', '') for x in os.listdir(dir_name)
                      if x.startswith('info')]

        for field in fields:
            # self.aux_info[field] = self.load_jsonlist_to_dict(
            #     os.path.join(dir_name, 'feat.%s.jsonl' % field))
            getter = lambda oid: self.get_object(obj_type, oid)
            entries = load_jsonlist_to_dict(os.path.join(dir_name, 'info.%s.jsonl' % field))
            for k, v in entries.items():
                try:
                    obj = getter(k)
                    obj.set_info(field, v)
                except:
                    continue

    def dump_info(self, obj_type, fields, dir_name=None):
        """
        writes attributes of objects in a corpus to disk.
        This function, along with load_info, supports cases where a particular attribute is to be stored separately from the other corpus files, for organization or efficiency. These attributes will not be read when the corpus is initialized; rather, they can be loaded on-demand using this function.

        For each attribute with name <NAME>, will write to a file called info.<NAME>.jsonl, where rows are json-serialized dictionaries structured as {"id": id of object, "value": value of attribute}.

        :param obj_type: type of object the attribute is associated with. can be one of "utterance", "speaker", "conversation".
        :param fields: a list of names of attributes to write to disk.
        :param dir_name: the directory to write attributes to. by default, or if set to None, will read from the directory that the Corpus was loaded from.
        :return: None
        """

        if (self.corpus_dirpath is None) and (dir_name is None):
            raise ValueError('must specify a directory to write to')

        if dir_name is None:
            dir_name = self.corpus_dirpath
        # if len(fields) == 0:
        #     fields = self.aux_info.keys()
        for field in fields:
            # if field not in self.aux_info:
            #     raise ValueError("field %s not in index" % field)
            iterator = self.iter_objs(obj_type)
            entries = {obj.id: obj.retrieve_meta(field) for obj in iterator}
            # self.dump_jsonlist_from_dict(self.aux_info[field],
            #     os.path.join(dir_name, 'feat.%s.jsonl' % field))
            dump_jsonlist_from_dict(entries, os.path.join(dir_name, 'info.%s.jsonl' % field))

    # def load_vector_reprs(self, field, dir_name=None):
    #     """
    #     reads vector representations of Corpus objects from disk.
    #
    #     Will read matrices from a file called vect_info.<field>.npy and corresponding object IDs from a file called vect_info.<field>.keys,
    #
    #     :param field: the name of the representation
    #     :param dir_name: the directory to read from; by default, or if set to None, will read from the directory that the Corpus was loaded from.
    #     :return: None
    #     """
    #
    #     if (self.corpus_dirpath is None) and (dir_name is None):
    #         raise ValueError('must specify a directory to read from')
    #     if dir_name is None:
    #         dir_name = self.corpus_dirpath
    #
    #     self._vector_matrices[field] = self._load_vectors(
    #         os.path.join(dir_name, 'vect_info.' + field)
    #     )
    #
    # def dump_vector_reprs(self, field, dir_name=None):
    #     """
    #     writes vector representations of Corpus objects to disk.
    #
    #     Will write matrices to a file called vect_info.<field>.npy and corresponding object IDs to a file called vect_info.<field>.keys,
    #
    #     :param field: the name of the representation to write to disk
    #     :param dir_name: the directory to write to. by default, or if set to None, will read from the directory that the Corpus was loaded from.
    #     :return: None
    #     """
    #
    #     if (self.corpus_dirpath is None) and (dir_name is None):
    #         raise ValueError('must specify a directory to write to')
    #
    #     if dir_name is None:
    #         dir_name = self.corpus_dirpath
    #
    #     self._dump_vectors(self._vector_matrices[field], os.path.join(dir_name, 'vect_info.' + field))

    def get_attribute_table(self, obj_type, attrs):
        """
        returns a DataFrame, indexed by the IDs of objects of `obj_type`, containing attributes of these objects.

        :param obj_type: the type of object to get attributes for. can be `'utterance'`, `'speaker'` or `'conversation'`.
        :param attrs: a list of names of attributes to get.
        :return: a Pandas DataFrame of attributes.
        """
        iterator = self.iter_objs(obj_type)

        table_entries = []
        for obj in iterator:
            entry = dict()
            entry['id'] = obj.id
            for attr in attrs:
                entry[attr] = obj.retrieve_meta(attr)
            table_entries.append(entry)
        return pd.DataFrame(table_entries).set_index('id')

    def set_speaker_convo_info(self, speaker_id, convo_id, key, value):
        """
        assigns speaker-conversation attribute `key` with `value` to speaker `speaker_id` in conversation `convo_id`.

        :param speaker_id: speaker
        :param convo_id: conversation
        :param key: name of attribute
        :param value: value of attribute
        :return: None
        """

        speaker = self.get_speaker(speaker_id)
        if 'conversations' not in speaker.meta:
            speaker.meta['conversations'] = {}
        if convo_id not in speaker.meta['conversations']:
            speaker.meta['conversations'][convo_id] = {}
        speaker.meta['conversations'][convo_id][key] = value

    def get_speaker_convo_info(self, speaker_id, convo_id, key=None):
        """
        retreives speaker-conversation attribute `key` for `speaker_id` in conversation `convo_id`.

        :param speaker_id: speaker
        :param convo_id: conversation
        :param key: name of attribute. if None, will return all attributes for that speaker-conversation.
        :return: attribute value
        """

        speaker = self.get_speaker(speaker_id)
        if 'conversations' not in speaker.meta: return None
        if key is None:
            return speaker.meta['conversations'].get(convo_id, {})
        return speaker.meta['conversations'].get(convo_id, {}).get(key)

    def organize_speaker_convo_history(self, utterance_filter=None):
        """
        For each speaker, pre-computes a list of all of their utterances, organized by the conversation they participated in. Annotates speaker with the following:
            * `n_convos`: number of conversations
            * `start_time`: time of first utterance, across all conversations
            * `conversations`: a dictionary keyed by conversation id, where entries consist of:
                * `idx`: the index of the conversation, in terms of the time of the first utterance contributed by that particular speaker (i.e., `idx=0` means this is the first conversation the speaker ever participated in)
                * `n_utterances`: the number of utterances the speaker contributed in the conversation
                * `start_time`: the timestamp of the speaker's first utterance in the conversation
                * `utterance_ids`: a list of ids of utterances contributed by the speaker, ordered by timestamp.
            In case timestamps are not provided with utterances, the present behavior is to sort just by utterance id.

        :param utterance_filter: function that returns True for an utterance that counts towards a speaker having participated in that conversation. (e.g., one could filter out conversations where the speaker contributed less than k words per utterance)
        """

        if utterance_filter is None:
            utterance_filter = lambda x: True
        else:
            utterance_filter = utterance_filter

        speaker_to_convo_utts = defaultdict(lambda: defaultdict(list))
        for utterance in self.iter_utterances():
            if not utterance_filter(utterance): continue

            speaker_to_convo_utts[utterance.speaker.id][utterance.conversation_id].append(
                (utterance.id, utterance.timestamp))
        for speaker, convo_utts in speaker_to_convo_utts.items():
            for convo, utts in convo_utts.items():
                sorted_utts = sorted(utts, key=lambda x: (x[1], x[0]))
                self.set_speaker_convo_info(speaker, convo, 'utterance_ids', [x[0] for x in sorted_utts])
                self.set_speaker_convo_info(speaker, convo, 'start_time', sorted_utts[0][1])
                self.set_speaker_convo_info(speaker, convo, 'n_utterances', len(sorted_utts))
        for speaker in self.iter_speakers():
            try:
                speaker.set_info('n_convos', len(speaker.retrieve_meta('conversations')))
            except:
                continue

            sorted_convos = sorted(speaker.retrieve_meta('conversations').items(),
                                   key=lambda x: (x[1]['start_time'], x[1]['utterance_ids'][0]))
            speaker.set_info('start_time', sorted_convos[0][1]['start_time'])
            for idx, (convo_id, _) in enumerate(sorted_convos):
                self.set_speaker_convo_info(speaker.id, convo_id, 'idx', idx)

    def get_speaker_convo_attribute_table(self, attrs):
        """
        returns a table where each row lists a (speaker, convo) level aggregate for each attribute in attrs.

        :param attrs: list of (speaker, convo) attribute names
        :return: DataFrame containing all speaker,convo attributes.
        """

        table_entries = []
        for speaker in self.iter_speakers():

            if 'conversations' not in speaker.meta: continue
            for convo_id, convo_dict in speaker.meta['conversations'].items():
                entry = {'id': '%s__%s' % (speaker.id, convo_id),
                         'speaker': speaker.id, 'convo_id': convo_id,
                         'convo_idx': convo_dict['idx']}

                for attr in attrs:
                    entry[attr] = convo_dict.get(attr, None)
                table_entries.append(entry)
        return pd.DataFrame(table_entries).set_index('id')

    def get_full_attribute_table(self, speaker_convo_attrs, speaker_attrs=None, convo_attrs=None,
                                 speaker_suffix='__speaker',
                                 convo_suffix='__convo'):
        """
        Returns a table where each row lists a (speaker, convo) level aggregate for each attribute in attrs,
        along with speaker-level and conversation-level attributes; by default these attributes are suffixed with
        '__speaker' and '__convo' respectively.

        :param speaker_convo_attrs: list of (speaker, convo) attribute names
        :param speaker_attrs: list of speaker attribute names
        :param convo_attrs: list of conversation attribute names
        :param speaker_suffix: suffix to append to names of speaker-level attributes
        :param convo_suffix: suffix to append to names of conversation-level attributes.
        :return: DataFrame containing all attributes.
        """
        if speaker_attrs is None:
            speaker_attrs = []
        if convo_attrs is None:
            convo_attrs = []

        uc_df = self.get_speaker_convo_attribute_table(speaker_convo_attrs)
        u_df = self.get_attribute_table('speaker', speaker_attrs)
        u_df.columns = [x + speaker_suffix for x in u_df.columns]
        c_df = self.get_attribute_table('conversation', convo_attrs)
        c_df.columns = [x + convo_suffix for x in c_df.columns]
        return uc_df.join(u_df, on='speaker').join(c_df, on='convo_id')

    @staticmethod
    def from_pandas(speakers_df: DataFrame, utterances_df: DataFrame, conversations_df: DataFrame) -> 'Corpus':
        
        """
        Generates a Corpus from speakers, utterances and conversations dataframes.
        If the 'id' column is absent, the dataframe index will be used as the id.
         
        Arguments:
            speakers_df {DataFrame} -- speakers data, in a pandas Dataframe, with metadata optional
            utterances_df {DataFrame} -- utterances data, in a pandas Dataframe. All primary data fields expected, with metadata optional
            conversations_df {DataFrame} -- conversations data, in a pandas Dataframe, with metadata optional
        
        Returns:
            Corpus -- the generated corpus
        """
        #dict containing all primary fields expected in utterance dataframe
        columns = ['speaker', 'id', 'timestamp', 'conversation_id', 'reply_to', 'text']

        for (df_type, df) in [('utterances', utterances_df), ('conversations', conversations_df),
                              ('speakers', speakers_df)]:
            if 'id' not in df.columns:
                print(f'ID column is not present in {df_type} dataframe, generated ID column from dataframe index...')
            df['id'] = df.index

        #checking if dataframes contain their respective required columns
        assert pd.Series(columns).isin(utterances_df.columns).all(), "Utterances dataframe must contain all primary data fields"

        utterance_meta_cols = extract_meta_from_df(utterances_df)
        speaker_meta_cols = extract_meta_from_df(speakers_df)
        speakers_df.columns = [col.replace('meta.', '') for col in speakers_df.columns]

        utterance_list = []
        for index, row in tqdm(utterances_df.iterrows()):
            
            # extracting utterance metadata
            if utterance_meta_cols:
                metadata = {}
                for meta_col in utterance_meta_cols:
                    metadata[meta_col] = row['meta.' + meta_col]
            else: 
                metadata = None
            
            # extracting speaker metadata from speakers_df
            speaker_meta = speakers_df[speakers_df['id'] == row['speaker']][speaker_meta_cols].to_dict(orient='records')[0] if speaker_meta_cols else None
            
            # adding utterance in utterance list
            utterance_list.append(Utterance(id=str(row['id']), speaker=Speaker(id=row['speaker'], meta=speaker_meta),
                                            conversation_id=row['conversation_id'], reply_to=row['reply_to'],
                                            timestamp=row['timestamp'], text=row['text'],
                                            meta=metadata))
        # initializing corpus using utterance_list
        corpus_from_pandas = Corpus(utterances=utterance_list)
        
        # updating conversation metadata in corpus
        corpus_from_pandas = add_conv_meta_df(conversations_df, corpus_from_pandas)
        
        return corpus_from_pandas

# def __repr__(self):
# def __eq__(self, other):
#     return True
