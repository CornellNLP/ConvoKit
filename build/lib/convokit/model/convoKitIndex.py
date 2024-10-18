from typing import Optional, Dict, List


class ConvoKitIndex:
    def __init__(
        self,
        owner,
        utterances_index: Optional[Dict[str, List[str]]] = None,
        speakers_index: Optional[Dict[str, List[str]]] = None,
        conversations_index: Optional[Dict[str, List[str]]] = None,
        overall_index: Optional[Dict[str, List[str]]] = None,
        vectors: Optional[List[str]] = None,
        version: Optional[int] = 0,
    ):
        self.owner = owner
        self.utterances_index = utterances_index if utterances_index is not None else {}
        self.speakers_index = speakers_index if speakers_index is not None else {}
        self.conversations_index = conversations_index if conversations_index is not None else {}
        self.overall_index = overall_index if overall_index is not None else {}
        self.indices = {
            "utterance": self.utterances_index,
            "conversation": self.conversations_index,
            "speaker": self.speakers_index,
            "corpus": self.overall_index,
        }
        self.vectors = set(vectors) if vectors is not None else set()
        self.version = version
        self.type_check = True  # toggle-able to enable/disable type checks on metadata additions
        self.lock_metadata_deletion = {"utterance": True, "conversation": True, "speaker": True}

    def create_new_index(self, obj_type: str, key: str):
        """
        Create a new entry in the obj_type index with a blank type list,
        representing an "Any" type which might be later refined.

        :param obj_type: utterance, conversation, or speaker
        :param key: string
        :param class_type: class type
        """
        if key not in self.indices[obj_type]:
            self.indices[obj_type][key] = []

    def update_index(self, obj_type: str, key: str, class_type: str):
        """
        Append the class_type to the index

        :param obj_type: utterance, conversation, or speaker
        :param key: string
        :param class_type: class type
        :return: None
        """
        assert type(key) == str
        assert "class" in class_type or class_type == "bin"
        if key not in self.indices[obj_type]:
            self.indices[obj_type][key] = []
        self.indices[obj_type][key].append(class_type)

    def set_index(self, obj_type: str, key: str, class_type: str):
        """
        Set the class_type of the index as [`class_type`].

        :param obj_type: utterance, conversation, or speaker
        :param key: string
        :param class_type: class type
        :return: None
        """
        assert type(key) == str
        assert "class" in class_type or class_type == "bin"
        self.indices[obj_type][key] = [class_type]

    def get_index(self, obj_type: str):
        return self.indices[obj_type]

    def del_from_index(self, obj_type: str, key: str):
        assert type(key) == str
        if key not in self.indices[obj_type]:
            return
        del self.indices[obj_type][key]
        #
        # corpus = self.owner
        # for corpus_obj in corpus.iter_objs(obj_type):
        #     if key in corpus_obj.meta:
        #         del corpus_obj.meta[key]

    def add_vector(self, vector_name):
        self.vectors.add(vector_name)

    def del_vector(self, vector_name):
        self.vectors.remove(vector_name)

    def update_from_dict(self, meta_index: Dict):
        self.conversations_index.update(meta_index["conversations-index"])
        self.utterances_index.update(meta_index["utterances-index"])
        speaker_index = "speakers-index" if "speakers-index" in meta_index else "users-index"
        self.speakers_index.update(meta_index[speaker_index])
        self.overall_index.update(meta_index["overall-index"])
        self.vectors = set(meta_index.get("vectors", set()))
        for index in self.indices.values():
            for k, v in index.items():
                if isinstance(v, str):
                    index[k] = [v]

        self.version = meta_index["version"]

    def to_dict(self, exclude_vectors: List[str] = None, force_version=None):
        retval = dict()
        retval["utterances-index"] = self.utterances_index
        retval["speakers-index"] = self.speakers_index
        retval["conversations-index"] = self.conversations_index
        retval["overall-index"] = self.overall_index

        if force_version is None:
            retval["version"] = self.version + 1
        else:
            retval["version"] = force_version

        if exclude_vectors is not None:
            retval["vectors"] = list(self.vectors - set(exclude_vectors))
        else:
            retval["vectors"] = list(self.vectors)

        return retval

    def enable_type_check(self):
        self.type_check = True

    def disable_type_check(self):
        self.type_check = False

    def __str__(self):
        return str(self.to_dict(force_version=self.version))

    def __repr__(self):
        return str(self)
