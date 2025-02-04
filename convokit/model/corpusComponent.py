from typing import List, Optional

from convokit.util import warn
from .convoKitMeta import ConvoKitMeta


class CorpusComponent:
    def __init__(
        self,
        obj_type: str,
        owner=None,
        id=None,
        initial_data=None,
        vectors: List[str] = None,
        meta=None,
    ):
        self.obj_type = obj_type  # utterance, speaker, conversation
        self._owner = owner
        self._id = id
        self.vectors = vectors if vectors is not None else []

        # if the CorpusComponent is initialized with an owner set up an entry
        # in the owner's backend; if it is not initialized with an owner
        # (i.e. it is a standalone object) set up a dict-based temp backend
        if self.owner is None:
            self._temp_backend = initial_data if initial_data is not None else {}
        else:
            self.owner.backend_mapper.initialize_data_for_component(
                self.obj_type,
                self._id,
                initial_value=(initial_data if initial_data is not None else {}),
            )

        if meta is None:
            meta = dict()
        self._meta = self.init_meta(meta)

    def get_owner(self):
        return self._owner

    def set_owner(self, owner):
        if owner is self._owner:
            # no action needed
            return
        # stash the metadata first since reassigning self._owner will break its backend connection
        meta_vals = {k: v for k, v in self.meta.items()}
        previous_owner = self._owner
        self._owner = owner
        if owner is not None:
            # when a new owner Corpus is assigned, we must take the following steps:
            # (1) transfer this component's data to the new owner's BackendMapper
            # (2) avoid duplicates by removing the data from the old owner (or temp backend if there was no prior owner)
            # (3) reinitialize the metadata instance
            data_dict = (
                dict(previous_owner.backend_mapper.get_data(self.obj_type, self.id))
                if previous_owner is not None
                else self._temp_backend
            )
            self.owner.backend_mapper.initialize_data_for_component(
                self.obj_type, self.id, initial_value=data_dict
            )
            if previous_owner is not None:
                previous_owner.backend_mapper.delete_data(self.obj_type, self.id)
                previous_owner.backend_mapper.delete_data("meta", self.meta.backend_key)
            else:
                del self._temp_backend
            self._meta = self.init_meta(meta_vals)

    owner = property(get_owner, set_owner)

    def init_meta(self, meta, overwrite=False):
        if self._owner is None:
            # ConvoKitMeta instances are not allowed for ownerless (standalone)
            # components since they must be backed by a BackendMapper. In this
            # case we must forcibly convert the ConvoKitMeta instance to dict
            if isinstance(meta, ConvoKitMeta):
                meta = meta.to_dict()
            return meta
        else:
            if isinstance(meta, ConvoKitMeta) and meta.owner is self._owner:
                return meta
            ck_meta = ConvoKitMeta(self, self.owner.meta_index, self.obj_type, overwrite=overwrite)
            for key, value in meta.items():
                ck_meta[key] = value
            return ck_meta

    def get_id(self):
        return self._id

    def set_id(self, value):
        if not isinstance(value, str) and value is not None:
            self._id = str(value)
            warn(
                "{} id must be a string. ID input has been casted to a string.".format(
                    self.obj_type
                )
            )
        else:
            self._id = value

    id = property(get_id, set_id)

    def get_meta(self):
        return self._meta

    def set_meta(self, new_meta):
        self._meta = self.init_meta(new_meta, overwrite=True)

    meta = property(get_meta, set_meta)

    def get_data(self, property_name):
        if self._owner is None:
            return self._temp_backend[property_name]
        return self.owner.backend_mapper.get_data(self.obj_type, self.id, property_name)

    def set_data(self, property_name, value):
        if self._owner is None:
            self._temp_backend[property_name] = value
        else:
            self.owner.backend_mapper.update_data(self.obj_type, self.id, property_name, value)

    # def __eq__(self, other):
    #     if type(self) != type(other): return False
    #     # do not compare 'utterances' and 'conversations' in Speaker.__dict__. recursion loop will occur.
    #     self_keys = set(self.__dict__).difference(['_owner', 'meta', 'utterances', 'conversations'])
    #     other_keys = set(other.__dict__).difference(['_owner', 'meta', 'utterances', 'conversations'])
    #     return self_keys == other_keys and all([self.__dict__[k] == other.__dict__[k] for k in self_keys])

    def retrieve_meta(self, key: str):
        """
        Retrieves a value stored under the key of the metadata of corpus object

        :param key: name of metadata attribute
        :return: value
        """
        return self.meta.get(key, None)

    def add_meta(self, key: str, value) -> None:
        """
        Adds a key-value pair to the metadata of the corpus object

        :param key: name of metadata attribute
        :param value: value of metadata attribute
        :return: None
        """
        self.meta[key] = value

    def get_vector(
        self, vector_name: str, as_dataframe: bool = False, columns: Optional[List[str]] = None
    ):
        """
        Get the vector stored as `vector_name` for this object.

        :param vector_name: name of vector
        :param as_dataframe: whether to return the vector as a dataframe (True) or in its raw array form (False). False by default.
        :param columns: optional list of named columns of the vector to include. All columns returned otherwise. This parameter is only used if as_dataframe is set to True
        :return: a numpy / scipy array
        """
        if vector_name not in self.vectors:
            raise ValueError(
                "This {} has no vector stored as '{}'.".format(self.obj_type, vector_name)
            )

        return self.owner.get_vector_matrix(vector_name).get_vectors(
            ids=[self.id], as_dataframe=as_dataframe, columns=columns
        )

    def add_vector(self, vector_name: str):
        """
        Logs in the Corpus component object's internal vectors list that the component object has a vector row associated with it in the vector matrix named `vector_name`.

        Transformers that add vectors to the Corpus should use this to update the relevant component objects during the transform() step.

        :param vector_name: name of vector matrix
        :return: None
        """
        if vector_name not in self.vectors:
            self.vectors.append(vector_name)

    def has_vector(self, vector_name: str):
        return vector_name in self.vectors

    def delete_vector(self, vector_name: str):
        """
        Delete a vector associated with this Corpus component object.

        :param vector_name:
        :return: None
        """
        self.vectors.remove(vector_name)

    def to_dict(self):
        return {
            "id": self.id,
            "vectors": self.vectors,
            "meta": self.meta if type(self.meta) == dict else self.meta.to_dict(),
        }

    def __str__(self):
        return "{}(id: {}, vectors: {}, meta: {})".format(
            self.obj_type.capitalize(), self.id, self.vectors, self.meta
        )

    def __hash__(self):
        return hash(self.obj_type + str(self.id))

    def __repr__(self):
        copy = self.__dict__.copy()
        deleted_keys = [
            "utterances",
            "conversations",
            "user",
            "_root",
            "_utterance_ids",
            "_speaker_ids",
        ]
        for k in deleted_keys:
            if k in copy:
                del copy[k]

        to_delete = [k for k in copy if k.startswith("_")]
        to_add = {k[1:]: copy[k] for k in copy if k.startswith("_")}

        for k in to_delete:
            del copy[k]

        copy.update(to_add)

        try:
            return self.obj_type.capitalize() + "(" + str(copy) + ")"
        except (
            AttributeError
        ):  # for backwards compatibility when corpus objects are saved as binary data, e.g. wikiconv
            return "(" + str(copy) + ")"
