from typing import Optional, List
from abc import ABCMeta, abstractmethod
from pymongo import MongoClient
from pymongo.database import Database
import bson
import pickle


class BackendMapper(metaclass=ABCMeta):
    """
    Abstraction layer for the concrete representation of data and metadata
    within corpus components (e.g., Utterance text and timestamps). All requests
    to access or modify corpusComponent fields (with the exception of ID) are
    actually routed through one of BackendMapper's concrete subclasses. Each
    subclass implements a concrete backend mapping from ConvoKit operations to actual data.
    (These mappings are referred to as collections.)
    """

    def __init__(self):
        # concrete data backend (i.e., collections) for each component type
        # this will be assigned in subclasses
        self.data = {"utterance": None, "conversation": None, "speaker": None, "meta": None}

    @abstractmethod
    def get_collection_ids(self, component_type: str):
        """
        Returns a list of all object IDs within the component_type collection
        """
        return NotImplemented

    @abstractmethod
    def has_data_for_component(self, component_type: str, component_id: str) -> bool:
        """
        Check if there is an existing entry for the component of type component_type
        with id component_id
        """
        return NotImplemented

    @abstractmethod
    def initialize_data_for_component(
        self, component_type: str, component_id: str, overwrite: bool = False, initial_value=None
    ):
        """
        Create a blank entry for a component of type component_type with id
        component_id. Will avoid overwriting any existing data unless the
        overwrite parameter is set to True.
        """
        return NotImplemented

    @abstractmethod
    def get_data(
        self,
        component_type: str,
        component_id: str,
        property_name: Optional[str] = None,
        index=None,
    ):
        """
        Retrieve the property data for the component of type component_type with
        id component_id. If property_name is specified return only the data for
        that property, otherwise return the dict containing all properties.
        Additionally, the expected type of the property to be fetched may be specified
        as a string; this is meant to be used for metadata in conjunction with the index.
        """
        return NotImplemented

    @abstractmethod
    def update_data(
        self,
        component_type: str,
        component_id: str,
        property_name: str,
        new_value,
        index=None,
    ):
        """
        Set or update the property data for the component of type component_type
        with id component_id. For metadata, the Python object type may also be
        specified, to be used in conjunction with the index.
        """
        return NotImplemented

    @abstractmethod
    def delete_data(
        self, component_type: str, component_id: str, property_name: Optional[str] = None
    ):
        """
        Delete a data entry from this BackendMapper for the component of type
        component_type with id component_id. If property_name is specified
        delete only that property, otherwise delete the entire entry.
        """
        return NotImplemented

    @abstractmethod
    def clear_all_data(self):
        """
        Erase all data from this BackendMapper (i.e., reset self.data to its
        initial empty state; Python will garbage-collect the now-unreferenced
        old data entries). This is used for cleanup after destructive Corpus
        operations.
        """
        return NotImplemented

    @abstractmethod
    def count_entries(self, component_type: str):
        """
        Count the number of entries held for the specified component type by
        this BackendMapper instance
        """
        return NotImplemented

    def get_collection(self, component_type: str):
        if component_type not in self.data:
            raise ValueError(
                'component_type must be one of "utterance", "conversation", "speaker", or "meta".'
            )
        return self.data[component_type]

    def purge_obsolete_entries(self, utterance_ids, conversation_ids, speaker_ids, meta_ids):
        """
        Compare the entries in this BackendMapper to the existing component ids
        provided as parameters, and delete any entries that are not found in the
        parameter ids.
        """
        ref_ids = {
            "utterance": set(utterance_ids),
            "conversation": set(conversation_ids),
            "speaker": set(speaker_ids),
            "meta": set(meta_ids),
        }
        for obj_type in self.data:
            for obj_id in self.get_collection_ids(obj_type):
                if obj_id not in ref_ids[obj_type]:
                    self.delete_data(obj_type, obj_id)


class MemMapper(BackendMapper):
    """
    Concrete BackendMapper implementation for in-memory data storage.
    Collections are implemented as vanilla Python dicts.
    """

    def __init__(self):
        super().__init__()

        # initialize component collections as dicts
        for key in self.data:
            self.data[key] = {}

    def get_collection_ids(self, component_type: str):
        return list(self.get_collection(component_type).keys())

    def has_data_for_component(self, component_type: str, component_id: str) -> bool:
        collection = self.get_collection(component_type)
        return component_id in collection

    def initialize_data_for_component(
        self, component_type: str, component_id: str, overwrite: bool = False, initial_value=None
    ):
        collection = self.get_collection(component_type)
        if overwrite or not self.has_data_for_component(component_type, component_id):
            collection[component_id] = initial_value if initial_value is not None else {}

    def get_data(
        self,
        component_type: str,
        component_id: str,
        property_name: Optional[str] = None,
        index=None,
    ):
        collection = self.get_collection(component_type)
        if component_id not in collection:
            raise KeyError(
                f"This BackendMapper does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            return collection[component_id]
        else:
            return collection[component_id][property_name]

    def update_data(
        self,
        component_type: str,
        component_id: str,
        property_name: str,
        new_value,
        index=None,
    ):
        collection = self.get_collection(component_type)
        # don't create new collections if the ID is not found; this is supposed to be handled in the
        # CorpusComponent constructor so if the ID is missing that indicates something is wrong
        if component_id not in collection:
            raise KeyError(
                f"This BackendMapper does not have an entry for the {component_type} with id {component_id}."
            )
        collection[component_id][property_name] = new_value

    def delete_data(
        self, component_type: str, component_id: str, property_name: Optional[str] = None
    ):
        collection = self.get_collection(component_type)
        if component_id not in collection:
            raise KeyError(
                f"This BackendMapper does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            del collection[component_id]
        else:
            del collection[component_id][property_name]

    def clear_all_data(self):
        for key in self.data:
            self.data[key] = {}

    def count_entries(self, component_type: str):
        return len(self.get_collection(component_type))


class DBMapper(BackendMapper):
    """
    Concrete BackendMapper implementation for database-backed data storage.
    Collections are implemented as MongoDB collections.
    """

    def __init__(self, collection_prefix, db_host: Optional[str] = None):
        super().__init__()

        self.collection_prefix = collection_prefix
        self.client = MongoClient(db_host)
        self.db = self.client["convokit"]

        # this special lock is used for reconnecting to an existing DB, whereupon
        # it is known that all the data already exists and so the initialization
        # step can be skipped, greatly saving time
        self.bypass_init = False

        # initialize component collections as MongoDB collections in the convokit db
        for key in self.data:
            self.data[key] = self.db[self._get_collection_name(key)]

    def _get_collection_name(self, component_type: str) -> str:
        return f"{self.collection_prefix}_{component_type}"

    def get_collection_ids(self, component_type: str):
        return [
            doc["_id"]
            for doc in self.db[self._get_collection_name(component_type)].find(projection=["_id"])
        ]

    def has_data_for_component(self, component_type: str, component_id: str) -> bool:
        collection = self.get_collection(component_type)
        lookup = collection.find_one({"_id": component_id})
        return lookup is not None

    def initialize_data_for_component(
        self, component_type: str, component_id: str, overwrite: bool = False, initial_value=None
    ):
        if self.bypass_init:
            return
        collection = self.get_collection(component_type)
        if overwrite or not self.has_data_for_component(component_type, component_id):
            data = initial_value if initial_value is not None else {}
            collection.replace_one({"_id": component_id}, data, upsert=True)

    def get_data(
        self,
        component_type: str,
        component_id: str,
        property_name: Optional[str] = None,
        index=None,
    ):
        collection = self.get_collection(component_type)
        all_fields = collection.find_one({"_id": component_id})
        if all_fields is None:
            raise KeyError(
                f"This BackendMapper does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            # if some data is known to be binary type, unpack it
            if index is not None:
                for key in all_fields:
                    if index.get(key, None) == ["bin"]:
                        all_fields[key] = pickle.loads(all_fields[key])
            # do not include the MongoDB-specific _id field
            del all_fields["_id"]
            return all_fields
        else:
            result = all_fields[property_name]
            if index is not None and index.get(property_name, None) == ["bin"]:
                # binary data must be unpacked
                result = pickle.loads(result)
            return result

    def update_data(
        self,
        component_type: str,
        component_id: str,
        property_name: str,
        new_value,
        index=None,
    ):
        data = self.get_data(component_type, component_id)
        if index is not None and index.get(property_name, None) == ["bin"]:
            # non-serializable types must go through pickling then be encoded as bson.Binary
            new_value = bson.Binary(pickle.dumps(new_value))
        data[property_name] = new_value
        collection = self.get_collection(component_type)
        collection.update_one({"_id": component_id}, {"$set": data})

    def delete_data(
        self, component_type: str, component_id: str, property_name: Optional[str] = None
    ):
        collection = self.get_collection(component_type)
        if property_name is None:
            # delete the entire document
            collection.delete_one({"_id": component_id})
        else:
            # delete only the specified property
            collection.update_one({"_id": component_id}, {"$unset": {property_name: ""}})

    def clear_all_data(self):
        for key in self.data:
            self.data[key].drop()
            self.data[key] = self.db[self._get_collection_name(key)]

    def count_entries(self, component_type: str):
        return self.get_collection(component_type).estimated_document_count()
