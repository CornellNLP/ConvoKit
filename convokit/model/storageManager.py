from typing import Optional
from abc import ABCMeta, abstractmethod
from pymongo import MongoClient
from pymongo.database import Database
import bson


class StorageManager(metaclass=ABCMeta):
    """
    Abstraction layer for the concrete representation of data and metadata
    within corpus components (e.g., Utterance text and timestamps). All requests
    to access or modify corpusComponent fields (with the exception of ID) are
    actually routed through one of StorageManager's concrete subclasses. Each
    subclass implements a storage backend that contains the actual data.
    """

    def __init__(self):
        # concrete data storage (i.e., collections) for each component type
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
    def get_data(self, component_type: str, component_id: str, property_name: Optional[str] = None):
        """
        Retrieve the property data for the component of type component_type with
        id component_id. If property_name is specified return only the data for
        that property, otherwise return the dict containing all properties.
        """
        return NotImplemented

    @abstractmethod
    def update_data(self, component_type: str, component_id: str, property_name: str, new_value):
        """
        Set or update the property data for the component of type component_type
        with id component_id
        """
        return NotImplemented

    @abstractmethod
    def delete_data(
        self, component_type: str, component_id: str, property_name: Optional[str] = None
    ):
        """
        Delete a data entry from this StorageManager for the component of type
        component_type with id component_id. If property_name is specified
        delete only that property, otherwise delete the entire entry.
        """
        return NotImplemented

    @abstractmethod
    def clear_all_data(self):
        """
        Erase all data from this StorageManager (i.e., reset self.data to its
        initial empty state; Python will garbage-collect the now-unreferenced
        old data entries). This is used for cleanup after destructive Corpus
        operations.
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
        Compare the entries in this StorageManager to the existing component ids
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


class MemStorageManager(StorageManager):
    """
    Concrete StorageManager implementation for in-memory data storage.
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

    def get_data(self, component_type: str, component_id: str, property_name: Optional[str] = None):
        collection = self.get_collection(component_type)
        if component_id not in collection:
            raise KeyError(
                f"This StorageManager does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            return collection[component_id]
        else:
            return collection[component_id][property_name]

    def update_data(self, component_type: str, component_id: str, property_name: str, new_value):
        collection = self.get_collection(component_type)
        # don't create new collections if the ID is not found; this is supposed to be handled in the
        # CorpusComponent constructor so if the ID is missing that indicates something is wrong
        if component_id not in collection:
            raise KeyError(
                f"This StorageManager does not have an entry for the {component_type} with id {component_id}."
            )
        collection[component_id][property_name] = new_value

    def delete_data(
        self, component_type: str, component_id: str, property_name: Optional[str] = None
    ):
        collection = self.get_collection(component_type)
        if component_id not in collection:
            raise KeyError(
                f"This StorageManager does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            del collection[component_id]
        else:
            del collection[component_id][property_name]

    def clear_all_data(self):
        for key in self.data:
            self.data[key] = {}


class DBStorageManager(StorageManager):
    """
    Concrete StorageManager implementation for database-backed data storage.
    Collections are implemented as MongoDB collections.
    """

    def __init__(self, collection_prefix, db_host: Optional[str] = None):
        super().__init__()

        self.collection_prefix = collection_prefix
        self.client = MongoClient(db_host)
        self.db = self.client["convokit"]

        # initialize component collections as MongoDB collections in the convokit db
        for key in self.data:
            self.data[key] = self.db[self._get_collection_name(key)]

    def _get_collection_name(self, component_type: str) -> str:
        return f"{self.collection_prefix}_{component_type}"

    def get_collection_ids(self, component_type: str):
        # from StackOverflow: get all keys in a MongoDB collection
        # https://stackoverflow.com/questions/2298870/get-names-of-all-keys-in-the-collection
        map = bson.Code("function() { for (var key in this) { emit(key, null); } }")
        reduce = bson.Code("function(key, stuff) { return null; }")
        result = self.db[self._get_collection_name(component_type)].map_reduce(
            map, reduce, "get_collection_ids_result"
        )
        return result.distinct("_id")

    def has_data_for_component(self, component_type: str, component_id: str) -> bool:
        collection = self.get_collection(component_type)
        lookup = collection.find_one({"_id": component_id})
        return lookup is not None

    def initialize_data_for_component(
        self, component_type: str, component_id: str, overwrite: bool = False, initial_value=None
    ):
        collection = self.get_collection(component_type)
        if overwrite or not self.has_data_for_component(component_type, component_id):
            data = initial_value if initial_value is not None else {}
            collection.update_one({"_id": component_id}, {"$set": data}, upsert=True)

    def get_data(self, component_type: str, component_id: str, property_name: Optional[str] = None):
        collection = self.get_collection(component_type)
        all_fields = collection.find_one({"_id": component_id})
        if all_fields is None:
            raise KeyError(
                f"This StorageManager does not have an entry for the {component_type} with id {component_id}."
            )
        if property_name is None:
            return all_fields
        else:
            return all_fields[property_name]

    def update_data(self, component_type: str, component_id: str, property_name: str, new_value):
        data = self.get_data(component_type, component_id)
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
