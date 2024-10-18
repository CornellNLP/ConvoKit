"""
Contains functions that help with the construction / dumping of a Corpus
"""

import json
import os
import pickle
from collections import defaultdict, deque
from typing import Dict, Optional, List, Iterable

import bson
from pymongo import UpdateOne

from convokit.util import warn, create_safe_id
from .conversation import Conversation
from .convoKitIndex import ConvoKitIndex
from .convoKitMeta import ConvoKitMeta
from .speaker import Speaker
from .backendMapper import BackendMapper, MemMapper, DBMapper
from .utterance import Utterance

BIN_DELIM_L, BIN_DELIM_R = "<##bin{", "}&&@**>"
KeyId = "id"
KeySpeaker = "speaker"
KeyConvoId = "conversation_id"
KeyReplyTo = "reply-to"
KeyTimestamp = "timestamp"
KeyText = "text"
DefinedKeys = {KeyId, KeySpeaker, KeyConvoId, KeyReplyTo, KeyTimestamp, KeyText}
KeyMeta = "meta"
KeyVectors = "vectors"

JSONLIST_BUFFER_SIZE = 1000


def get_corpus_id(
    db_collection_prefix: Optional[str], filename: Optional[str], backend: str
) -> Optional[str]:
    if db_collection_prefix is not None:
        # treat the unique collection prefix as the ID (even if a filename is specified)
        corpus_id = db_collection_prefix
    elif filename is not None:
        # automatically derive an ID from the file path
        corpus_id = os.path.basename(os.path.normpath(filename))
    else:
        corpus_id = None

    if backend == "db" and corpus_id is not None:
        compatibility_msg = check_id_for_mongodb(corpus_id)
        if compatibility_msg is not None:
            random_id = create_safe_id()
            warn(
                f'Attempting to use "{corpus_id}" as DB collection prefix failed because: {compatibility_msg}. Will instead use randomly generated prefix {random_id}.'
            )
            corpus_id = random_id

    return corpus_id


def check_id_for_mongodb(corpus_id):
    # List of collection name restrictions from official MongoDB docs:
    # https://www.mongodb.com/docs/manual/reference/limits/#mongodb-limit-Restriction-on-Collection-Names
    if "$" in corpus_id:
        return "contains the restricted character '$'"
    if len(corpus_id) == 0:
        return "string is empty"
    if "\0" in corpus_id:
        return "contains a null character"
    if "system." in corpus_id:
        return 'starts with the restricted prefix "system."'
    if not (corpus_id[0] == "_" or corpus_id[0].isalpha()):
        return "name must start with an underscore or letter character"
    return None


def get_corpus_dirpath(filename: str) -> Optional[str]:
    if filename is None:
        return None
    elif os.path.isdir(filename):
        return filename
    else:
        return os.path.dirname(filename)


def initialize_backend(
    corpus: "Corpus", backend_mapper: Optional[BackendMapper], backend: str, db_host: Optional[str]
):
    if backend_mapper is not None:
        return backend_mapper
    else:
        if backend == "mem":
            return MemMapper()
        elif backend == "db":
            if db_host is None:
                db_host = corpus.config.db_host
            return DBMapper(corpus.id, db_host)
        else:
            raise ValueError(
                f"Unrecognized setting '{backend}' for backend type; should be either 'mem' or 'db'."
            )


def load_utterance_info_from_dir(
    dirname, utterance_start_index, utterance_end_index, exclude_utterance_meta
):
    assert dirname is not None
    assert os.path.isdir(dirname)

    if utterance_start_index is None:
        utterance_start_index = 0
    if utterance_end_index is None:
        utterance_end_index = float("inf")

    if os.path.exists(os.path.join(dirname, "utterances.jsonl")):
        with open(os.path.join(dirname, "utterances.jsonl"), "r") as f:
            utterances = []
            idx = 0
            for line in f:
                if utterance_start_index <= idx <= utterance_end_index:
                    utterances.append(json.loads(line))
                idx += 1

    elif os.path.exists(os.path.join(dirname, "utterances.json")):
        with open(os.path.join(dirname, "utterances.json"), "r") as f:
            utterances = json.load(f)

    if exclude_utterance_meta:
        for utt in utterances:
            for field in exclude_utterance_meta:
                del utt["meta"][field]

    return utterances


def load_speakers_data_from_dir(filename, exclude_speaker_meta):
    speaker_file = "speakers.json" if "speakers.json" in os.listdir(filename) else "users.json"
    with open(os.path.join(filename, speaker_file), "r") as f:
        id_to_speaker_data = json.load(f)

        if (
            len(id_to_speaker_data) > 0
            and len(next(iter(id_to_speaker_data.values())))
            and "vectors" in id_to_speaker_data == 2
        ):
            # has vectors data
            for _, speaker_data in id_to_speaker_data.items():
                for k in exclude_speaker_meta:
                    if k in speaker_data["meta"]:
                        del speaker_data["meta"][k]
        else:
            for _, speaker_data in id_to_speaker_data.items():
                for k in exclude_speaker_meta:
                    if k in speaker_data:
                        del speaker_data[k]
    return id_to_speaker_data


def load_convos_data_from_dir(filename, exclude_conversation_meta):
    """

    :param filename:
    :param exclude_conversation_meta:
    :return: a mapping from convo id to convo meta
    """
    with open(os.path.join(filename, "conversations.json"), "r") as f:
        id_to_convo_data = json.load(f)

        if (
            len(id_to_convo_data) > 0
            and len(next(iter(id_to_convo_data.values())))
            and "vectors" in id_to_convo_data == 2
        ):
            # has vectors data
            for _, convo_data in id_to_convo_data.items():
                for k in exclude_conversation_meta:
                    if k in convo_data["meta"]:
                        del convo_data["meta"][k]
        else:
            for _, convo_data in id_to_convo_data.items():
                for k in exclude_conversation_meta:
                    if k in convo_data:
                        del convo_data[k]
    return id_to_convo_data


def load_corpus_meta_from_dir(filename, corpus_meta, exclude_overall_meta):
    """
    Updates corpus meta object with fields from corpus.json
    """
    with open(os.path.join(filename, "corpus.json"), "r") as f:
        for k, v in json.load(f).items():
            if k in exclude_overall_meta:
                continue
            corpus_meta[k] = v


def unpack_binary_data_for_utts(utterances, filename, utterance_index, exclude_meta, KeyMeta):
    """

    :param utterances: mapping from utterance id to {'meta': ..., 'vectors': ...}
    :param filename: filepath containing corpus files
    :param utterance_index: utterance meta index
    :param exclude_meta: list of metadata attributes to exclude
    :param KeyMeta: name of metadata key, should be 'meta'
    :return:
    """
    for field, field_types in utterance_index.items():
        if len(field_types) > 0 and field_types[0] == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-bin.p"), "rb") as f:
                l_bin = pickle.load(f)
            for i, ut in enumerate(utterances):
                for k, v in ut[KeyMeta].items():
                    if (
                        k == field
                        and type(v) == str
                        and v.startswith(BIN_DELIM_L)
                        and v.endswith(BIN_DELIM_R)
                    ):
                        idx = int(v[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        utterances[i][KeyMeta][k] = l_bin[idx]
    for field in exclude_meta:
        del utterance_index[field]


def unpack_binary_data(filename, objs_data, object_index, obj_type, exclude_meta):
    """
    Unpack binary data for Speakers or Conversations

    :param filename: filepath containing the corpus data
    :param objs_data: a mapping from object id to a dictionary with two keys: 'meta' and 'vectors';
        in older versions, this is a mapping from object id to the metadata dictionary
    :param object_index: the meta_index dictionary for the component type
    :param obj_type: object type (i.e. speaker or conversation)
    :param exclude_meta: list of metadata attributes to exclude
    :return: None (mutates objs_data)
    """
    """
    Unpack binary data for Speakers or Conversations

    """
    # unpack speaker meta
    for field, field_types in object_index.items():
        if len(field_types) > 0 and field_types[0] == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-{}-bin.p".format(obj_type)), "rb") as f:
                l_bin = pickle.load(f)
            for obj, data in objs_data.items():
                metadata = data["meta"] if len(data) == 2 and "vectors" in data else data
                for k, v in metadata.items():
                    if (
                        k == field
                        and type(v) == str
                        and str(v).startswith(BIN_DELIM_L)
                        and str(v).endswith(BIN_DELIM_R)
                    ):
                        idx = int(v[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        metadata[k] = l_bin[idx]
    for field in exclude_meta:
        del object_index[field]


def unpack_all_binary_data(
    filename: str,
    meta_index: ConvoKitIndex,
    meta: ConvoKitMeta,
    utterances: List[Utterance],
    speakers_data: Dict[str, Dict],
    convos_data: Dict[str, Dict],
    exclude_utterance_meta: List[str],
    exclude_speaker_meta: List[str],
    exclude_conversation_meta: List[str],
    exclude_overall_meta: List[str],
):
    # unpack binary data for utterances
    unpack_binary_data_for_utts(
        utterances,
        filename,
        meta_index.utterances_index,
        exclude_utterance_meta,
        KeyMeta,
    )
    # unpack binary data for speakers
    unpack_binary_data(
        filename,
        speakers_data,
        meta_index.speakers_index,
        "speaker",
        exclude_speaker_meta,
    )

    # unpack binary data for conversations
    unpack_binary_data(
        filename,
        convos_data,
        meta_index.conversations_index,
        "convo",
        exclude_conversation_meta,
    )

    # unpack binary data for overall corpus
    unpack_binary_data(
        filename,
        meta,
        meta_index.overall_index,
        "overall",
        exclude_overall_meta,
    )


def load_from_utterance_file(filename, utterance_start_index, utterance_end_index):
    """
    where filename is "utterances.json" or "utterances.jsonl" for example
    """
    with open(filename, "r") as f:
        try:
            ext = filename.split(".")[-1]
            if ext == "json":
                utterances = json.load(f)
            elif ext == "jsonl":
                utterances = []
                if utterance_start_index is None:
                    utterance_start_index = 0
                if utterance_end_index is None:
                    utterance_end_index = float("inf")
                idx = 0
                for line in f:
                    if utterance_start_index <= idx <= utterance_end_index:
                        utterances.append(json.loads(line))
                    idx += 1
        except Exception as e:
            raise Exception(
                "Could not load corpus. Expected json file, encountered error: \n" + str(e)
            )
    return utterances


def initialize_speakers_and_utterances_objects(corpus, utterances, speakers_data):
    """
    Initialize Speaker and Utterance objects
    """
    if len(utterances) > 0:  # utterances might be empty for invalid corpus start/end indices
        KeySpeaker = "speaker" if "speaker" in utterances[0] else "user"
        KeyConvoId = "conversation_id" if "conversation_id" in utterances[0] else "root"

    for i, u in enumerate(utterances):
        u = defaultdict(lambda: None, u)
        speaker_key = u[KeySpeaker]
        if speaker_key not in corpus.speakers:
            if u[KeySpeaker] not in speakers_data:
                warn(
                    "CorpusLoadWarning: Missing speaker metadata for speaker ID: {}. "
                    "Initializing default empty metadata instead.".format(u[KeySpeaker])
                )
                speakers_data[u[KeySpeaker]] = {}
            if KeyMeta in speakers_data[u[KeySpeaker]]:
                corpus.speakers[speaker_key] = Speaker(
                    owner=corpus, id=u[KeySpeaker], meta=speakers_data[u[KeySpeaker]][KeyMeta]
                )
            else:
                corpus.speakers[speaker_key] = Speaker(
                    owner=corpus, id=u[KeySpeaker], meta=speakers_data[u[KeySpeaker]]
                )

        speaker = corpus.speakers[speaker_key]
        speaker.vectors = speakers_data[u[KeySpeaker]].get(KeyVectors, [])

        # temp fix for reddit reply_to
        if "reply_to" in u:
            reply_to_data = u["reply_to"]
        else:
            reply_to_data = u[KeyReplyTo]
        utt = Utterance(
            owner=corpus,
            id=u[KeyId],
            speaker=speaker,
            conversation_id=u[KeyConvoId],
            reply_to=reply_to_data,
            timestamp=u[KeyTimestamp],
            text=u[KeyText],
            meta=u[KeyMeta],
        )
        utt.vectors = u.get(KeyVectors, [])
        corpus.utterances[utt.id] = utt


def merge_utterance_lines(utt_dict):
    """
    For merging adjacent utterances by the same speaker
    """
    new_utterances = {}
    merged_with = {}
    for uid, utt in utt_dict.items():
        merged = False
        if utt.reply_to is not None and utt.speaker is not None:
            u0 = utt_dict[utt.reply_to]
            if u0.conversation_id == utt.conversation_id and u0.speaker == utt.speaker:
                merge_target = merged_with[u0.id] if u0.id in merged_with else u0.id
                new_utterances[merge_target].text += " " + utt.text
                merged_with[utt.id] = merge_target
                merged = True
        if not merged:
            if utt.reply_to in merged_with:
                utt.reply_to = merged_with[utt.reply_to]
            new_utterances[utt.id] = utt
    return new_utterances


def _update_reply_to_chain_with_conversation_id(
    utterances_dict: Dict[str, Utterance],
    utt_ids_to_replier_ids: Dict[str, Iterable[str]],
    root_utt_id: str,
    conversation_id: str,
):
    repliers = utt_ids_to_replier_ids.get(root_utt_id, deque())
    while len(repliers) > 0:
        replier_id = repliers.popleft()
        utterances_dict[replier_id].conversation_id = conversation_id
        repliers.extend(utt_ids_to_replier_ids[replier_id])


def fill_missing_conversation_ids(utterances_dict: Dict[str, Utterance]) -> None:
    """
    Populates `conversation_id` in Utterances that have `conversation_id` set to `None`, with a Conversation root-specific generated ID
    :param utterances_dict:
    :return:
    """
    utts_without_convo_ids = [
        utt for utt in utterances_dict.values() if utt.conversation_id is None
    ]
    utt_ids_to_replier_ids = defaultdict(deque)
    convo_roots_without_convo_ids = []
    convo_roots_with_convo_ids = []
    for utt in utterances_dict.values():
        if utt.reply_to is None:
            if utt.conversation_id is None:
                convo_roots_without_convo_ids.append(utt.id)
            else:
                convo_roots_with_convo_ids.append(utt.id)
        else:
            utt_ids_to_replier_ids[utt.reply_to].append(utt.id)

    # connect the reply-to edges for convo roots without convo ids
    for root_utt_id in convo_roots_without_convo_ids:
        generated_conversation_id = Conversation.generate_default_conversation_id(
            utterance_id=root_utt_id
        )
        utterances_dict[root_utt_id].conversation_id = generated_conversation_id
        _update_reply_to_chain_with_conversation_id(
            utterances_dict=utterances_dict,
            utt_ids_to_replier_ids=utt_ids_to_replier_ids,
            root_utt_id=root_utt_id,
            conversation_id=generated_conversation_id,
        )

    # Previous section handles all *new* conversations
    # Next section handles utts that belong to existing conversations
    for root_utt_id in convo_roots_with_convo_ids:
        conversation_id = utterances_dict[root_utt_id].conversation_id
        _update_reply_to_chain_with_conversation_id(
            utterances_dict=utterances_dict,
            utt_ids_to_replier_ids=utt_ids_to_replier_ids,
            root_utt_id=root_utt_id,
            conversation_id=conversation_id,
        )

    # It's still possible to have utts that reply to non-existent utts
    # These are the utts that do not have a conversation_id even at this step
    for utt in utts_without_convo_ids:
        if utt.conversation_id is None:
            raise ValueError(
                f"Invalid Utterance found: Utterance {utt.id} replies to an Utterance '{utt.reply_to}' that does not exist."
            )


def initialize_conversations(
    corpus, convos_data, convo_to_utts=None, fill_missing_convo_ids: bool = False
):
    """
    Initialize Conversation objects from utterances and conversations data.
    If a mapping from Conversation IDs to their constituent Utterance IDs is
    already known (e.g., as a side effect of a prior computation) they can be
    directly provided via the convo_to_utts parameter, otherwise the mapping
    will be computed by iteration over the Utterances in utt_dict.
    """
    if fill_missing_convo_ids:
        fill_missing_conversation_ids(corpus.utterances)

    # organize utterances by conversation
    if convo_to_utts is None:
        convo_to_utts = defaultdict(list)  # temp container identifying utterances by conversation
        for utt in corpus.utterances.values():
            convo_key = (
                utt.conversation_id
            )  # each conversation_id is considered a separate conversation
            convo_to_utts[convo_key].append(utt.id)
    conversations = {}
    for convo_id in convo_to_utts:
        # look up the metadata associated with this conversation, if any
        convo_data = convos_data.get(convo_id, None)
        if convo_data is not None:
            if KeyMeta in convo_data:
                convo_meta = convo_data[KeyMeta]
            else:
                convo_meta = convo_data
        else:
            convo_meta = None

        convo = Conversation(
            owner=corpus, id=convo_id, utterances=convo_to_utts[convo_id], meta=convo_meta
        )

        if convo_data is not None and KeyVectors in convo_data and KeyMeta in convo_data:
            convo.vectors = convo_data.get(KeyVectors, [])
        conversations[convo_id] = convo
    return conversations


def dump_helper_bin(d: ConvoKitMeta, d_bin: Dict, fields_to_skip=None) -> Dict:  # object_idx
    """

    :param d: The ConvoKitMeta to encode
    :param d_bin: The dict of accumulated lists of binary attribs
    :return:
    """
    if fields_to_skip is None:
        fields_to_skip = []

    obj_idx = d.index.get_index(d.obj_type)
    d_out = {}
    for k, v in d.items():
        if k in fields_to_skip:
            continue
        try:
            if len(obj_idx[k]) > 0 and obj_idx[k][0] == "bin":
                d_out[k] = "{}{}{}".format(BIN_DELIM_L, len(d_bin[k]), BIN_DELIM_R)
                d_bin[k].append(v)
            else:
                d_out[k] = v
        except KeyError:
            # fails silently (object has no such metadata that was indicated in metadata index)
            pass
    return d_out


def dump_corpus_component(
    corpus, dir_name, filename, obj_type, bin_name, exclude_vectors, fields_to_skip
):
    with open(os.path.join(dir_name, filename), "w") as f:
        d_bin = defaultdict(list)
        objs = defaultdict(dict)
        for obj_id in corpus.get_object_ids(obj_type):
            objs[obj_id][KeyMeta] = dump_helper_bin(
                corpus.get_object(obj_type, obj_id).meta, d_bin, fields_to_skip.get(obj_type, [])
            )
            obj_vectors = corpus.get_object(obj_type, obj_id).vectors
            objs[obj_id][KeyVectors] = (
                obj_vectors
                if exclude_vectors is None
                else list(set(obj_vectors) - set(exclude_vectors))
            )
        json.dump(objs, f)

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-{}-bin.p".format(bin_name)), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)


def dump_utterances(corpus, dir_name, exclude_vectors, fields_to_skip):
    with open(os.path.join(dir_name, "utterances.jsonl"), "w") as f:
        d_bin = defaultdict(list)

        for ut in corpus.iter_utterances():
            ut_obj = {
                KeyId: ut.id,
                KeyConvoId: ut.conversation_id,
                KeyText: ut.text,
                KeySpeaker: ut.speaker.id,
                KeyMeta: dump_helper_bin(ut.meta, d_bin, fields_to_skip.get("utterance", [])),
                KeyReplyTo: ut.reply_to,
                KeyTimestamp: ut.timestamp,
                KeyVectors: (
                    ut.vectors
                    if exclude_vectors is None
                    else list(set(ut.vectors) - set(exclude_vectors))
                ),
            }
            json.dump(ut_obj, f)
            f.write("\n")

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-bin.p"), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)


def load_jsonlist_to_dict(filename, index_key="id", value_key="value"):
    entries = {}
    with open(filename, "r") as f:
        for line in f:
            entry = json.loads(line)
            entries[entry[index_key]] = entry[value_key]
    return entries


def dump_jsonlist_from_dict(entries, filename, index_key="id", value_key="value"):
    with open(filename, "w") as f:
        for k, v in entries.items():
            json.dump({index_key: k, value_key: v}, f)
            f.write("\n")


def extract_meta_from_df(df):
    meta_cols = [col.split(".")[1] for col in df if col.startswith("meta")]
    return meta_cols


def load_binary_metadata(filename, index, exclude_meta=None):
    binary_data = {"utterance": {}, "conversation": {}, "speaker": {}, "corpus": {}}
    for component_type in binary_data:
        meta_index = index.get_index(component_type)
        for meta_key, meta_type in meta_index.items():
            if meta_type == ["bin"] and (
                exclude_meta is None or meta_key not in exclude_meta[component_type]
            ):
                # filename format differs for utterances versus everything else
                filename_suffix = (
                    "-bin.p"
                    if component_type == "utterance"
                    else "-{}-bin.p".format(component_type)
                )
                try:
                    with open(os.path.join(filename, meta_key + filename_suffix), "rb") as f:
                        l_bin = pickle.load(f)
                        binary_data[component_type][meta_key] = l_bin
                except FileNotFoundError:
                    warn(
                        f"Metadata field {meta_key} is specified to have binary type but no saved binary data was found. This field will be skipped."
                    )
                    # update the exclude_meta list to force this field to get skipped
                    # in the subsequent corpus loading logic
                    if exclude_meta is None:
                        exclude_meta = defaultdict(list)
                    exclude_meta[component_type].append(meta_key)
    return binary_data, exclude_meta


def load_jsonlist_to_db(
    filename,
    db,
    collection_prefix,
    start_line=None,
    end_line=None,
    exclude_meta=None,
    bin_meta=None,
):
    """
    Populate the specified MongoDB database with the utterance data contained in
    the given filename (which should point to an utterances.jsonl file).
    """
    utt_collection = db[f"{collection_prefix}_utterance"]
    meta_collection = db[f"{collection_prefix}_meta"]
    inserted_ids = set()
    speaker_key = None
    convo_key = None
    reply_key = None
    with open(filename) as f:
        utt_insertion_buffer = []
        meta_insertion_buffer = []
        for ln, line in enumerate(f):
            if start_line is not None and ln < start_line:
                continue
            if end_line is not None and ln > end_line:
                break
            utt_obj = json.loads(line)
            if speaker_key is None:
                # backwards compatibility for corpora made before the user->speaker rename
                speaker_key = "speaker" if "speaker" in utt_obj else "user"
            if convo_key is None:
                # backwards compatibility for corpora made before the root->conversation_id rename
                convo_key = "conversation_id" if "conversation_id" in utt_obj else "root"
            if reply_key is None:
                # fix for misnamed reply_to in subreddit corpora
                reply_key = "reply-to" if "reply-to" in utt_obj else "reply_to"
            utt_obj = defaultdict(lambda: None, utt_obj)
            utt_insertion_buffer.append(
                UpdateOne(
                    {"_id": utt_obj["id"]},
                    {
                        "$set": {
                            "speaker_id": utt_obj[speaker_key],
                            "conversation_id": utt_obj[convo_key],
                            "reply_to": utt_obj[reply_key],
                            "timestamp": utt_obj["timestamp"],
                            "text": utt_obj["text"],
                        }
                    },
                    upsert=True,
                )
            )
            utt_meta = utt_obj["meta"]
            if exclude_meta is not None:
                for exclude_key in exclude_meta:
                    if exclude_key in utt_meta:
                        del utt_meta[exclude_key]
            if bin_meta is not None:
                for key, bin_list in bin_meta.items():
                    bin_locator = utt_meta.get(key, None)
                    if (
                        type(bin_locator) == str
                        and bin_locator.startswith(BIN_DELIM_L)
                        and bin_locator.endswith(BIN_DELIM_R)
                    ):
                        bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        utt_meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
            meta_insertion_buffer.append(
                UpdateOne({"_id": "utterance_" + utt_obj["id"]}, {"$set": utt_meta}, upsert=True)
            )
            inserted_ids.add(utt_obj["id"])
            if len(utt_insertion_buffer) >= JSONLIST_BUFFER_SIZE:
                utt_collection.bulk_write(utt_insertion_buffer)
                meta_collection.bulk_write(meta_insertion_buffer)
                utt_insertion_buffer = []
                meta_insertion_buffer = []
        # after loop termination, insert any remaining items in the buffer
        if len(utt_insertion_buffer) > 0:
            utt_collection.bulk_write(utt_insertion_buffer)
            meta_collection.bulk_write(meta_insertion_buffer)
            utt_insertion_buffer = []
            meta_insertion_buffer = []
    return inserted_ids


def load_json_to_db(
    filename, db, collection_prefix, component_type, exclude_meta=None, bin_meta=None
):
    """
    Populate the specified MongoDB database with corpus component data from
    either the speakers.json or conversations.json file located in a directory
    containing valid ConvoKit Corpus data. The component_type parameter controls
    which JSON file gets used.
    """
    component_collection = db[f"{collection_prefix}_{component_type}"]
    meta_collection = db[f"{collection_prefix}_meta"]
    if component_type == "speaker":
        json_data = load_speakers_data_from_dir(filename, exclude_meta)
    elif component_type == "conversation":
        json_data = load_convos_data_from_dir(filename, exclude_meta)
    component_insertion_buffer = []
    meta_insertion_buffer = []
    for component_id, component_data in json_data.items():
        if KeyMeta in component_data:
            # contains non-metadata entries
            payload = {k: v for k, v in component_data.items() if k not in {"meta", "vectors"}}
            meta = component_data[KeyMeta]
        else:
            # contains only metadata, with metadata at the top level
            payload = {}
            meta = component_data
        component_insertion_buffer.append(
            UpdateOne({"_id": component_id}, {"$set": payload}, upsert=True)
        )
        if bin_meta is not None:
            for key, bin_list in bin_meta.items():
                bin_locator = meta.get(key, None)
                if (
                    type(bin_locator) == str
                    and bin_locator.startswith(BIN_DELIM_L)
                    and bin_locator.endswith(BIN_DELIM_R)
                ):
                    bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                    meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
        meta_insertion_buffer.append(
            UpdateOne({"_id": f"{component_type}_{component_id}"}, {"$set": meta}, upsert=True)
        )
    component_collection.bulk_write(component_insertion_buffer)
    meta_collection.bulk_write(meta_insertion_buffer)


def load_corpus_info_to_db(filename, db, collection_prefix, exclude_meta=None, bin_meta=None):
    """
    Populate the specified MongoDB database with Corpus metadata loaded from the
    corpus.json file of a directory containing valid ConvoKit Corpus data.
    """
    if exclude_meta is None:
        exclude_meta = {}
    meta_collection = db[f"{collection_prefix}_meta"]
    with open(os.path.join(filename, "corpus.json")) as f:
        corpus_meta = {k: v for k, v in json.load(f).items() if k not in exclude_meta}
        if bin_meta is not None:
            for key, bin_list in bin_meta.items():
                bin_locator = corpus_meta.get(key, None)
                if (
                    type(bin_locator) == str
                    and bin_locator.startswith(BIN_DELIM_L)
                    and bin_locator.endswith(BIN_DELIM_R)
                ):
                    bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                    corpus_meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
        meta_collection.update_one(
            {"_id": f"corpus_{collection_prefix}"}, {"$set": corpus_meta}, upsert=True
        )


def load_info_to_mem(corpus, dir_name, obj_type, field):
    """
    Helper for load_info in mem mode that reads the file for the specified extra
    info field, loads it into memory, and assigns the entries to their
    corresponding corpus components.
    """
    getter = lambda oid: corpus.get_object(obj_type, oid)
    entries = load_jsonlist_to_dict(os.path.join(dir_name, "info.%s.jsonl" % field))
    for k, v in entries.items():
        try:
            obj = getter(k)
            obj.add_meta(field, v)
        except:
            continue


def load_info_to_db(corpus, dir_name, obj_type, field, index_key="id", value_key="value"):
    """
    Helper for load_info in DB mode that reads the jsonlist file for the
    specified extra info field in a batched line-by-line manner, populates
    its contents into the DB, and updates the Corpus' metadata index
    """
    filename = os.path.join(dir_name, "info.%s.jsonl" % field)
    meta_collection = corpus.backend_mapper.get_collection("meta")

    # attept to use saved type information
    index_file = os.path.join(dir_name, "index.json")
    with open(index_file) as f:
        raw_index = json.load(f)
        try:
            field_type = raw_index[f"{obj_type}s-index"][field]
            corpus.meta_index.get_index(obj_type)[field] = field_type
            index_updated = True
        except:
            # field not recorded in the index file; we will need to infer
            # types during insertion time
            index_updated = False

    # iteratively insert the info in the DB in batched fashion
    with open(filename) as f:
        info_insertion_buffer = []
        for line in f:
            info_json = json.loads(line)
            obj_id, info_val = info_json[index_key], info_json[value_key]
            if not index_updated:
                # we were previously unable to fetch the type info from the
                # index file, so we must infer it now
                ConvoKitMeta._check_type_and_update_index(
                    corpus.meta_index, obj_type, field, info_val
                )
            info_insertion_buffer.append(
                UpdateOne(
                    {"_id": "{}_{}".format(obj_type, obj_id)},
                    {"$set": {field: info_val}},
                    upsert=True,
                )
            )
            if len(info_insertion_buffer) >= JSONLIST_BUFFER_SIZE:
                meta_collection.bulk_write(info_insertion_buffer)
                info_insertion_buffer = []
        # after loop termination, insert any remaining items in the buffer
        if len(info_insertion_buffer) > 0:
            meta_collection.bulk_write(info_insertion_buffer)
            info_insertion_buffer = []


def clean_up_excluded_meta(meta_index, exclude_meta):
    """
    Remove excluded metadata from the metadata index
    """
    for component_type, excluded_keys in exclude_meta.items():
        for key in excluded_keys:
            meta_index.del_from_index(component_type, key)


def populate_db_from_file(
    filename,
    db,
    collection_prefix,
    meta_index,
    utterance_start_index,
    utterance_end_index,
    exclude_utterance_meta,
    exclude_conversation_meta,
    exclude_speaker_meta,
    exclude_overall_meta,
):
    """
    Populate all necessary collections of a MongoDB database so that it can be
    used by a DBMapper, sourcing data from the valid ConvoKit Corpus
    data pointed to by the filename parameter.
    """
    binary_meta, updated_exclude_meta = load_binary_metadata(
        filename,
        meta_index,
        {
            "utterance": exclude_utterance_meta,
            "conversation": exclude_conversation_meta,
            "speaker": exclude_speaker_meta,
            "corpus": exclude_overall_meta,
        },
    )

    # exclusion lists may have changed if errors were encountered while loading
    # the binary metadata
    if updated_exclude_meta is not None:
        exclude_utterance_meta = updated_exclude_meta["utterance"]
        exclude_conversation_meta = updated_exclude_meta["conversation"]
        exclude_speaker_meta = updated_exclude_meta["speaker"]
        exclude_overall_meta = updated_exclude_meta["corpus"]

    # first load the utterance data
    inserted_utt_ids = load_jsonlist_to_db(
        os.path.join(filename, "utterances.jsonl"),
        db,
        collection_prefix,
        utterance_start_index,
        utterance_end_index,
        exclude_utterance_meta,
        binary_meta["utterance"],
    )
    # next load the speaker and conversation data
    for component_type in ["speaker", "conversation"]:
        load_json_to_db(
            filename,
            db,
            collection_prefix,
            component_type,
            (exclude_speaker_meta if component_type == "speaker" else exclude_conversation_meta),
            binary_meta[component_type],
        )
    # finally, load the corpus metadata
    load_corpus_info_to_db(
        filename, db, collection_prefix, exclude_overall_meta, binary_meta["corpus"]
    )

    # make sure skipped metadata isn't kept in the final index
    clean_up_excluded_meta(
        meta_index,
        {
            "utterance": exclude_utterance_meta,
            "conversation": exclude_conversation_meta,
            "speaker": exclude_speaker_meta,
            "corpus": exclude_overall_meta,
        },
    )

    return inserted_utt_ids


def init_corpus_from_backend_manager(corpus, utt_ids=None):
    """
    Use an already-populated MongoDB database to initialize the components of
    the specified Corpus (which should be empty before this function is called)
    """
    # we will bypass the initialization step when constructing components since
    # we know their necessary data already exists within the db
    corpus.backend_mapper.bypass_init = True

    # fetch object ids from the DB and initialize corpus components for them
    # create speakers first so we can refer to them when initializing utterances
    speakers = {}
    for speaker_doc in corpus.backend_mapper.data["speaker"].find(projection=["_id"]):
        speaker_id = speaker_doc["_id"]
        speakers[speaker_id] = Speaker(owner=corpus, id=speaker_id)
    corpus.speakers = speakers

    # next, create utterances
    utterances = {}
    convo_to_utts = defaultdict(list)
    for utt_doc in corpus.backend_mapper.data["utterance"].find(
        projection=["_id", "speaker_id", "conversation_id"]
    ):
        utt_id = utt_doc["_id"]
        if utt_ids is None or utt_id in utt_ids:
            convo_to_utts[utt_doc["conversation_id"]].append(utt_id)
            utterances[utt_id] = Utterance(
                owner=corpus, id=utt_id, speaker=speakers[utt_doc["speaker_id"]]
            )
    corpus.utterances = utterances

    # run post-construction integrity steps as in regular constructor
    corpus.conversations = initialize_conversations(corpus, {}, convo_to_utts)
    corpus.meta_index.enable_type_check()
    corpus.update_speakers_data()

    # restore the BackendMapper's init behavior to default
    corpus.backend_mapper.bypass_init = False
