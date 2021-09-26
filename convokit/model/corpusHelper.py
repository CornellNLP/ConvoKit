"""
Contains functions that help with the construction / dumping of a Corpus
"""

import os
import json
from collections import defaultdict
from typing import Dict
import pickle

from .speaker import Speaker
from .utterance import Utterance
from .conversation import Conversation
from .convoKitMeta import ConvoKitMeta
from convokit.util import warn

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

def load_uttinfo_from_dir(dirname, utterance_start_index, utterance_end_index, exclude_utterance_meta):
    assert dirname is not None
    assert os.path.isdir(dirname)

    if utterance_start_index is None: utterance_start_index = 0
    if utterance_end_index is None: utterance_end_index = float('inf')

    if os.path.exists(os.path.join(dirname, 'utterances.jsonl')):
        with open(os.path.join(dirname, 'utterances.jsonl'), 'r') as f:
            utterances = []
            idx = 0
            for line in f:
                if utterance_start_index <= idx <= utterance_end_index:
                    utterances.append(json.loads(line))
                idx += 1

    elif os.path.exists(os.path.join(dirname, 'utterances.json')):
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

        if len(id_to_speaker_data) > 0 and len(next(iter(id_to_speaker_data.values()))) and \
                'vectors' in id_to_speaker_data == 2:
            # has vectors data
            for _, speaker_data in id_to_speaker_data.items():
                for k in exclude_speaker_meta:
                    if k in speaker_data['meta']:
                        del speaker_data['meta'][k]
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

        if len(id_to_convo_data) > 0 and len(next(iter(id_to_convo_data.values()))) and \
                'vectors' in id_to_convo_data == 2:
            # has vectors data
            for _, convo_data in id_to_convo_data.items():
                for k in exclude_conversation_meta:
                    if k in convo_data['meta']:
                        del convo_data['meta'][k]
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
            if k in exclude_overall_meta: continue
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
        if field_types[0] == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-bin.p"), "rb") as f:
                l_bin = pickle.load(f)
            for i, ut in enumerate(utterances):
                for k, v in ut[KeyMeta].items():
                    if k == field and type(v) == str and v.startswith(BIN_DELIM_L) and \
                            v.endswith(BIN_DELIM_R):
                        idx = int(v[len(BIN_DELIM_L):-len(BIN_DELIM_R)])
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
        if field_types[0] == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-{}-bin.p".format(obj_type)), "rb") as f:
                l_bin = pickle.load(f)
            for obj, data in objs_data.items():
                metadata = data['meta'] if len(data) == 2 and 'vectors' in data else data
                for k, v in metadata.items():
                    if k == field and type(v) == str and str(v).startswith(BIN_DELIM_L) and \
                            str(v).endswith(BIN_DELIM_R):
                        idx = int(v[len(BIN_DELIM_L):-len(BIN_DELIM_R)])
                        metadata[k] = l_bin[idx]
    for field in exclude_meta:
        del object_index[field]


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
                if utterance_start_index is None: utterance_start_index = 0
                if utterance_end_index is None: utterance_end_index = float('inf')
                idx = 0
                for line in f:
                    if utterance_start_index <= idx <= utterance_end_index:
                        utterances.append(json.loads(line))
                    idx += 1
        except Exception as e:
            raise Exception("Could not load corpus. Expected json file, encountered error: \n" + str(e))
    return utterances


def initialize_speakers_and_utterances_objects(corpus, utt_dict, utterances, speakers_dict, speakers_data):
    """
    Initialize Speaker and Utterance objects
    """
    if len(utterances) > 0: # utterances might be empty for invalid corpus start/end indices
        KeySpeaker = "speaker" if "speaker" in utterances[0] else "user"
        KeyConvoId = "conversation_id" if "conversation_id" in utterances[0] else "root"

    for i, u in enumerate(utterances):
        u = defaultdict(lambda: None, u)
        speaker_key = u[KeySpeaker]
        if speaker_key not in speakers_dict:
            if u[KeySpeaker] not in speakers_data:
                warn("CorpusLoadWarning: Missing speaker metadata for speaker ID: {}. "
                     "Initializing default empty metadata instead.".format(u[KeySpeaker]))
                speakers_data[u[KeySpeaker]] = {}
            if KeyMeta in speakers_data[u[KeySpeaker]]:
                speakers_dict[speaker_key] = Speaker(owner=corpus, id=u[KeySpeaker],
                                                     meta=speakers_data[u[KeySpeaker]][KeyMeta])
            else:
                speakers_dict[speaker_key] = Speaker(owner=corpus, id=u[KeySpeaker],
                                                     meta=speakers_data[u[KeySpeaker]])

        speaker = speakers_dict[speaker_key]
        speaker.vectors = speakers_data[u[KeySpeaker]].get(KeyVectors, [])

        # temp fix for reddit reply_to
        if "reply_to" in u:
            reply_to_data = u["reply_to"]
        else:
            reply_to_data = u[KeyReplyTo]
        utt = Utterance(owner=corpus, id=u[KeyId], speaker=speaker,
                        conversation_id=u[KeyConvoId],
                        reply_to=reply_to_data, timestamp=u[KeyTimestamp],
                        text=u[KeyText], meta=u[KeyMeta])
        utt.vectors = u.get(KeyVectors, [])
        utt_dict[utt.id] = utt


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


def initialize_conversations(corpus, utt_dict, convos_data):
    """
    Initialize Conversation objects from utterances and conversations data
    """
    # organize utterances by conversation
    convo_to_utts = defaultdict(list) # temp container identifying utterances by conversation
    for u in utt_dict.values():
        convo_key = u.conversation_id # each conversation_id is considered a separate conversation
        convo_to_utts[convo_key].append(u.id)
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

        convo = Conversation(owner=corpus, id=convo_id,
                             utterances=convo_to_utts[convo_id],
                             meta=convo_meta)

        if convo_data is not None and KeyVectors in convo_data and KeyMeta in convo_data:
            convo.vectors = convo_data.get(KeyVectors, [])
        conversations[convo_id] = convo
    return conversations


def dump_helper_bin(d: ConvoKitMeta, d_bin: Dict, fields_to_skip=None) -> Dict: # object_idx
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
        if k in fields_to_skip: continue
        try:
            if obj_idx[k][0] == "bin":
                d_out[k] = "{}{}{}".format(BIN_DELIM_L, len(d_bin[k]), BIN_DELIM_R)
                d_bin[k].append(v)
            else:
                d_out[k] = v
        except KeyError:
            # fails silently (object has no such metadata that was indicated in metadata index)
            pass
    return d_out


def dump_corpus_component(corpus, dir_name, filename, obj_type, bin_name, exclude_vectors, fields_to_skip):
    with open(os.path.join(dir_name, filename), "w") as f:
        d_bin = defaultdict(list)
        objs = defaultdict(dict)
        for obj_id in corpus.get_object_ids(obj_type):
            objs[obj_id][KeyMeta] = dump_helper_bin(corpus.get_object(obj_type, obj_id).meta,
                                                   d_bin, fields_to_skip.get(obj_type, []))
            obj_vectors = corpus.get_object(obj_type, obj_id).vectors
            objs[obj_id][KeyVectors] = obj_vectors if exclude_vectors is None else \
                                      list(set(obj_vectors) - set(exclude_vectors))
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
                KeyMeta: dump_helper_bin(ut.meta, d_bin, fields_to_skip.get('utterance', [])),
                KeyReplyTo: ut.reply_to,
                KeyTimestamp: ut.timestamp,
                KeyVectors: ut.vectors if exclude_vectors is None else list(set(ut.vectors) - set(exclude_vectors))
            }
            json.dump(ut_obj, f)
            f.write("\n")

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-bin.p"), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)


def load_jsonlist_to_dict(filename, index_key='id', value_key='value'):
    entries = {}
    with open(filename, 'r') as f:
        for line in f:
            entry = json.loads(line)
            entries[entry[index_key]] = entry[value_key]
    return entries


def dump_jsonlist_from_dict(entries, filename, index_key='id', value_key='value'):
    with open(filename, 'w') as f:
        for k, v in entries.items():
            json.dump({index_key: k, value_key: v}, f)
            f.write('\n')

def extract_meta_from_df(df):
    meta_cols = [col for col in df if col.startswith('meta')]
    return meta_cols

def add_conv_meta_df(conversations_df, corpus):
    conv_meta_cols = extract_meta_from_df(conversations_df)
    for convo in corpus.iter_conversations():
        # get the conv_id for the conversation by checking from utterance info
        convo_id = convo.get_id()
        conversation_meta = conversations_df[conversations_df['id'] == convo_id][conv_meta_cols].to_dict(orient='records')[0] if conv_meta_cols else None

        convo.meta.update(conversation_meta)
    
    return corpus