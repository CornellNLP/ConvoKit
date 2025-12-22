import json
import sys

from matplotlib import text

sys.path.insert(0, "/home/jonathan/research/Cornell-Conversational-Analysis-Toolkit")
from convokit import Corpus, Speaker, Utterance, Conversation
import os
from datetime import datetime, timedelta
import copy
import shutil


def main():

    # PARAMETERS FOR USER TO SET
    data_directory_input = "/kitchen/wikiconv-convokit-processing/output/English"  # directory where the files to transform into the Convokit format are kept
    data_directory_intermediate = "/kitchen/wikiconv-convokit-processing/store_test_merging/"  # intermediate directory where the split Convokit files are also kept
    data_directory_output = "/kitchen/wikiconv-convokit-processing/final/English/"  # directory to output the merged Convokit files
    delete_intermediate_files = True  # set to True for data_directory_intermediate to be immediately deleted; otherwise intermediate directory is also stored
    utterances_before_output = 5 * (
        10**5
    )  # number of utterances before output is generated to limit memory consumption, initalized to half a million

    # Setting up transformation script varialbes
    complete_utterance_list = []
    master_conversation_level_dict = {}
    input_files_read = set()
    count_of_data_splits = 0
    # Read all files from the data input
    for input_filename in os.listdir(data_directory_input):
        if input_filename.endswith(".jsonlist") and (not (input_filename in input_files_read)):
            input_files_read.add(input_filename)
            final_input_path = os.path.join(data_directory_input, input_filename)
            print(str(final_input_path))  # print the current file as an indicator of progress

            (
                individual_utterance_list,
                conversation_individual_dict,
            ) = create_utterances_from_individual_file(final_input_path)
            master_conversation_level_dict.update(conversation_individual_dict)

            # Error checking - see check_for_dictionary_vals method
            # for value in individual_utterance_list:
            #     check_for_dictionary_vals(value)

            # to reduce memory consumption, reset the utterance list if it gets too long
            complete_utterance_list.extend(individual_utterance_list)
            if len(complete_utterance_list) > utterances_before_output:
                dictionary_of_year_lists = separate_by_year(complete_utterance_list)
                for key, value in dictionary_of_year_lists.items():
                    set_meta_dump_corpus(
                        value,
                        master_conversation_level_dict,
                        data_directory_intermediate,
                        "conv_corpus_year_" + str(key) + "_data_split_" + str(count_of_data_splits),
                    )
                complete_utterance_list = []
                master_conversation_level_dict = {}
                count_of_data_splits += 1

    # Once all the data has been converted to a list of utterances, break up the data by the year
    dictionary_of_year_lists = separate_by_year(complete_utterance_list)
    for key, value in dictionary_of_year_lists.items():
        set_meta_dump_corpus(
            value,
            master_conversation_level_dict,
            data_directory_intermediate,
            "conv_corpus_year_" + str(key) + "_data_split_" + str(count_of_data_splits),
        )

    # Now merge files into the final master and remove intermediate files if neccessary
    merge_files(data_directory_output, data_directory_intermediate)
    if delete_intermediate_files:
        shutil.rmtree(data_directory_intermediate)


# def check_dict_for_values(dict_val):
#     list_of_keys = dict_val.keys():


# Un-comment for error checking, ensure that every value 3 in the utterance's metadata is a dictionary;
# def check_for_dictionary_vals(utterance):
#     modification_list = utterance.meta['modification']
#     deletion_list = utterance.meta['deletion']
#     restoration_list = utterance.meta['restoration']

#     if (len(modification_list)>0):
#         for value in modification_list:
#             # print(value)
#             if (not(type(value) is dict)):
#                 print (type(value))
#     if (len(deletion_list)>0):
#         for value in deletion_list:
#             if (not(type(value) is dict)):
#                 print (type(value))

#     if (len(restoration_list)>0):
#         for value in restoration_list:
#             if (not(type(value) is dict)):
#                 print (type(value))

# function for files to merge
def merge_files(final_directory, input_directory):
    input_subdirectory_paths = [x[0] for x in os.walk(input_directory)]
    for year_x in range(1990, 2021):
        print(year_x)
        full_lst_corpora = []
        full_lst_corpora.extend(search_list_for_year(input_subdirectory_paths, str(year_x)))
        master_corpora(final_directory, full_lst_corpora, str(year_x))


def search_list_for_year(lst_elements, year):
    matching_elements = []
    for file in lst_elements:
        if year in file:
            matching_elements.append(file)
    return matching_elements


def master_corpora(final_directory, paths_lst, year):
    # print ('YEAR: ' + str(year) + ' list is ' + str(paths_lst))
    if len(paths_lst) == 0:
        pass

    elif len(paths_lst) == 1:
        corpus_1 = Corpus(filename=paths_lst[0])
        corpus_1.dump(final_directory + "wikiconv_corpus_merged_" + year)

    else:
        corpus_1 = Corpus(filename=paths_lst[0])
        corpus_2 = Corpus(filename=paths_lst[1])
        # merged_corpus = corpus_1.merge(corpus_2)
        merged_corpus = Corpus.merge(corpus_1, corpus_2)
        if len(paths_lst) > 2:
            for val in paths_lst[2:]:
                merged_corpus = merged_corpus.merge(merged_corpus, Corpus(filename=val))
        merged_corpus.dump(final_directory + "wikiconv_corpus_merged_" + str(year))


def is_empty_utterance(utterance):
    """check if an utterance is empty (both text and original are empty/None)"""
    # Check if text is empty or None
    text_is_empty = (
        not utterance.text or utterance.text.strip() == "" or utterance.text.strip() == "-"
    )

    # Check if original is empty or None
    # original = utterance.get('original')
    # original_is_empty = True

    # if original is not None:
    #     # If original exists, check if its text is empty
    #     original_text = getattr(original, 'text', None)
    #     if original_text:
    #         original_is_empty = False

    # Filter out only if BOTH are empty
    if text_is_empty and utterance.meta["original"] is None:
        # print("breaK")
        print(utterance.text)
        print(utterance.meta["original"])
    return text_is_empty and utterance.meta["original"] is None


def set_meta_dump_corpus(
    complete_utterance_list, master_conversation_level_dict, data_directory_output, corpus_name
):
    filtered_utterance_list = [u for u in complete_utterance_list if not is_empty_utterance(u)]

    print(
        f"Filtered {len(complete_utterance_list) - len(filtered_utterance_list)} empty utterances"
    )
    print(f"Remaining utterances: {len(filtered_utterance_list)}")

    if len(filtered_utterance_list) == 0:
        print(f"Warning: No utterances remaining for corpus {corpus_name}")
        return

    conversation_corpus = Corpus(utterances=filtered_utterance_list)
    # Set the conversation level meta data in the corpus
    for conversation in conversation_corpus.iter_conversations():
        conversation.meta = master_conversation_level_dict[conversation.id]
    conversation_corpus.dump(data_directory_output + corpus_name)

    # conversation_corpus = Corpus(utterances = complete_utterance_list)
    # #Set the conversation level meta data in the corpus
    # for conversation in conversation_corpus.iter_conversations():
    #     conversation.meta = master_conversation_level_dict[conversation.id]
    # conversation_corpus.dump(data_directory_output + corpus_name)


# Separate utterances into lists by year
def separate_by_year(individual_utterance_list):
    different_year_timestamps = set()
    dictionary_of_lists = {}

    for utterance in individual_utterance_list:
        timestamp_value = utterance.timestamp
        datetime_value = datetime.fromtimestamp(timestamp_value)
        year_value = datetime_value.year
        if year_value in dictionary_of_lists:
            dictionary_of_lists[year_value].append(utterance)
        else:
            dictionary_of_lists[year_value] = [utterance]

    return dictionary_of_lists


def create_utterances_from_individual_file(name_of_file_to_convert):
    list_of_utterances = []
    conversation_level_info = {}
    with open(name_of_file_to_convert, "r") as f:
        for line in f:
            dict_of_utterances = {}
            list_of_order_identification_values = []
            json_val = json.loads(line)

            # conversational level data
            conversation_id = json_val["conversation_id"]

            # Create the conversation meta values
            page_id = json_val["page_id"]
            page_title = json_val["page_title"]
            page_type = json_val["page_type"]
            conversation_meta_dict = {
                "page_id": page_id,
                "page_title": page_title,
                "page_type": page_type,
            }
            conversation_level_info[conversation_id] = conversation_meta_dict

            # Reformat each set of conversations in the final form
            comments = json_val["comments"]
            if len(comments) > 0:
                for comment_number in range(len(comments)):
                    comment_val = comments[comment_number]
                    if comment_number == 0:
                        section_header_value = True
                    else:
                        section_header_value = False
                    utterance_val, order_identification_val = reformat_comment(
                        comment_val, section_header_value
                    )

                    dict_of_utterances[utterance_val.id] = utterance_val
                    list_of_order_identification_values.append(order_identification_val)

                correct_order_of_comments = correct_comment_order(
                    list_of_order_identification_values
                )

                final_utterance_list = final_dict(dict_of_utterances, correct_order_of_comments)
                flat_utterance_list = [
                    utterance_sub for sublist in final_utterance_list for utterance_sub in sublist
                ]
                list_of_utterances.extend(flat_utterance_list)

    return (list_of_utterances, conversation_level_info)


# Return the correct dictionary order
def final_dict(dict_of_utterances, correct_order_identification_values):
    final_list = []
    if len(correct_order_identification_values) == 1:
        first_comment = correct_order_identification_values[0][0]
        final_list.append([dict_of_utterances[first_comment["id"]]])
        return final_list

    for list_of_comments in correct_order_identification_values:
        # print (list_of_comments)
        if len(list_of_comments) == 1:
            first_comment = list_of_comments[0]
            final_list.append([dict_of_utterances[first_comment["id"]]])
        else:
            # In all cases the original comment is the first comment in the thread (the original comment's modification/deletion/restoration will always be empty)
            original_comment_order_id = list_of_comments[0]
            original_utterance = dict_of_utterances[original_comment_order_id["id"]]
            # original_utterance.meta['original']  = original_utterance
            final_list.append([original_utterance])

            # Now fill out the modification/deletion/restoration objects and add to the final list if its an addition object
            for x in range(1, (len(list_of_comments))):
                current_comment_order_id = list_of_comments[x]
                current_comment_order_id_type = current_comment_order_id["type"].lower()
                # print (current_comment_order_id_type)
                if current_comment_order_id_type == "addition":
                    utterance_to_append = dict_of_utterances[current_comment_order_id["id"]]
                    final_list.append([utterance_to_append])

                for finalized_utterance_list in final_list:
                    for utterance in finalized_utterance_list:
                        # print ((current_comment_order_id['parent_id']))
                        # print('hiiiiiii')
                        if current_comment_order_id["parent_id"] == utterance.id:
                            original_utterance_value = copy.deepcopy(utterance)
                            comment_to_append = dict_of_utterances[current_comment_order_id["id"]]
                            utterance.meta[current_comment_order_id_type].append(comment_to_append)
                            if current_comment_order_id_type == "deletion":
                                rewrite_utterance_deletion(utterance, comment_to_append)
                            else:
                                rewrite_utterance_data(utterance, comment_to_append)
                            if utterance.meta["original"] is None:
                                # print("hm")
                                utterance.meta["original"] = utterance_to_dict(
                                    original_utterance_value
                                )
                        check_action_lists(
                            utterance,
                            current_comment_order_id,
                            dict_of_utterances,
                            current_comment_order_id_type,
                        )
                        convert_utterance_values_to_dict(
                            utterance,
                            current_comment_order_id,
                            dict_of_utterances,
                            current_comment_order_id_type,
                        )

    # return utterance_list
    return final_list


# If an utterance is deleted, make the final utterance text empty and change the id, speaker, reply-to and timestamp
def rewrite_utterance_deletion(utterance, comment_to_append):
    utterance.id = comment_to_append.id
    utterance.speaker = comment_to_append.speaker
    utterance.reply_to = comment_to_append.reply_to
    utterance.timestamp = comment_to_append.timestamp
    utterance.text = " "


# If an utterance is modified, make the final utteranxe text equal to the modified text and change the id, speaker, reply-to and timestamp
def rewrite_utterance_data(utterance, comment_to_append):
    utterance.id = comment_to_append.id
    utterance.speaker = comment_to_append.speaker
    utterance.reply_to = comment_to_append.reply_to
    utterance.timestamp = comment_to_append.timestamp
    utterance.text = comment_to_append.text


def create_utterance_list(list_of_comments):
    final_list_of_utterances = []
    for individual_comment_list in list_of_comments:
        ind_com = individual_comment_list[0]

        # Construct the speaker value of type Speaker Class
        speaker_dict = ind_com["speaker_info"]
        speaker_value = Speaker(
            id=speaker_dict["speaker"], meta={"speaker_id": speaker_dict["speaker_id"]}
        )

        # Construct the utterance value of type Utterance Class
        utterance_value = Utterance(
            ind_com["id"],
            speaker_value,
            ind_com["root"],
            ind_com["reply-to"],
            ind_com["timestamp"],
            ind_com["text"],
            None,
            meta=ind_com["meta"],
        )
        final_list_of_utterances.append(utterance_value)
    return final_list_of_utterances


# Check within the lists
def check_action_lists(
    utterance, current_comment_order_id, dict_of_utterances, current_comment_order_id_type
):
    modification_list = utterance.meta["modification"]
    deletion_list = utterance.meta["deletion"]
    restoration_list = utterance.meta["restoration"]
    comment_to_append = dict_of_utterances[current_comment_order_id["id"]]
    if len(modification_list) > 0:
        for utterance_val in modification_list:
            check_id_add(
                utterance,
                utterance_val,
                current_comment_order_id,
                current_comment_order_id_type,
                comment_to_append,
            )
    if len(deletion_list) > 0:
        for utterance_val in deletion_list:
            check_id_add(
                utterance,
                utterance_val,
                current_comment_order_id,
                current_comment_order_id_type,
                comment_to_append,
            )

    if len(restoration_list) > 0:
        for utterance_val in restoration_list:
            check_id_add(
                utterance,
                utterance_val,
                current_comment_order_id,
                current_comment_order_id_type,
                comment_to_append,
            )


# Convert the utterance's data into a dictionary
def utterance_to_dict(utterance_val):
    dict_rep = {}
    if type(utterance_val) == Utterance:
        dict_rep["id"] = utterance_val.id
        dict_rep["speaker"] = {
            "id": utterance_val.speaker.id,
            "speaker_id": utterance_val.speaker.meta["speaker_id"],
        }

        # Speaker(id = speaker, meta = {'speaker_id': speaker_id_val})
        dict_rep["root"] = utterance_val.conversation_id
        dict_rep["reply_to"] = utterance_val.reply_to
        dict_rep["timestamp"] = utterance_val.timestamp
        dict_rep["text"] = utterance_val.text
        dict_rep["meta_dict"] = utterance_val.meta
        # print (str((utterance_val.meta)))
        return dict_rep
    else:
        return None


def convert_utterance_values_to_dict(
    utterance, current_comment_order_id, dict_of_utterances, current_comment_order_id_type
):
    modification_list = utterance.meta["modification"]
    deletion_list = utterance.meta["deletion"]
    restoration_list = utterance.meta["restoration"]
    new_modification_list = []
    new_deletion_list = []
    new_restoration_list = []

    if len(modification_list) > 0:
        for utterance_val in modification_list:
            returned_val = utterance_to_dict(utterance_val)
            if returned_val is not None:
                new_modification_list.append(returned_val)
    if len(deletion_list) > 0:
        for utterance_val in deletion_list:
            returned_val = utterance_to_dict(utterance_val)
            if returned_val is not None:
                new_deletion_list.append(returned_val)
    if len(restoration_list) > 0:
        for utterance_val in restoration_list:
            returned_val = utterance_to_dict(utterance_val)
            if returned_val is not None:
                new_restoration_list.append(returned_val)

    utterance.meta["modification"] = new_modification_list
    utterance.meta["deletion"] = new_deletion_list
    utterance.meta["restoration"] = new_restoration_list


# Since top level data is stored as an utterance and metadata as a dictionary, uniform method to get the id
def get_id_from_utt_or_dict(utterance_val):
    # find the utterance value
    if isinstance(utterance_val, dict):
        id_val_utterance = utterance_val.get("id")
    else:
        id_val_utterance = utterance_val.id

    return id_val_utterance


# Add the comment to the action list
def check_id_add(
    utterance,
    utterance_val,
    current_comment_order_id,
    current_comment_order_id_type,
    comment_to_append,
):

    if current_comment_order_id["parent_id"] == get_id_from_utt_or_dict(
        utterance_val
    ) and check_not_in_list(utterance.meta[current_comment_order_id_type], comment_to_append):
        utterance.meta[current_comment_order_id_type].append(comment_to_append)
        if current_comment_order_id_type == "deletion":
            rewrite_utterance_deletion(utterance, comment_to_append)
        else:
            rewrite_utterance_data(utterance, comment_to_append)


# Make sure that this comment ahs not already been added
def check_not_in_list(list_of_values, utterance):
    utterance_id = utterance.id
    timestamp_val = utterance.timestamp
    for value in list_of_values:
        if get_id_from_utt_or_dict(value) == utterance_id and value.timestamp == timestamp_val:
            return False
    return True


# Find the correct order of comments to display
def correct_comment_order(list_of_order_identification_values):
    correct_order = []
    for value in list_of_order_identification_values:
        current_type = value["type"].lower()
        if current_type == "creation" or current_type == "addition":
            correct_order.append([value])
        elif (
            current_type == "modification"
            or current_type == "deletion"
            or current_type == "restoration"
        ):
            id_changed = value["parent_id"]
            for each_ordered_list in correct_order:
                if len(each_ordered_list) > 0:
                    for each_comment in each_ordered_list:
                        if each_comment["id"] == id_changed:
                            each_ordered_list.append(value)

    return correct_order


# Create common information doc from the json val and store as an utterance
def reformat_comment(json_val, section_header_value):
    # Top level/ required information
    id_val = json_val["id"]
    root_val = json_val["conversation_id"]
    reply_to = json_val["replyTo_id"]
    text = json_val["text"]
    timestamp = json_val["timestamp"]  # not required but placed in the top level

    # Access the infrmoation necessary for the speaker class
    speaker = json_val["user_text"]
    speaker_id_val = json_val["user_id"]

    # Construct the Speaker value of type Speaker Class
    speaker_value = Speaker(id=speaker, meta={"speaker_id": speaker_id_val})
    # speaker_value = {'id':speaker; 'speaker_id':speaker_id}

    # Values for the meta dictionary
    is_section_header = section_header_value
    indentation = json_val["indentation"]
    toxicity = json_val["toxicity"]
    sever_toxicity = json_val["sever_toxicity"]
    ancestor_id = json_val["ancestor_id"]
    rev_id = json_val["rev_id"]

    # Used to identify order
    parent_id = json_val["parent_id"]
    type_val = json_val["type"]

    # Build the meta dict
    meta_dict = {
        "is_section_header": is_section_header,
        "indentation": indentation,
        "toxicity": toxicity,
        "sever_toxicity": sever_toxicity,
        "ancestor_id": ancestor_id,
        "rev_id": rev_id,
        "parent_id": parent_id,
        "original": None,
        "modification": [],
        "deletion": [],
        "restoration": [],
    }

    # Construct the utterance value of type Utterance Class
    utterance_value = Utterance(
        id=id_val,
        speaker=speaker_value,
        conversation_id=root_val,
        reply_to=reply_to,
        timestamp=timestamp,
        text=text,
        meta=meta_dict,
    )
    order_identification = {
        "id": id_val,
        "parent_id": parent_id,
        "type": type_val,
        "timestamp": timestamp,
    }

    return (utterance_value, order_identification)


if __name__ == "__main__":
    main()
