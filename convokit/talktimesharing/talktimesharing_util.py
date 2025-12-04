import re
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from convokit import Corpus

random.seed(42)


def _tokenize(text):
    text = text.lower()
    text = re.findall("[a-z]+", text)
    return text


def _longer_than_xwords(corpus, utt_id, min_utt_words, x=None):
    """
    Returns True if the utterance has at least x words (defaulting to min_utt_words);
    otherwise, returns False.
    """
    if x is None:
        x = min_utt_words
    utt = corpus.get_utterance(utt_id)
    return len(_tokenize(utt.text)) >= x


def _rhythm_count_utt_time(corpus, utt_lst, min_utt_words):
    """
    Calculates total speaking time for each speaker group from a list of utterance IDs.

    Filters out utterances shorter than min_utt_words and returns the cumulative speaking
    time (in seconds) for groupA and groupB.
    """
    valid_utt = [utt_id for utt_id in utt_lst if _longer_than_xwords(corpus, utt_id, min_utt_words)]
    if len(valid_utt) == 0:
        return 0, 0
    time_A, time_B = 0, 0
    for utt_id in valid_utt:
        utt = corpus.get_utterance(utt_id)
        if utt.meta["utt_group"] == "groupA":
            time_A += utt.meta["stop"] - utt.meta["start"]
        elif utt.meta["utt_group"] == "groupB":
            time_B += utt.meta["stop"] - utt.meta["start"]
    return time_A, time_B


def _get_ps(corpus, convo, remove_first_last_utt, min_utt_words, primary_threshold):
    """
    Determines the primary speaker group in a conversation based on speaking time of each speaker group.

    Returns 'groupA' or 'groupB' if one group exceeds the primary_threshold proportion of total speaking time;
    otherwise, returns None.
    """
    assert primary_threshold > 0.5, "Primary Threshold should greater than 0.5"
    if remove_first_last_utt:
        utt_lst = convo.get_utterance_ids()[1:-1]
    else:
        utt_lst = convo.get_utterance_ids()
    time_A, time_B = _rhythm_count_utt_time(corpus, utt_lst, min_utt_words)
    total_speaking_time = time_A + time_B
    if time_A > (total_speaking_time * primary_threshold):
        return "groupA"
    elif time_B > (total_speaking_time * primary_threshold):
        return "groupB"
    else:
        return None


def _sliding_window(
    corpus, convo_id, window_size, sliding_size, remove_first_last_utt, min_utt_words
):
    """
    Computes sliding window segments of a conversation and calculates total speaking time
    for each speaker group (groupA and groupB) within each window.

    Returns a list of dictionaries, each containing the speaking time per group for a window.
    """
    convo = corpus.get_conversation(convo_id)
    if remove_first_last_utt:
        utt_lst = convo.get_utterance_ids()[1:-1]
    else:
        utt_lst = convo.get_utterance_ids()
    utt_lst = [utt_id for utt_id in utt_lst if _longer_than_xwords(corpus, utt_id, min_utt_words)]
    all_windows = []
    cur_start_time = corpus.get_utterance(utt_lst[0]).meta["start"]
    cur_end_time = cur_start_time + (
        window_size * 60
    )  # window_size is in minutes, converted to seconds
    prev_window_last_utt_id = utt_lst[0]
    convo_end_time = corpus.get_utterance(utt_lst[-1]).meta["stop"]

    while prev_window_last_utt_id != utt_lst[-1] and cur_end_time < convo_end_time:
        cur_window_groupA_speaking_time = 0
        cur_window_groupB_speaking_time = 0

        for i, utt_id in enumerate(utt_lst):
            utt = corpus.get_utterance(utt_id)
            # case 1: utterances in previous windows and not in current window at all
            if utt.meta["stop"] < cur_start_time:
                continue

            # case 2: last utt of the current window
            if utt.meta["stop"] > cur_end_time:
                # the entire utt not in the window, meaning previous utt is in the window and this one is not
                if utt.meta["start"] > cur_end_time:
                    prev_window_last_utt_id = utt_lst[i - 1]
                # special case: the utt span longer than the entire window
                elif utt.meta["start"] < cur_start_time:
                    if utt.meta["utt_group"] == "groupA":
                        cur_window_groupA_speaking_time += cur_end_time - cur_start_time
                    elif utt.meta["utt_group"] == "groupB":
                        cur_window_groupB_speaking_time += cur_end_time - cur_start_time
                    prev_window_last_utt_id = utt_id
                # part of the utt in the window
                else:
                    if utt.meta["utt_group"] == "groupA":
                        cur_window_groupA_speaking_time += cur_end_time - utt.meta["start"]
                    elif utt.meta["utt_group"] == "groupB":
                        cur_window_groupB_speaking_time += cur_end_time - utt.meta["start"]
                    prev_window_last_utt_id = utt_id
                # put window data in all_windows only at the terminating point: last utt of the window
                all_windows.append(
                    {
                        "groupA": cur_window_groupA_speaking_time,
                        "groupB": cur_window_groupB_speaking_time,
                    }
                )
                break

            # case 3: utterances in the window but not the last utterance of the window
            if utt.meta["stop"] > cur_start_time:
                # part of the utt in window
                if utt.meta["start"] < cur_start_time and utt.meta["stop"] > utt.meta["start"]:
                    if utt.meta["utt_group"] == "groupA":
                        cur_window_groupA_speaking_time += utt.meta["stop"] - cur_start_time
                    elif utt.meta["utt_group"] == "groupB":
                        cur_window_groupB_speaking_time += utt.meta["stop"] - cur_start_time
                # entire utt in window
                else:
                    if utt.meta["utt_group"] == "groupA":
                        cur_window_groupA_speaking_time += utt.meta["stop"] - utt.meta["start"]
                    elif utt.meta["utt_group"] == "groupB":
                        cur_window_groupB_speaking_time += utt.meta["stop"] - utt.meta["start"]

        # update window start end time
        cur_start_time += sliding_size
        cur_end_time += sliding_size

    return all_windows


def _convo_balance_score(corpus, convo_id, remove_first_last_utt, min_utt_words):
    """
    Computes the overall balance score of a conversation based on speaking time.

    Returns the proportion of speaking time for the more dominant group (groupA or groupB), or None if total speaking time is zero.
    """
    convo = corpus.get_conversation(convo_id)
    if remove_first_last_utt:
        utt_lst = convo.get_utterance_ids()[1:-1]
    else:
        utt_lst = convo.get_utterance_ids()
    timeA, timeB = _rhythm_count_utt_time(corpus, utt_lst, min_utt_words)
    total_time = timeA + timeB
    if total_time == 0:
        return None
    return timeA / total_time if timeA >= timeB else timeB / total_time


def _convo_balance_lst(
    corpus,
    convo_id,
    window_ps_threshold,
    window_ss_threshold,
    window_size,
    sliding_size,
    remove_first_last_utt,
    min_utt_words,
):
    """
    Generates a list representing local talk-time sharing dynamics across sliding windows in a conversation.

    Each value in the list is 1 (primary speaker dominance), -1 (secondary speaker dominance), or 0 (balanced),
    based on whether the dominant group exceeds the window_ps_threshold within that window.
    """
    groups = _sliding_window(
        corpus, convo_id, window_size, sliding_size, remove_first_last_utt, min_utt_words
    )
    balance_lst = []
    no_speaking_time_count = 0
    all_window_count = 0
    convo = corpus.get_conversation(convo_id)
    for window in groups:
        all_window_count += 1
        window_ps_time = window[convo.meta["primary_speaker"]]
        window_ss_time = window[convo.meta["secondary_speaker"]]
        window_total_time = window_ps_time + window_ss_time
        window_id = 0
        if window_total_time == 0:
            window_id = -100
            no_speaking_time_count += 1
            continue
        elif window_ps_time / window_total_time > window_ps_threshold:
            window_id = window_ps_time / window_total_time
        elif window_ss_time / window_total_time > window_ss_threshold:
            window_id = -1 * window_ss_time / window_total_time

        if window_id == 0:
            balance_lst.append(0)
        elif window_id > 0:
            balance_lst.append(1)
        else:
            balance_lst.append(-1)
    return balance_lst


def plot_color_blocks(data_dict, block_length=0.5, plot_name=None):
    """
    Visualizes conversation dynamics as a horizontal sequence of colored blocks.

    Each block represents a window: blue for primary speaker dominance, red for secondary, and grey for balance.
    Block opacity reflects the strength of dominance. Optionally saves the plot to a file.
    """
    convo_id = list(data_dict.keys())[0]
    data = data_dict[convo_id]
    fig, ax = plt.subplots(figsize=(10, 2))

    # Plot each block, the higher absolute value of "value", darker the block
    for i, value in enumerate(data):
        if value > 0:  # plot blue
            ax.add_patch(
                plt.Rectangle((i * block_length, 0), block_length, 0.2, color=(0, 0, 1, value))
            )
        elif value < 0:  # plot orange
            ax.add_patch(
                plt.Rectangle((i * block_length, 0), block_length, 0.2, color=(1, 0, 0, -value))
            )
        elif value == 0:  # plot lightgrey
            ax.add_patch(plt.Rectangle((i * block_length, 0), block_length, 0.2, color="lightgrey"))
    ax.set_xlim(0, len(data) * block_length)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    # ax.text(0, 0.22, f"{convo_id}", fontsize=12, ha='left')
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()


def plot_color_blocks_multi(data_lists, block_length=0.5, plot_name=None):
    """
    Plots multiple conversations' windowed talk-time sharing dynamics as side-by-side color block visualizations.
    Each subplot represents one conversation.
    """
    num_lists = len(data_lists)
    num_columns = 2
    num_rows = (num_lists + 1) // num_columns

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(8, num_rows * 0.5))

    axs = axs.flatten()

    # Plot each list in its own subplot
    for idx, data_dict in enumerate(data_lists):
        # Plot each block, the higher absolute value of "value", darker the block
        for convo_id, data in data_dict.items():
            for i, value in enumerate(data):
                if value > 0:  # plot blue
                    axs[idx].add_patch(
                        plt.Rectangle(
                            (i * block_length, 0), block_length, 0.5, color=(0, 0, 1, value)
                        )
                    )
                elif value < 0:  # plot red
                    axs[idx].add_patch(
                        plt.Rectangle(
                            (i * block_length, 0), block_length, 0.5, color=(1, 0, 0, -value)
                        )
                    )
                elif value == 0:  # plot lightgrey
                    axs[idx].add_patch(
                        plt.Rectangle((i * block_length, 0), block_length, 0.5, color="lightgrey")
                    )
                else:
                    print("invalid other case")
            axs[idx].set_xlim(0, len(data) * block_length)
            axs[idx].set_ylim(0, 0.1)
            axs[idx].set_aspect("auto")
            axs[idx].axis("off")
            # axs[idx].text(0, -0.02, f"{convo_id}", fontsize=7, ha='left')

    if num_lists % num_columns:
        axs[-1].axis("off")

    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()


def _plot_individual_conversation_floors(
    corpus,
    convo_id,
    window_ps_threshold,
    window_ss_threshold,
    window_size,
    sliding_size,
    remove_first_last_utt,
    min_utt_words,
    plot_name=None,
):
    """
    Visualizes turn-taking dominance in a single conversation using color-coded windowed balance scores.

    Applies a sliding window over the conversation to compute talk-time balance between the primary and
    secondary speaker groups, then plots the resulting sequence as colored blocks.
    """
    groups = _sliding_window(
        corpus,
        convo_id,
        window_size=window_size,
        sliding_size=sliding_size,
        remove_first_last_utt=remove_first_last_utt,
        min_utt_words=min_utt_words,
    )
    convo_plot_lst = []
    score_lst = []
    convo = corpus.get_conversation(convo_id)
    ps = convo.meta["primary_speaker"]
    ss = convo.meta["secondary_speaker"]
    for window in groups:
        window_ps_time = window[ps]
        window_ss_time = window[ss]
        window_total_time = window_ps_time + window_ss_time
        window_id = 0
        if window_total_time == 0:  # No Speaking Time in the window
            window_id = -100
            continue  # skipping no speaking time windows for now
            # no_speaking_time_count += 1
        elif window_ps_time / window_total_time > window_ps_threshold:
            window_id = window_ps_time / window_total_time
        elif window_ss_time / window_total_time > window_ss_threshold:
            window_id = -1 * window_ss_time / window_total_time

        convo_plot_lst.append(window_id)
        score_lst.append(round(window_id, 2))
    plot_color_blocks({convo_id: convo_plot_lst}, plot_name=plot_name)
    try:
        print(
            f"red : {round(convo.meta['percent_red'], 2)}, blue : {round(convo.meta['percent_blue'], 2)}, gray : {round(convo.meta['percent_gray'], 2)}"
        )
    except:
        pass


def _plot_multi_conversation_floors(
    corpus,
    convo_id_lst,
    window_ps_threshold,
    window_ss_threshold,
    window_size,
    sliding_size,
    remove_first_last_utt,
    min_utt_words,
    plot_name=None,
):
    """
    Generates side-by-side visualizations of turn-taking dynamics across multiple conversations.

    For each conversation, computes sliding window balance scores between primary and secondary speakers,
    and plots the results as color-coded block sequences. Optionally saves the combined visualization.
    """
    result_lst = []
    for convo_id in convo_id_lst:
        groups = _sliding_window(
            corpus,
            convo_id,
            window_size=window_size,
            sliding_size=sliding_size,
            remove_first_last_utt=remove_first_last_utt,
            min_utt_words=min_utt_words,
        )
        convo = corpus.get_conversation(convo_id)
        ps = convo.meta["primary_speaker"]
        ss = convo.meta["secondary_speaker"]
        convo_plot_lst = []
        for window in groups:
            window_ps_time = window[ps]
            window_ss_time = window[ss]
            window_total_time = window_ps_time + window_ss_time
            window_id = 0
            if window_total_time == 0:  # No Speaking Time in the window
                window_id = -100
                continue  # skipping no speaking time windows for now
                # no_speaking_time_count += 1
            elif window_ps_time / window_total_time > window_ps_threshold:
                window_id = window_ps_time / window_total_time
            elif window_ss_time / window_total_time > window_ss_threshold:
                window_id = -1 * window_ss_time / window_total_time

            convo_plot_lst.append(window_id)
        result_lst.append({convo_id: convo_plot_lst})
    plot_color_blocks_multi(result_lst, plot_name=plot_name)
