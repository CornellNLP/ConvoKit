import json
import os
import re
from itertools import chain
import pkg_resources
from collections import defaultdict
from typing import Dict, List, Tuple

from convokit.politeness_collections.marker_utils import (
    load_ngram_markers,
    extract_regex_strategies,
    extract_markers_from_sent,
)

LEXICON_DIR = "politeness_collections/politeness_cscw_zh/lexicons"

ngram_path = pkg_resources.resource_filename(
    "convokit", os.path.join(LEXICON_DIR, "ngram_markers.json")
)

starter_path = pkg_resources.resource_filename(
    "convokit", os.path.join(LEXICON_DIR, "starter_markers.json")
)

non_starter_path = pkg_resources.resource_filename(
    "convokit", os.path.join(LEXICON_DIR, "non_starter_markers.json")
)


PLEASE_PATTERN = re.compile(r"([烦劳还]?\s?请)|([烦劳]您)")
START_QN_PATTERN = re.compile(r"^\W*([为凭]什么\s?|几\s?|哪\s?|多少\s?|怎\s?|谁\s?|咋\s?)")
CAN_YOU_PATTERN = re.compile(r"[你您]\s?[是可想觉要].+?[吗呢呀？]")
COULD_YOU_PATTERN = re.compile(r"[你您]\s?(?P<A>[可想觉要])\s?不\s?(?P=A)")


# strategy functions (regex)
def please(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    tokens = [x["tok"] for x in sent_parsed]
    return extract_regex_strategies(PLEASE_PATTERN, tokens, sent_idx, offset=1)


def start_question(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    tokens = [x["tok"] for x in sent_parsed]
    return extract_regex_strategies(START_QN_PATTERN, tokens, sent_idx)


def can_you(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    tokens = [x["tok"] for x in sent_parsed]
    return extract_regex_strategies(CAN_YOU_PATTERN, tokens, sent_idx)


def could_you(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    tokens = [x["tok"] for x in sent_parsed]
    return extract_regex_strategies(COULD_YOU_PATTERN, tokens, sent_idx)


# full list of strategies
STRATEGIES = [
    "apologetic",
    "best_wishes",
    "can_you",
    "could_you",
    "emergency",
    "factuality",
    "first_person_plural",
    "first_person_singular",
    "gratitude",
    "greeting",
    "hedge",
    "honorifics",
    "indirect_btw",
    "ingroup_iden",
    "praise",
    "promise",
    "please",
    "start_i",
    "start_please",
    "start_question",
    "start_so",
    "start_you",
    "taboo",
    "together",
    "you_direct",
    "you_honorific",
]


# different types of markers
NGRAMS = load_ngram_markers(ngram_path)
STARTERS = load_ngram_markers(starter_path)
NON_STARTERS = load_ngram_markers(non_starter_path)
MARKER_FNS = [please, start_question, can_you, could_you]


def get_chinese_politeness_strategy_features(
    parses: List[List],
) -> Tuple[Dict[str, int], Dict[str, List[Tuple]]]:
    """
    Extract strategies given a parsed utterance
    """

    markers = {k: [] for k in STRATEGIES}

    for sent_idx, sent_parsed in enumerate(parses):
        sent_markers = extract_markers_from_sent(
            sent_parsed, sent_idx, NGRAMS, STARTERS, NON_STARTERS, MARKER_FNS
        )
        # update markers
        for k, v in sent_markers.items():
            markers[k].extend(v)

    # binary features
    features = {k: int(len(marker_list) > 0) for k, marker_list in markers.items()}

    return features, markers
