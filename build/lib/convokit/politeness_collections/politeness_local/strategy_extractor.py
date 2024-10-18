import json
import os
import re
from itertools import chain
import pkg_resources
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from convokit.politeness_collections.marker_utils import (
    load_ngram_markers,
    extract_ngram_markers,
    extract_markers_from_sent,
)

LEXICON_DIR = "politeness_collections/politeness_local/lexicons"

ngram_path = pkg_resources.resource_filename(
    "convokit", os.path.join(LEXICON_DIR, "ngram_markers.json")
)

starter_path = pkg_resources.resource_filename(
    "convokit", os.path.join(LEXICON_DIR, "starter_markers.json")
)

non_starter_path = pkg_resources.resource_filename(
    "convokit", os.path.join(LEXICON_DIR, "non_starter_markers.json")
)

# strategy functions


def extract_dep_parse_markers(
    sent_parsed: List[Dict],
    sent_idx: int,
    child: str,
    parents: Optional[Set] = None,
    relation: Optional[str] = None,
):
    matched = []

    for i, tok in enumerate(sent_parsed):
        # tok matches
        if tok["tok"].lower() == child:
            # check relation (if applicable)
            if not relation or tok["dep"] == relation:
                # check parent (if applicable)
                if not parents:
                    matched.append((tok["tok"], sent_idx, i))

                elif tok["dep"] != "ROOT" and sent_parsed[tok["up"]]["tok"] in parents:
                    # keep both child and parent
                    matched.extend(
                        [
                            (tok["tok"], sent_idx, i),
                            (sent_parsed[tok["up"]]["tok"], sent_idx, tok["up"]),
                        ]
                    )

    return matched


def actually(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    # two types of matches
    cond1 = extract_dep_parse_markers(
        sent_parsed, sent_idx, "the", parents={"point", "reality", "truth"}, relation="det"
    )
    cond2 = extract_dep_parse_markers(
        sent_parsed, sent_idx, "fact", parents={"in"}, relation="pobj"
    )

    return cond1 + cond2


def adv_just(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    return extract_dep_parse_markers(sent_parsed, sent_idx, "just", relation="advmod")


def apology(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    cond1 = extract_dep_parse_markers(
        sent_parsed, sent_idx, "me", parents={"forgive", "excuse"}, relation="dobj"
    )
    cond2 = extract_dep_parse_markers(
        sent_parsed, sent_idx, "i", parents={"apologize"}, relation="nsubj"
    )
    return cond1 + cond2


def gratitude(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    return extract_dep_parse_markers(
        sent_parsed, sent_idx, "i", parents={"appreciate"}
    ) + extract_dep_parse_markers(sent_parsed, sent_idx, "we", parents={"appreciate"})


def swearing(sent_parsed: List[Dict], sent_idx: int) -> Dict[str, List]:
    return extract_dep_parse_markers(
        sent_parsed, sent_idx, "the", parents=["fuck", "hell", "heck"], relation="det"
    )


# all strategies being considered
STRATEGIES = [
    "Actually",
    "Adverb.Just",
    "Affirmation",
    "Apology",
    "By.The.Way",
    "Conj.Start",
    "Filler",
    "For.Me",
    "For.You",
    "Gratitude",
    "Greeting",
    "Hedges",
    "Indicative",
    "Please",
    "Please.Start",
    "Reassurance",
    "Subjunctive",
    "Swearing",
]

# different types of markers
NGRAMS = load_ngram_markers(ngram_path)
STARTERS = load_ngram_markers(starter_path)
NON_STARTERS = load_ngram_markers(non_starter_path)
MARKER_FNS = [actually, adv_just, apology, gratitude, swearing]
NAMES = ["Actually", "Adverb.Just", "Apology", "Gratitude", "Swearing"]


def get_local_politeness_strategy_features(
    parses: List[List],
) -> Tuple[Dict[str, int], Dict[str, List[Tuple]]]:
    """
    Extract strategies given a parsed utterance
    """

    markers = {k: [] for k in STRATEGIES}

    for sent_idx, sent_parsed in enumerate(parses):
        sent_markers = extract_markers_from_sent(
            sent_parsed, sent_idx, NGRAMS, STARTERS, NON_STARTERS, MARKER_FNS, NAMES
        )
        # update markers
        for k, v in sent_markers.items():
            markers[k].extend(v)

    # binary features
    features = {k: int(len(marker_list) > 0) for k, marker_list in markers.items()}

    return features, markers
