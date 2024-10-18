import json
import re
from collections import defaultdict
from typing import Dict, List, Callable, Pattern, Optional

STOP_TOK = "*"


def load_ngram_markers(ngram_path: str) -> Dict[str, Dict]:
    """
    Load ngram markers from file
    """

    # load markers
    with open(ngram_path, "r") as f:
        data = json.load(f)

    # organize markers into a trie
    trie = defaultdict(dict)
    for strategy, ngrams in data.items():
        for ngram in ngrams:
            words = ngram.split()
            # insert
            curr = trie
            for word in words:
                curr = curr.setdefault(word, {})
            # end with strategy name
            curr[STOP_TOK] = strategy

    return trie


def extract_ngram_markers_given_start(
    words: List[str], sent_idx: int, i: int, ngrams: Dict[str, Dict]
) -> Dict[str, List]:
    """
    Extract strategies based on ngram markers, starting from the given position
    """
    strategies = defaultdict(list)
    curr, j = ngrams, i

    while j < len(words):
        word = words[j]
        if word not in curr:
            break
        curr = curr[word]
        if STOP_TOK in curr:
            extracted = [(words[k], sent_idx, k) for k in range(i, j + 1)]
            strategies[curr[STOP_TOK]].append(extracted)
        j += 1

    return strategies


def extract_ngram_markers(
    words: List[str], sent_idx: int, ngrams: Dict[str, Dict], offset: int = 0
) -> Dict[str, List]:
    """
    Extract strategies uses (with marker positions) from the parse of a sentence
    """
    strategies = defaultdict(list)
    for i in range(offset, len(words)):
        extracted = extract_ngram_markers_given_start(words, sent_idx, i, ngrams)
        for k, v in extracted.items():
            strategies[k].extend(v)

    return strategies


def extract_starter_markers(
    words: List[str], sent_idx: int, ngrams: Dict[str, Dict]
) -> Dict[str, List]:
    """
    Extract markers for sentence-starting strategies
    """
    return extract_ngram_markers_given_start(words, sent_idx, 0, ngrams)


def extract_regex_strategies(pattern: Pattern, tokens: List[str], sent_idx: int, offset: int = 0):
    """
    Extract markers for a given strategy based on regex patterns
    """

    # find matches
    sent = " ".join(tokens[offset:])
    matches = [match.span() for match in re.finditer(pattern, sent)]

    extracted = []
    for match_start, match_end in matches:
        # idx of starting token of the matched span
        tok_start = len(sent[0:match_start].split())
        # idx of ending token
        tok_end = len(sent[0:match_end].split())
        extracted_toks = [
            (tokens[i + offset], sent_idx, i + offset) for i in range(tok_start, tok_end)
        ]
        extracted.append(extracted_toks)

    return extracted


def extract_markers_from_sent(
    sent_parsed: List[Dict],
    sent_idx: int,
    ngrams: Optional[Dict[str, Dict]] = None,
    starters: Optional[Dict[str, Dict]] = None,
    non_starters: Optional[Dict[str, Dict]] = None,
    marker_fns: Optional[List[Callable[..., Dict[str, List]]]] = None,
    names: Optional[List[str]] = None,
) -> Dict[str, List]:
    """
    Extracting markers from a parsed sentence

    :param sent_parsed: parsed sentence
    :param sent_idx: idx of the current sentence in the utterance
    :param ngrams: ngram markers
    :param starters: starter markers
    :param non_starters:
    :param marker_fns: list of functions to extract strategies not covered by lexicons
    """

    sent_summary = defaultdict(list)
    words = [x["tok"].lower() for x in sent_parsed]

    # ngrams markers
    if ngrams:
        ngram_markers = extract_ngram_markers(words, sent_idx, ngrams)
        for k, ngrams in ngram_markers.items():
            sent_summary[k].extend(ngrams)

    # starter markers
    if starters:
        starter_markers = extract_starter_markers(words, sent_idx, starters)
        for k, starters in starter_markers.items():
            sent_summary[k].extend(starters)

    # non-starter strategies
    if non_starters:
        non_starters = extract_ngram_markers(words, sent_idx, non_starters, offset=1)
        for k, starters in non_starters.items():
            sent_summary[k].extend(starters)

    # update results for strategies that have other special conditions
    if marker_fns:
        if not names:
            names = [fn.__name__ for fn in marker_fns]
        for name, fn in zip(names, marker_fns):
            extracted = fn(sent_parsed, sent_idx)
            for markers in extracted:
                sent_summary[name].extend(markers)

    return sent_summary
