import pkg_resources
import os

#####
# Word lists

hedges = [
    "think",
    "thought",
    "thinking",
    "almost",
    "apparent",
    "apparently",
    "appear",
    "appeared",
    "appears",
    "approximately",
    "around",
    "assume",
    "assumed",
    "certain amount",
    "certain extent",
    "certain level",
    "claim",
    "claimed",
    "doubt",
    "doubtful",
    "essentially",
    "estimate",
    "estimated",
    "feel",
    "felt",
    "frequently",
    "from our perspective",
    "generally",
    "guess",
    "in general",
    "in most cases",
    "in most instances",
    "in our view",
    "indicate",
    "indicated",
    "largely",
    "likely",
    "mainly",
    "may",
    "maybe",
    "might",
    "mostly",
    "often",
    "on the whole",
    "ought",
    "perhaps",
    "plausible",
    "plausibly",
    "possible",
    "possibly",
    "postulate",
    "postulated",
    "presumable",
    "probable",
    "probably",
    "relatively",
    "roughly",
    "seems",
    "should",
    "sometimes",
    "somewhat",
    "suggest",
    "suggested",
    "suppose",
    "suspect",
    "tend to",
    "tends to",
    "typical",
    "typically",
    "uncertain",
    "uncertainly",
    "unclear",
    "unclearly",
    "unlikely",
    "usually",
    "broadly",
    "tended to",
    "presumably",
    "suggests",
    "from this perspective",
    "from my perspective",
    "in my view",
    "in this view",
    "in our opinion",
    "in my opinion",
    "to my knowledge",
    "fairly",
    "quite",
    "rather",
    "argue",
    "argues",
    "argued",
    "claims",
    "feels",
    "indicates",
    "supposed",
    "supposes",
    "suspects",
    "postulates",
]

# Positive and negative words from Liu
pos_filename = pkg_resources.resource_filename(
    "convokit", os.path.join("data", "liu-positive-words.txt")
)
neg_filename = pkg_resources.resource_filename(
    "convokit", os.path.join("data", "liu-negative-words.txt")
)


positive_words = set(map(lambda x: x.strip(), open(pos_filename).read().splitlines()))
negative_words = set(
    map(lambda x: x.strip(), open(neg_filename, encoding="ISO-8859-1").read().splitlines())
)

#####
# Lambda Functions

please = lambda p: check_word([{"tok": "_"}] + p[1:], ["please"])
please.__name__ = "Please"

please_start = lambda p: check_word_at(p, 0, tok=["please"])
please_start.__name__ = "Please start"

has_hedge = lambda p: check_word(p, tok=hedges)
has_hedge.__name__ = "HASHEDGE"

btw = lambda p: check_word_at(p, 2, up_tok=["by"], tok=["way"], dep=["pobj"])
btw.__name__ = "Indirect (btw)"

hashedges = lambda p: check_word(p, dep=["nsubj"], up_tok=hedges)
hashedges.__name__ = "Hedges"

factuality = lambda p: combine_results(
    [
        check_word(p, up_tok=["in"], tok=["fact"], dep=["pobj"]),
        check_word(
            p,
            tok=["the"],
            up_tok=["point", "reality", "truth"],
            dep=["det"],
            precede=["point", "reality", "truth"],
        ),
        check_word(p, tok=["really", "actually", "honestly", "surely"]),
    ]
)
factuality.__name__ = "Factuality"

deference = lambda p: check_word_at(
    p, 0, tok=["great", "good", "nice", "good", "interesting", "cool", "excellent", "awesome"]
)
deference.__name__ = "Deference"

gratitude = lambda p: combine_results(
    [check_word(p, tok=["thank", "thanks"]), check_word(p, tok=["i"], up_tok=["appreciate"])]
)
gratitude.__name__ = "Gratitude"

apologize = lambda p: combine_results(
    [
        check_word(p, tok=["sorry", "woops", "oops"]),
        check_word(p, tok=["i"], up_tok=["apologize"], dep=["nsubj"]),
        check_word(p, tok=["me"], up_tok=["forgive", "excuse"], dep=["dobj"]),
    ]
)
apologize.__name__ = "Apologizing"

groupidentity = lambda p: check_word(p, tok=["we", "our", "us", "ourselves"])
groupidentity.__name__ = "1st person pl."

firstperson = lambda p: check_word([{"tok": "_"}] + p[1:], tok=["i", "my", "mine", "myself"])
firstperson.__name__ = "1st person"

firstperson_start = lambda p: check_word_at(p, 0, tok=["i", "my", "mine", "myself"])
firstperson_start.__name__ = "1st person start"

secondperson = lambda p: check_word(
    [{"tok": "_"}] + p[1:], tok=["you", "your", "yours", "yourself"]
)
secondperson.__name__ = "2nd person"

secondperson_start = lambda p: check_word_at(p, 0, tok=["you", "your", "yours", "yourself"])
secondperson_start.__name__ = "2nd person start"

hello = lambda p: check_word_at(p, 0, tok=["hi", "hello", "hey"])
hello.__name__ = "Indirect (greeting)"

why = lambda p: check_word(p[:2], tok=["what", "why", "who", "how"])
why.__name__ = "Direct question"

conj = lambda p: check_word_at(p, 0, tok=["so", "then", "and", "but", "or"])
conj.__name__ = "Direct start"

has_positive = lambda p: check_word(p, tok=positive_words)
has_positive.__name__ = "HASPOSITIVE"

has_negative = lambda p: check_word(p, tok=negative_words)
has_negative.__name__ = "HASNEGATIVE"

subjunctive = lambda p: check_word(p, tok=["could", "would"], precede=["you"])
subjunctive.__name__ = "SUBJUNCTIVE"

indicative = lambda p: check_word(p, tok=["can", "will"], precede=["you"])
indicative.__name__ = "INDICATIVE"


#####
# Helper functions and variables


def combine_results(lst):
    """
    combines list of results
    ex: [[1, ["hey", 0]], [0,[]], [1, ["you", 1]]] -> [1, [["hey", 0],["you", 1]]]
    """
    a = 0
    b = []
    for x in lst:
        a = max(a, x[0])
        if x[1] != []:
            b += x[1]
    return a, b


def check_word_at(p, ind, tok=None, dep=None, up_tok=None, up_dep=None, precede=None):
    """
    Returns an indicator and a marker
    If parameters match word at index:
        returns 1, [tok at ind, ind]
    Else:
        returns 0, []
    """
    if len(p) <= ind:
        return 0, []
    if tok != None and p[ind]["tok"].lower() not in tok:
        return 0, []
    if dep != None and p[ind]["dep"] not in dep:
        return 0, []
    if up_tok != None and ("up" not in p[ind] or p[p[ind]["up"]]["tok"].lower() not in up_tok):
        return 0, []
    if up_dep != None and p[p[ind]["up"]]["dep"] not in up_dep:
        return 0, []
    if precede != None and (len(p) <= ind + 1 or p[ind + 1]["tok"] not in precede):
        return 0, []
    return 1, [[(p[ind]["tok"], ind)]]


def check_word(p, tok=None, dep=None, up_tok=None, up_dep=None, precede=None):
    """
    Returns an indicator and a marker
    If parameters match any word in the sentence:
        returns 1, markers for each occurance
    Else:
        returns 0, []
    """
    toks = []
    for ind, x in enumerate(p):
        if tok != None and x["tok"].lower() not in tok:
            continue
        if dep != None and x["dep"] not in dep:
            continue
        if up_tok != None and ("up" not in x or p[x["up"]]["tok"].lower() not in up_tok):
            continue
        if up_dep != None and p[x["up"]]["dep"] not in up_dep:
            continue
        if precede != None and (len(p) <= ind + 1 or p[ind + 1]["tok"] not in precede):
            continue
        if up_tok != None:
            toks += [[(x["tok"], ind), (p[x["up"]]["tok"].lower(), x["up"])]]
        else:
            toks += [[(x["tok"], ind)]]
    if toks != []:
        return 1, toks
    else:
        return 0, []


# Feature function list
F = [
    please,
    please_start,
    has_hedge,
    btw,
    hashedges,
    factuality,
    deference,
    gratitude,
    apologize,
    groupidentity,
    firstperson,
    firstperson_start,
    secondperson,
    secondperson_start,
    hello,
    why,
    conj,
    has_positive,
    has_negative,
    subjunctive,
    indicative,
]

fnc2feature_name = lambda f, keys: [key + "_==%s==" % f.__name__.replace(" ", "_") for key in keys]


def get_politeness_strategy_features(parses):
    """
    :param utt- the utterance to be processed
    :type utterance- Object with attributes including text and meta

        utt.meta is a dictionary with the following form:
        {
            parsed: [
                { 'rt': 5
                  'toks': [{'dep': 'intj', 'dn': [], 'tag': 'UH', 'tok': 'hello', 'up': 2}, #sent 1, word 1
                          ... {sent 1 word 2} ,{sent 1 word 3}...]

                },
                { 'rt': 12
                'toks': [{'dep': 'nsubj', 'dn': [], 'tag': 'PRP', 'tok': 'i', 'up': 1}, # sent 2, word 1
                         {'dep': 'ROOT', 'dn': [0, 2, 3], 'tag': 'VBP', 'tok': 'need'},
                         ...]
                }
            ]
        }
    Returns- feature dictionary and marker dictionary
    feature dictionary:
        {
            feature_name: 1 or 0
        }
    marker dictionary:
        {
            marker_name: list of [token, sentence index, word index]
        }
    """

    # build dictionary
    features = {}
    markers = {}
    for fnc in F:
        f = fnc2feature_name(fnc, ["feature_politeness", "politeness_markers"])
        features[f[0]] = 0
        markers[f[1]] = []

    # runs lambda functions
    for sent_ind, sentence in enumerate(parses):
        for fnc in F:
            feature, marker = fnc(sentence)
            f = fnc2feature_name(fnc, ["feature_politeness", "politeness_markers"])
            features[f[0]] = max(features[f[0]], feature)

            # adds sent_ind to marker information
            if len(marker) > 0:
                for occ in marker:
                    markers[f[1]] += [[(mark[0], sent_ind, mark[1]) for mark in occ]]

    return features, markers
