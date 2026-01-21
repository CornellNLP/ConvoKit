from convokit import download, Corpus
from convokit.text_processing import TextParser
import json
import os
import sys

"""
	some code to update datasets with new parse format.
"""


VERBOSITY = 1000
ROOT_DIR = "/kitchen/clean-corpora"
INCREMENT_VERSION = False
PARSE = True

if __name__ == "__main__":
    print("downloading corpus")
    corpus_name = sys.argv[1]
    filename = download(corpus_name, data_dir=ROOT_DIR)

    print("purging parses")
    with open(os.path.join(filename, "index.json")) as f:
        index = json.load(f)
    try:
        del index["utterances-index"]["parsed"]
    except:
        pass
    if INCREMENT_VERSION:
        index["version"] += 1
    with open(os.path.join(filename, "index.json"), "w") as f:
        json.dump(index, f)

    if os.path.exists(os.path.join(filename, "utterances.json")):
        with open(os.path.join(filename, "utterances.json")) as f:
            utterances = json.load(f)
    else:
        utterances = []
        with open(os.path.join(filename, "utterances.jsonl")) as f:
            for line in f:
                utterances.append(json.loads(line))
    for utt in utterances:
        try:
            del utt["meta"]["parsed"]
        except:
            continue

    with open(os.path.join(filename, "utterances.jsonl"), "w") as f:
        for utt in utterances:
            json.dump(utt, f)
            f.write("\n")

    if os.path.exists(os.path.join(filename, "utterances.json")):
        os.remove(os.path.join(filename, "utterances.json"))
    if os.path.exists(os.path.join(filename, "parsed-bin.p")):
        os.remove(os.path.join(filename, "parsed-bin.p"))

    if PARSE:
        print("loading corpus")
        corpus = Corpus(filename)
        print("parsing corpus")
        textparser = TextParser(verbosity=VERBOSITY)
        corpus = textparser.transform(corpus)

        print("dumping parses")
        corpus.dump_info("utterance", ["parsed"])
    os.remove(os.path.join(ROOT_DIR, corpus_name + ".zip"))
