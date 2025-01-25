# This example extracts politeness strategies from the Conversations Gone Awry dataset,
#   one of the steps in the Conversations Gone Awry paper (http://www.cs.cornell.edu/~cristian/Conversations_gone_awry.html).
#   For code reproducing the full results of the paper, see the example notebook in the
#   `conversations-gone-awry` example subdirectory.

import pandas as pd
# from .model import *
#     from .util import *
#     from .coordination import *
#     from .transformer import *
#     from .convokitPipeline import *
#     from .hyperconvo import *
#     from .speakerConvoDiversity import *
#     from .phrasing_motifs import *
#     from .prompt_types import *
#     from .classifier import *
#     from .ranker import *
#     from .forecaster import *
#     from .fighting_words import *
#     from .paired_prediction import *
#     from .bag_of_words import *
#     from .expected_context_framework import *
#     from .surprise import *
#     from .convokitConfig import *
import convokit
from convokit import Transformer, ConvokitPipeline, HyperConvo, SpeakerConvoDiversity, Classifier, Ranker, Forecaster, FightingWords, PairedPrediction, bag_of_words, surprise, convokitConfig
from convokit import phrasing_motifs
from convokit import text_processing
from convokit import politenessStrategies
# print("Loading awry corpus...")
# corpus = Corpus(filename=download("conversations-gone-awry-corpus"))

# # extract the politeness strategies.
# # Note: politeness strategies are a hand-engineered feature set, so no fitting is needed.
# ps = PolitenessStrategies(verbose=100)
# print("Extracting politeness strategies...")
# corpus = ps.transform(corpus)

# values = []
# idx = []
# for utterance in corpus.iter_utterances():
#     values.append(utterance.meta["politeness_strategies"])
#     idx.append(utterance.id)
# pd.DataFrame(values, index=idx).to_csv("awry_strategy_df_v2.csv")
# print("Done, results written to awry_strategy_df_v2.csv")
