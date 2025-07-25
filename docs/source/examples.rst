Examples
========

An index of useful examples to help you interactively explore ConvoKit's features.

Be sure to take a look at the `introductory tutorial <https://convokit.cornell.edu/documentation/tutorial.html>`_ before exploring these examples!

General ConvoKit usage (starting resource)
------------------------------------------
- `Introductory tutorial to ConvoKit <https://github.com/CornellNLP/ConvoKit/blob/master/examples/Introduction_to_ConvoKit.ipynb>`_
- Creating a ConvoKit Corpus from existing data:
   - `Converting custom dataset into ConvoKit format corpus <https://github.com/CornellNLP/ConvoKit/blob/master/examples/converting_movie_corpus.ipynb>`_
   - `Constructing ConvoKit format corpus from pandas dataframe <https://github.com/CornellNLP/ConvoKit/blob/master/examples/corpus_from_pandas.ipynb>`_
- `Using vector data in ConvoKit <https://github.com/CornellNLP/ConvoKit/blob/master/examples/vectors/vector_demo.ipynb>`_
- `Pre-processing text, e.g. computing dependency parses <https://github.com/CornellNLP/ConvoKit/blob/master/examples/text-processing/text_preprocessing_demo.ipynb>`_

Intermediate corpus functionality
---------------------------------
- `Merging two different Corpora (even when there are overlaps or conflicts in Corpus data) <https://github.com/CornellNLP/ConvoKit/blob/master/examples/merging/corpus_merge_demo.ipynb>`_
- `Partially loading utterances from an included dataset <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/tests/notebook_testers/test_corpus_partial_load.ipynb>`_

Classifier
------------
- `Extracting bag-of-Words vectors from utterances and using them in various classification tasks <https://github.com/CornellNLP/ConvoKit/blob/master/examples/vectors/bag-of-words-demo.ipynb>`_
- `Using common politeness strategies for various predictive tasks <https://github.com/CornellNLP/ConvoKit/blob/master/examples/politeness-strategies/politeness_demo.ipynb>`_


Coordination
------------
- `Exploring the balance of power in Wikipedia and the US Supreme Court <https://github.com/CornellNLP/ConvoKit/blob/master/examples/coordination/examples.ipynb>`_


Expected Conversational Context Framework
-----------------------------------------
- `deriving question types and other characterizations in British parliamentary question periods <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/parliament_demo.ipynb>`_
- exploration of Switchboard dialog acts corpus `using ExpectedContextModelTransformer <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/switchboard_exploration_demo.ipynb>`_, and `using DualContextWrapper <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/switchboard_exploration_dual_demo.ipynb>`_
- `examining Wikipedia talk page discussions <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/wiki_awry_demo.ipynb>`_
- `computing the orientation of justice utterances in the US Supreme Court <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/scotus_orientation_demo.ipynb>`_


Fighting Words
--------------
- `Examining the Fighting Words of mixed-gender vs. single-gender conversations in movies <https://github.com/CornellNLP/ConvoKit/blob/master/examples/sigdial-demo.ipynb>`_
- `Examining the Fighting Words of r/atheism vs r/Christianity <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/fighting_words/demos/fightingwords_demo.ipynb>`_

Forecaster
----------
- `CRAFT forecasting of conversational derailment <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/forecaster/CRAFT/demos/craft_demo_new.ipynb>`_
- `Forecasting of conversational derailment using a cumulative bag-of-words model <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/forecaster/tests/cumulativeBoW_demo.ipynb>`_
- `Evaluating Transformer Fine-tuned Models (i.e. RoBERTa, Gemma) <https://github.com/CornellNLP/ConvoKit/blob/master/examples/forecaster/Run%20Transformer%20Fine-tuned%20Models.ipynb>`_

Hyperconvo
----------
- `Categorizing and analyzing subreddits using Hyperconvo features <https://github.com/CornellNLP/ConvoKit/blob/master/examples/hyperconvo/demo.ipynb>`_
- `Using Hyperconvo features to predict conversation growth on Reddit in a paired setting <https://github.com/CornellNLP/ConvoKit/blob/master/examples/hyperconvo/predictive_tasks.ipynb>`_

.. Prompt Types
.. ------------
.. - `Exploring common types of questioning in the UK Parliament <https://github.com/CornellNLP/ConvoKit/blob/master/examples/prompt-types/prompt-type-demo.ipynb>`_
.. - `Using prompt types and politeness strategies to predict Wikipedia conversations going awry <https://github.com/CornellNLP/ConvoKit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb>`_

Politeness Strategies
---------------------
- `Exploring common politeness strategies used in Stanford Politeness Corpus <https://github.com/CornellNLP/ConvoKit/blob/master/examples/politeness-strategies/politeness_demo.ipynb>`_
- `Using politeness strategies to predict Wikipedia conversations gone awry <https://github.com/CornellNLP/ConvoKit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb>`_

Ranker
------
- `Ranking users in r/Cornell by the number of comments they have made <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/ranker/demos/ranker_demo.ipynb>`_

Speaker Convo Diversity
-----------------------
- `Speaker conversation attributes and diversity example on ChangeMyView <https://github.com/CornellNLP/ConvoKit/blob/master/examples/speaker-convo-attributes/speaker-convo-diversity-demo.ipynb>`_
