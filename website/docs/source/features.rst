Features & APIs
===============

ConvoKit provides a comprehensive set of analysis tools for extracting conversational features and studying social phenomena.

.. raw:: html

   <div class="feature-search-container">
     <input type="text" id="feature-search" placeholder="Search features by name or tags..." />
     <div class="filter-label">Filter by category:</div>
     <div class="tag-filters">
       <!-- Tags will be dynamically populated by JavaScript -->
     </div>
     <button class="clear-filters">Clear Filters</button>
   </div>

   <div id="features-container">

TextParser
----------

.. raw:: html

   <div class="feature-card" data-tags="pre-processing, parsing, utterance, linguistic">

Dependency-parses each utterance in a corpus using SpaCy. This parsing step is a prerequisite for several other ConvoKit transformers.

* **API:** `TextParser <https://convokit.cornell.edu/documentation/textParser.html>`_
* **Tags:** pre-processing, parsing, utterance, linguistic

**Example:** `Text Preprocessing <https://github.com/CornellNLP/ConvoKit/blob/master/examples/text-processing/text_preprocessing_demo.ipynb>`_

.. raw:: html

   </div>

TextToArcs
----------

.. raw:: html

   <div class="feature-card" data-tags="pre-processing, structural, parsing, utterance, linguistic, pattern">

Converts dependency parse output into arc-based representations, where each sentence is expressed as a collection of dependency arc strings. Requires TextParser to be run first.

* **API:** `TextToArcs <https://convokit.cornell.edu/documentation/textToArcs.html>`_
* **Tags:** pre-processing, structural, parsing, utterance, linguistic, pattern

.. raw:: html

   </div>

TextCleaner
-----------

.. raw:: html

   <div class="feature-card" data-tags="pre-processing, utterance, linguistic">

Cleans utterance text by fixing unicode errors, lowercasing, removing line breaks, and replacing URLs, emails, phone numbers, and currency symbols with special tokens. Supports custom cleaning functions.

* **API:** `TextCleaner <https://convokit.cornell.edu/documentation/textCleaner.html>`_
* **Tags:** pre-processing, utterance, linguistic

.. raw:: html

   </div>

TextProcessor (base class)
--------------------------

.. raw:: html

   <div class="feature-card" data-tags="pre-processing, utterance, linguistic">

Abstract base class for text processing transformers in ConvoKit. Provides a shared interface for transformers that read and write utterance text fields, enabling pipeline composition of text processing steps.

* **API:** `Transformer (base class) <https://convokit.cornell.edu/documentation/transformer.html>`_
* **Tags:** pre-processing, utterance, linguistic

**Example:** `Text Preprocessing <https://github.com/CornellNLP/ConvoKit/blob/master/examples/text-processing/text_preprocessing_demo.ipynb>`_

.. raw:: html

   </div>

Bag-of-Words
------------

.. raw:: html

   <div class="feature-card" data-tags="feature extraction, vectorization, speaker, utterance, conversation">

Vectorizes corpus objects (utterances, speakers, or conversations) using bag-of-words representations. Compatible with any sklearn-style vectorizer including TF-IDF. Stores representations as ConvoKitMatrix objects for downstream use with classifiers.

* **API:** `BoWTransformer <https://convokit.cornell.edu/documentation/bow.html>`_
* **Tags:** feature extraction, vectorization, speaker, utterance, conversation

**Example:** `Bag-of-Words classification <https://github.com/CornellNLP/ConvoKit/blob/master/examples/vectors/bag-of-words-demo.ipynb>`_

.. raw:: html

   </div>

Column Normalized Tf-Idf
------------------------

.. raw:: html

   <div class="feature-card" data-tags="feature extraction, utterance, representation">

A modified Tf-Idf transformer that normalizes by columns (term-wise) rather than rows, producing more balanced term representations across a corpus. Often used as input to the Expected Conversational Context Framework.

* **API:** `ColNormedTfidfTransformer <https://convokit.cornell.edu/documentation/col_normed_tfidf.html>`_
* **Tags:** feature extraction, utterance, representation

.. raw:: html

   </div>

Hypergraph Conversation Representation
---------------------------------------

.. raw:: html

   <div class="feature-card" data-tags="structural, graph, conversation, pattern">

Extracts structural features of conversations through a hypergraph model, computing degree distribution statistics and motif counts for both full conversations and mid-threads. Forms the basis for ThreadEmbedder and CommunityEmbedder.

* **API:** `HyperConvo <https://convokit.cornell.edu/documentation/hyperconvo.html>`_
* **Research:** `Patterns of Participant Interactions <http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html>`_
* **Tags:** structural, graph, conversation, pattern

**Example:** `Reddit hypergraph analysis <https://github.com/CornellNLP/ConvoKit/blob/master/examples/hyperconvo/hyperconvo_demo.ipynb>`_

.. raw:: html

   </div>

Phrasing Motifs
---------------

.. raw:: html

   <div class="feature-card" data-tags="feature extraction, utterance, linguistic, pragmatics">

Extracts arc-based phrasing patterns from dependency-parsed utterances by abstracting away content words, capturing common syntactic structures independently of topic. Used as input to the Prompt Types framework.

* **API:** `PhrasingMotifs <https://convokit.cornell.edu/documentation/phrasingMotifs.html>`_
* **Research:** `Asking Too Much? <https://www.cs.cornell.edu/~cristian/Asking_too_much.html>`_
* **Tags:** feature extraction, utterance, linguistic, pragmatics

**Example:** `Phrasing motifs in prompt type models <https://github.com/CornellNLP/ConvoKit/blob/master/examples/prompt-types/prompt-type-demo.ipynb>`_

.. raw:: html

   </div>

Politeness Strategies
---------------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, statistical, conversation, linguistic, social, politeness, pragmatics">

Detects lexical and parse-based politeness and impoliteness strategies in utterances based on the Brown and Levinson politeness framework, producing binary feature vectors over a set of validated linguistic markers.

* **API:** `PolitenessStrategies <https://convokit.cornell.edu/documentation/politenessStrategies.html>`_
* **Research:** `A Computational Approach to Politeness <https://www.cs.cornell.edu/~cristian/Politeness.html>`_
* **Tags:** measurement, statistical, conversation, linguistic, social, politeness, pragmatics

**Example:** `Extracting politeness features and markers <https://github.com/CornellNLP/ConvoKit/blob/master/examples/politeness-strategies/Politeness_Marker_and_Summarize_Demo.ipynb>`_

.. raw:: html

   </div>

Prompt Types
------------

.. raw:: html

   <div class="feature-card" data-tags="feature extraction, vectorization, utterance, context, pragmatics">

Infers latent types of conversational prompts based on how they are phrased, using SVD-based embeddings of phrasing motifs relative to their response contexts. Assigns each utterance a prompt type and vector representation.

* **API:** `PromptTypes <https://convokit.cornell.edu/documentation/promptTypes.html>`_
* **Research:** `Asking Too Much? <https://www.cs.cornell.edu/~cristian/Asking_too_much.html>`_
* **Tags:** feature extraction, vectorization, utterance, context, pragmatics

**Examples:**

* `Prompt Type Wrapper <https://github.com/CornellNLP/ConvoKit/blob/master/examples/prompt-types/prompt-type-wrapper-demo.ipynb>`_
* `Phrasing and rhetorical intent <https://github.com/CornellNLP/ConvoKit/blob/master/examples/prompt-types/prompt-type-demo.ipynb>`_

.. raw:: html

   </div>

Expected Conversational Context Framework
------------------------------------------

.. raw:: html

   <div class="feature-card" data-tags="structural, modeling, utterance, exchange, linguistic, context">

Derives representations of utterances and terms based on their expected conversational context — the replies they tend to elicit or the utterances they tend to appear near. Supports both forward and backward context modeling.

* **API:** `ExpectedContextModel <https://convokit.cornell.edu/documentation/expected_context_model.html>`_
* **Research:** `Expected Context Framework <https://tisjune.github.io/research/dissertation>`_
* **Tags:** structural, modeling, utterance, exchange, linguistic, context

**Examples:**

* `Question types in British Parliament <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/parliament_demo.ipynb>`_
* `Switchboard Corpus <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/switchboard_exploration_demo.ipynb>`_
* `Wikipedia discussions <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/wiki_awry_demo.ipynb>`_

.. raw:: html

   </div>

Redirection and Utterance Likelihood
-------------------------------------

.. raw:: html

   <div class="feature-card" data-tags="structural, modeling, utterance, exchange, conversation-flow, detection, LLM, simulation">

Measures the extent to which an utterance redirects conversational flow away from its context, and computes utterance log-likelihoods given surrounding context using a language model.

* **API:** `Redirection and UtteranceLikelihood <https://convokit.cornell.edu/documentation/redirectionAndUtteranceLikelihood.html>`_
* **Research:** `Conversational Redirection in Therapy <https://www.cs.cornell.edu/~cristian/Redirection_in_Therapy.html>`_
* **Tags:** structural, modeling, utterance, exchange, conversation-flow, detection, LLM, simulation

**Example:** `Redirection in Supreme Court <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/redirection/redirectionDemo.ipynb>`_

.. raw:: html

   </div>

Pivotal Moment Measure
-----------------------

.. raw:: html

   <div class="feature-card" data-tags="prediction, modeling, conversation, turning-points, simulation">

Identifies pivotal moments in a conversation where simulated alternative responses would most change the predicted outcome. Combines a simulator model and a forecaster model to score each conversational position.

* **API:** `PivotalMomentMeasure <https://convokit.cornell.edu/documentation/pivotal.html>`_
* **Tags:** prediction, modeling, conversation, turning-points, simulation

**Example:** `Pivotal moments in conversations gone awry <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/pivotal_framework/pivotal_demo.ipynb>`_

.. raw:: html

   </div>

LLM Prompt Transformer
-----------------------

.. raw:: html

   <div class="feature-card" data-tags="feature extraction, utterance, conversation, speaker, corpus, pragmatics">

Applies custom LLM prompts to corpus objects at any level — utterances, conversations, speakers, or the entire corpus — and stores responses as metadata. Supports multiple LLM providers including OpenAI GPT, Google Gemini, and local models.

* **API:** `LLMPromptTransformer <https://convokit.cornell.edu/documentation/llmprompttransformer.html>`_
* **Tags:** feature extraction, utterance, conversation, speaker, corpus, pragmatics

**Example:** `GenAI module <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/genai/example/example.ipynb>`_

.. raw:: html

   </div>

Classifier
----------

.. raw:: html

   <div class="feature-card" data-tags="classification, modeling, utterance, conversation, speaker, labeling">

Trains and applies a classifier on corpus object metadata features. Uses an sklearn-compatible classifier and exposes evaluation utilities including cross-validation and train-test split scoring.

* **API:** `Classifier <https://convokit.cornell.edu/documentation/classifier.html>`_
* **Tags:** classification, modeling, utterance, conversation, speaker, labeling

**Example:** `Politeness <https://github.com/CornellNLP/ConvoKit/blob/master/examples/politeness-strategies/politeness_demo.ipynb>`_

.. raw:: html

   </div>

Vector Classifier
-----------------

.. raw:: html

   <div class="feature-card" data-tags="classification, modeling, vectorization, utterance, conversation, speaker, labeling">

Trains and applies a classifier on corpus object vector representations (e.g. bag-of-words, TF-IDF). Inherits from Classifier. Requires a ConvoKitMatrix with the specified vector name to be present on the corpus.

* **API:** `VectorClassifier <https://convokit.cornell.edu/documentation/vectorClassifier.html>`_
* **Tags:** classification, modeling, vectorization, utterance, conversation, speaker, labeling

**Example:** `Bag-of-Words classification <https://github.com/CornellNLP/ConvoKit/blob/master/examples/vectors/bag-of-words-demo.ipynb>`_

.. raw:: html

   </div>

Linguistic Coordination
-----------------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, statistical, conversation, linguistic, power, influence">

Measures the propensity of a speaker to echo the function words used by another speaker in a conversation, serving as a proxy for linguistic coordination and relative power dynamics between individuals or groups.

* **API:** `Coordination <https://convokit.cornell.edu/documentation/coordination.html>`_
* **Research:** `Echoes of Power <http://www.cs.cornell.edu/~cristian/Echoes_of_power.html>`_
* **Tags:** measurement, statistical, conversation, linguistic, power, influence

**Example:** `Power balance in U.S. Supreme Court <https://github.com/CornellNLP/ConvoKit/blob/master/examples/coordination/examples.ipynb>`_

.. raw:: html

   </div>

Fighting Words
--------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, statistical, utterance, conversation, speaker, power, influence, social, pattern, comparison">

Identifies the n-gram features that most distinguish two groups of corpus objects, using Monroe et al.'s Dirichlet-multinomial method. Annotates objects with the top fighting words of each class.

* **API:** `FightingWords <https://convokit.cornell.edu/documentation/fightingwords.html>`_
* **Tags:** measurement, statistical, utterance, conversation, speaker, power, influence, social, pattern, comparison

**Examples:** `r/atheism vs r/Christianity <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/fighting_words/demos/fightingwords_demo.ipynb>`_

.. raw:: html

   </div>

Forecaster
----------

.. raw:: html

   <div class="feature-card" data-tags="prediction, machine learning, neural, utterance, forecasting, LLM">

A framework for forecasting future conversation outcomes as they develop in real time. Wraps any ForecasterModel (e.g. CRAFT, BERT-based models) and feeds a chronological stream of context tuples to enable per-utterance prediction.

* **API:** `Forecaster <https://convokit.cornell.edu/documentation/forecaster.html>`_
* **Research:** `Trouble on the Horizon <https://arxiv.org/abs/1909.01362>`_
* **Tags:** prediction, machine learning, neural, utterance, forecasting, LLM

**Example:** `CRAFT on CGA <https://github.com/CornellNLP/ConvoKit/blob/master/examples/forecaster/CRAFT%20Forecaster%20demo.ipynb>`_

.. raw:: html

   </div>

Thread Embedder
---------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, feature extraction, statistical, corpus, pattern">

Embeds thread-level hypergraph statistics from HyperConvo into a low-dimensional space using SVD or other dimensionality reduction. Useful for visualizing and comparing thread structure across a corpus.

* **API:** `ThreadEmbedder <https://convokit.cornell.edu/documentation/threadEmbedder.html>`_
* **Research:** `Patterns of Participant Interactions <http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html>`_
* **Tags:** measurement, feature extraction, statistical, corpus, pattern

.. raw:: html

   </div>

Community Embedder
------------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, feature extraction, statistical, corpus, pattern">

Embeds community-level hypergraph statistics from HyperConvo into a low-dimensional space, enabling comparison and visualization of structural differences across communities or subreddits.

* **API:** `CommunityEmbedder <https://convokit.cornell.edu/documentation/communityEmbedder.html>`_
* **Research:** `Patterns of Participant Interactions <http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html>`_
* **Tags:** measurement, feature extraction, statistical, corpus, pattern

.. raw:: html

   </div>

Pairer
------

.. raw:: html

   <div class="feature-card" data-tags="pre-processing, prediction, statistical, utterance, conversation, speaker, representation, pattern">

Annotates corpus objects with pairing information needed for paired prediction analyses. Controls for conversational context by pairing objects from the same conversation, enabling comparisons that isolate the variable of interest.

* **API:** `Pairer <https://convokit.cornell.edu/documentation/pairer.html>`_
* **Tags:** pre-processing, prediction, statistical, utterance, conversation, speaker, representation, pattern

.. raw:: html

   </div>

Paired Prediction
-----------------

.. raw:: html

   <div class="feature-card" data-tags="prediction, classification, machine learning, corpus, detection">

A quasi-experimental prediction method that controls for confounding priors by comparing matched pairs of corpus objects from the same conversation, enabling more rigorous causal inference.

* **API:** `PairedPrediction <https://convokit.cornell.edu/documentation/pairedprediction.html>`_
* **Research:** `Antisocial Behavior in Online Discussion Communities <https://arxiv.org/abs/1504.00680>`_
* **Tags:** prediction, classification, machine learning, corpus, detection

**Example:** `Predicting conversation growth on Reddit <https://github.com/CornellNLP/ConvoKit/blob/master/examples/hyperconvo/predictive_tasks.ipynb>`_

.. raw:: html

   </div>

Ranker
------

.. raw:: html

   <div class="feature-card" data-tags="sorting, statistical, utterance, conversation, speaker, representation">

Sorts and annotates corpus objects with rankings based on a user-defined scoring function. Supports ranking of utterances, speakers, or conversations by any derived or metadata feature.

* **API:** `Ranker <https://convokit.cornell.edu/documentation/ranker.html>`_
* **Tags:** sorting, statistical, utterance, conversation, speaker, representation

**Example:** `Ranking users in r/Cornell by comment count <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/ranker/demos/ranker_demo.ipynb>`_

.. raw:: html

   </div>

Linguistic Diversity
--------------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, statistical, speaker, corpus, diversity, development">

Computes the linguistic divergence between a speaker's language in each conversation and a reference language model trained on other conversations or speakers, measuring how a speaker's voice develops over time.

* **API:** `SpeakerConvoDiversity <https://convokit.cornell.edu/documentation/speakerConvoDiversity.html>`_
* **Research:** `Finding Your Voice <http://www.cs.cornell.edu/~cristian/Finding_your_voice__linguistic_development.html>`_
* **Tags:** measurement, statistical, speaker, corpus, diversity, development

**Example:** `Linguistic diversity on ChangeMyView <https://github.com/CornellNLP/ConvoKit/blob/master/examples/speaker-convo-attributes/speaker-convo-diversity-demo.ipynb>`_

.. raw:: html

   </div>

Summary of Conversation Dynamics (SCD)
---------------------------------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, feature extraction, LLM, corpus, pattern, conversation-flow, context">

Generates structured natural-language summaries of conversational dynamics using the LLM Prompt Transformer. Summaries describe how interactions unfold over time, capturing turn-by-turn shifts in tone, topic, and social dynamics.

* **API:** `SCD <https://convokit.cornell.edu/documentation/scd.html>`_
* **Research:** `How Did We Get Here? Summarizing Conversation Dynamics <https://aclanthology.org/2024.naacl-long.414/>`_
* **Tags:** measurement, feature extraction, LLM, corpus, pattern, conversation-flow, context

**Example:** `SCD on conversations gone awry <https://github.com/CornellNLP/ConvoKit/blob/master/examples/conversations-gone-awry-cmv/scd-example.ipynb>`_

.. raw:: html

   </div>

Conversation Dynamics Similarity (ConDynS)
-------------------------------------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, feature extraction, LLM, corpus, pattern, conversation-flow, context, comparison">

Measures the similarity between two conversations with respect to their conversational dynamics using SCD-based representations and sequence alignment. Enables topic-independent comparison of how conversations unfold.

* **API:** `ConvoDynamicsSimilarity <https://convokit.cornell.edu/documentation/condyns.html>`_
* **Research:** `A Similarity Measure for Comparing Conversational Dynamics <https://arxiv.org/abs/2507.18956>`_
* **Tags:** measurement, feature extraction, LLM, corpus, pattern, conversation-flow, context, comparison

.. raw:: html

   </div>

Talk-Time Sharing Dynamics
---------------------------

.. raw:: html

   <div class="feature-card" data-tags="measurement, feature extraction, statistical, corpus, pattern, conversation-flow, social, comparison">

Analyzes how talk-time is distributed and evolves between speakers throughout a conversation, capturing both overall balance and moment-to-moment dynamics in participation patterns.

* **API:** `TalkTimeSharingDynamics <https://convokit.cornell.edu/documentation/talktimesharing.html>`_
* **Tags:** measurement, feature extraction, statistical, corpus, pattern, conversation-flow, social, comparison

**Example:** `Talk-time in CANDOR and Supreme Court <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/talktimesharing/talktimesharing_example.ipynb>`_

.. raw:: html

   </div>

   </div>
