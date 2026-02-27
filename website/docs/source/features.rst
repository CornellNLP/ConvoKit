Features & APIs
===============

ConvoKit provides a comprehensive set of analysissss tools for extracting conversational features and studying social phenomena. All features follow a scikit-learn-inspired interface.

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

Linguistic Coordination
-----------------------

.. raw:: html

   <div class="feature-card" data-tags="linguistic, power, influence, measurement, statistical">

A measure of linguistic influence and relative power between individuals or groups based on function word usage.

* **Use cases:** Power dynamics analysis, influence measurement
* **API:** `coordination <https://convokit.cornell.edu/documentation/coordination.html>`_
* **Research:** `Echoes of Power <https://www.cs.cornell.edu/~cristian/Echoes_of_power.html>`_
* **Tags:** linguistic, power, influence, measurement, statistical
* **Tags:** linguistic, power, influence

**Example:** `Power balance in U.S. Supreme Court <https://github.com/CornellNLP/ConvoKit/blob/master/examples/coordination/examples.ipynb>`_

.. code-block:: python

   from convokit import Coordination

   coord = Coordination()
   corpus = coord.transform(corpus)

.. raw:: html

   </div>

Politeness Strategies
---------------------

.. raw:: html

   <div class="feature-card" data-tags="linguistic, social, politeness, classification, statistical">

Lexical and parse-based features that correlate with politeness and impoliteness in conversations.

* **Use cases:** Politeness analysis, request analysis, conflict detection
* **API:** `politenessStrategies <https://convokit.cornell.edu/documentation/politenessStrategies.html>`_
* **Research:** `A Computational Approach to Politeness <https://www.cs.cornell.edu/~cristian/Politeness.html>`_
* **Tags:** linguistic, social, politeness, classification, statistical

**Example:** `Politeness in Wikipedia conflicts <https://github.com/CornellNLP/ConvoKit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb>`_

.. code-block:: python

   from convokit import PolitenessStrategies

   ps = PolitenessStrategies()
   corpus = ps.transform(corpus)

.. raw:: html

   </div>

Expected Conversational Context Framework
------------------------------------------

.. raw:: html

   <div class="feature-card" data-tags="context, linguistic, modeling, classification, utterance, pragmatics">

A framework for characterizing utterances and terms based on their expected conversational context.

* **Use cases:** Question type derivation, dialog act classification, utterance characterization
* **API:** `expected_context_model <https://convokit.cornell.edu/documentation/expected_context_model.html>`_
* **Research:** `Expected Context Framework <https://tisjune.github.io/research/dissertation>`_
* **Tags:** context, linguistic, modeling, classification, utterance, pragmatics

**Examples:**

* `Question types in British Parliament <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/parliament_demo.ipynb>`_
* `Switchboard dialog acts <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/switchboard_exploration_demo.ipynb>`_
* `Wikipedia discussions <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/wiki_awry_demo.ipynb>`_
* `Supreme Court orientations <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/scotus_orientation_demo.ipynb>`_

.. raw:: html

   </div>

Hypergraph Conversation Representation
---------------------------------------

.. raw:: html

   <div class="feature-card" data-tags="structural, graph, pattern, conversation, representation">

Extract structural features of conversations through hypergraph representation.

* **Use cases:** Conversation structure analysis, participation patterns, thread dynamics
* **API:** `hyperconvo <https://convokit.cornell.edu/documentation/hyperconvo.html>`_
* **Research:** `Patterns of Participant Interactions <http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html>`_
* **Tags:** structural, graph, pattern, conversation, representation

**Example:** `Reddit hypergraph analysis <https://github.com/CornellNLP/ConvoKit/blob/master/examples/hyperconvo/hyperconvo_demo.ipynb>`_

.. code-block:: python

   from convokit import HyperConvo

   hc = HyperConvo()
   corpus = hc.transform(corpus)

.. raw:: html

   </div>

Linguistic Diversity
--------------------

.. raw:: html

   <div class="feature-card" data-tags="linguistic, diversity, development, speaker, measurement, statistical">

Compute linguistic diversity of individuals within and across conversations.

* **Use cases:** Voice development, linguistic adaptation, speaker characterization
* **API:** `speakerConvoDiversity <https://convokit.cornell.edu/documentation/speakerConvoDiversity.html>`_
* **Research:** `Finding Your Voice <http://www.cs.cornell.edu/~cristian/Finding_your_voice__linguistic_development.html>`_
* **Tags:** linguistic, diversity, development, speaker, measurement, statistical

**Example:** `Diversity on ChangeMyView <https://github.com/CornellNLP/ConvoKit/blob/master/examples/speaker-convo-attributes/speaker-convo-diversity-demo.ipynb>`_

.. raw:: html

   </div>

CRAFT: Conversational Forecasting
----------------------------------

.. raw:: html

   <div class="feature-card" data-tags="prediction, neural, machine learning, forecasting, conversation, detection">

Neural model for forecasting future conversation outcomes (e.g., derailment into personal attacks) as they develop.

* **Use cases:** Real-time moderation, derailment prediction, outcome forecasting
* **API:** `forecaster <https://convokit.cornell.edu/documentation/forecaster.html>`_
* **Research:** `Trouble on the Horizon <https://arxiv.org/abs/1909.01362>`_
* **Tags:** prediction, neural, machine learning, forecasting, conversation, detection

**Interactive notebooks:**

* `Full version (fine-tuning + inference) <https://colab.research.google.com/drive/1SH4iMEHdoH4IovN-b9QOSK4kG4DhAwmb>`_
* `Inference-only <https://colab.research.google.com/drive/1GvICZN0VwZQSWw3pJaEVY-EQGoO-L5lH>`_

.. raw:: html

   </div>

Redirection and Utterance Likelihood
-------------------------------------

.. raw:: html

   <div class="feature-card" data-tags="structural, modeling, conversation-flow, utterance, detection">

Measure the extent to which utterances redirect conversational flow and compute utterance log-likelihoods given context.

* **Use cases:** Topic shift detection, conversational control, response predictability
* **API:** `redirectionAndUtteranceLikelihood <https://convokit.cornell.edu/documentation/redirectionAndUtteranceLikelihood.html>`_
* **Research:** `Conversational Redirection <https://www.cs.cornell.edu/~cristian/Redirection_in_Therapy.html>`_
* **Tags:** structural, modeling, conversation-flow, utterance, detection

**Example:** `Redirection in Supreme Court <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/redirection/redirectionDemo.ipynb>`_

.. raw:: html

   </div>

Pivotal Moment Measure
-----------------------

.. raw:: html

   <div class="feature-card" data-tags="structural, prediction, turning-points, conversation, detection">

Identify pivotal moments in conversations where outcomes may change.

* **Use cases:** Critical moment detection, trajectory analysis, intervention timing
* **API:** `pivotal <https://convokit.cornell.edu/documentation/pivotal.html>`_
* **Tags:** structural, prediction, turning-points, conversation, detection

**Example:** `Pivotal moments in conflicts <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/pivotal_framework/pivotal_demo.ipynb>`_

.. raw:: html

   </div>

   </div>
