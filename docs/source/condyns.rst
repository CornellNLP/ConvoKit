Conversation Dynamics Similarity (ConDynS)
==========================================

ConDynS is a similarity measure for comparing conversations with respect to their dynamics, as introduced in the paper `"A Similarity Measure for Comparing Conversational Dynamics" <https://arxiv.org/abs/2507.18956>`_. The quality of a conversation goes beyond the individual quality of each reply, and instead emerges from how these combine into interactional patterns that give the conversation its distinctive overall "shape". ConDynS provides a robust automated method for comparing conversations in terms of their overall interactional dynamics.

In this module, we provide a comprehensive framework for computing ConDynS, including:

* **ConDynS**: Main similarity computation using bidirectional comparison between SCD patterns and conversation transcripts
* **NaiveConDynS**: Simplified similarity computation using only SoP comparison without transcripts
* **ConDynSBaselines**: Baseline methods for comparison including BERTScore, cosine similarity, and LLM-based direct comparison

ConDynS builds on top of the `SCD (Summary of Conversation Dynamics) <scd.html>`_ module, which generates structured summaries of conversation dynamics. To compute ConDynS, you first need to extract SCD summaries from your conversations using the SCD transformer.

Note that ConDynS computation requires access to a LLM. We provide a unified interface for working with LLMs in the `GenAI module <genai.html>`_. It is recommended to setup for GenAI models in the module beforehand to compute ConDynS.

We provide experiments notebooks from the paper:

* `Validation experiments <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/validation/validation.ipynb>`_
* `Baseline comparisons <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/validation/baselines.ipynb>`_
* `Applications to online communities <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/applications/applications.ipynb>`_
* Applications on `WikiConv German <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/applications/wiki_german/wiki_german_condyns.ipynb>`_ and `Friends <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/applications/friends/friends_condyns.ipynb>`_

To see a simple example of using both SCD and ConDynS together, check out `this example notebook <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/example.ipynb>`_.

Modules
-------

ConDynS
^^^^^^^^

.. automodule:: convokit.convo_similarity.condyns
    :members:

NaiveConDynS
^^^^^^^^^^^^

.. automodule:: convokit.convo_similarity.naive_condyns
    :members:

Baseline Methods
^^^^^^^^^^^^^^^^

.. automodule:: convokit.convo_similarity.baseline
    :members: