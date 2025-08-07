Conversation Dynamics Similarity
================================

ConDynS is a similarity measure for comparing conversations with respect to their dynamics, as introduced in the paper `"A Similarity Measure for Comparing Conversational Dynamics" <https://arxiv.org/abs/2507.18956>`_. The quality of a conversation goes beyond the individual quality of each reply, and instead emerges from how these combine into interactional patterns that give the conversation its distinctive overall "shape". ConDynS provides a robust automated method for comparing conversations in terms of their overall interactional dynamics.

In this module, we provide a comprehensive framework for computing ConDynS, including:

* **SCDWriter**: Generates Summary of Conversation Dynamics (SCD) from conversation transcripts and extracts Sequence of Patterns (SoP) from SCD
* **ConDynS**: Main similarity computation using bidirectional comparison between SCD patterns and conversation transcripts
* **NaiveConDynS**: Simplified similarity computation using only SoP comparison without transcripts
* **ConDynSBaselines**: Baseline methods for comparison including BERTScore, cosine similarity, and LLM-based direct comparison

Note that ConDynS computation requires access to a LLM. We provide a unified interface for working with LLMs in the `GenAI module <genai.html>`_. It is recommended to setup for GenAI models in the module beforehand to compute ConDynS.

We provide validation experiments from the paper: `validation experiments <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/validation/validation.ipynb>`_ and showcase the baseline methods in `baseline methods <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/validation/baselines.ipynb>`_.

We also provide example usages with different applications with ConDynS: `applications to online communities <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/applications/applications.ipynb>`_.

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

SCDWriter
^^^^^^^^^

.. automodule:: convokit.convo_similarity.summary
    :members:

Baseline Methods
^^^^^^^^^^^^^^^^

.. automodule:: convokit.convo_similarity.baseline
    :members: