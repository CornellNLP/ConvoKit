Summary of Conversation Dynamics (SCD)
========================================

SCD (Summary of Conversation Dynamics) is a ConvoKit Transformer that generates summaries of conversational dynamics from conversation transcripts, as introduced in the paper `"How did we get here? Summarizing conversation dynamics" <https://arxiv.org/pdf/2404.19007>`_. 

SCD extracts structured representations of conversation dynamics in two forms:

* **Summary of Conversation Dynamics (SCD)**: A summary describing the overall dynamics in a conversation
* **Sequence of Patterns (SoP)**: A structured sequence of interaction patterns extracted from the SCD, introduced in the paper `"A Similarity Measure for Comparing Conversational Dynamics" <https://arxiv.org/abs/2507.18956>`_

Note that SCD computation requires access to a LLM. We provide a unified interface for working with LLMs in the `GenAI module <genai.html>`_. It is recommended to setup for GenAI models in the module beforehand to compute SCD.

To see the use of SCD Transformer in action, check out:

* `Simple example notebook <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/convo_similarity/examples/example.ipynb>`_ showcasing basic SCD usage.

Module Reference
----------------

.. automodule:: convokit.convo_similarity.scd
    :members:

