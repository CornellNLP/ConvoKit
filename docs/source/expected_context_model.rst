Expected Context Framework
==========================

Implements the Expected Context Framework as described in `this dissertation <https://tisjune.github.io/research/dissertation>`_.

Contains:

* Basic `ExpectedContextModelTransformer <https://convokit.cornell.edu/documentation/expected_context_model.html#convokit.expected_context_framework.expected_context_model.ExpectedContextModelTransformer>`_
* Wrapper `DualContextWrapper <https://convokit.cornell.edu/documentation/expected_context_model.html#convokit.expected_context_framework.dual_context_wrapper.DualContextWrapper>`_ that handles two choices of conversational context
* Wrapper pipelines `ExpectedContextModelPipeline <https://convokit.cornell.edu/documentation/expected_context_model.html#convokit.expected_context_framework.expected_context_model_pipeline.ExpectedContextModelPipeline>`_ and `DualContextPipeline <https://convokit.cornell.edu/documentation/expected_context_model.html#convokit.expected_context_framework.expected_context_model_pipeline.DualContextPipeline>`_


Example usage:

* `deriving question types and other characterizations in British parliamentary question periods <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/parliament_demo.ipynb>`_
* exploration of Switchboard dialog acts corpus `using ExpectedContextModelTransformer <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/switchboard_exploration_demo.ipynb>`_, and `using DualContextWrapper <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/switchboard_exploration_dual_demo.ipynb>`_
* `examining Wikipedia talk page discussions <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/wiki_awry_demo.ipynb>`_
* `computing the orientation of justice utterances in the US Supreme Court <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/scotus_orientation_demo.ipynb>`_

.. automodule:: convokit.expected_context_framework.expected_context_model
	:members:
	:member-order: bysource

.. automodule:: convokit.expected_context_framework.dual_context_wrapper
	:members:
	:member-order: bysource

.. automodule:: convokit.expected_context_framework.expected_context_model_pipeline
	:members:
	:member-order: bysource
