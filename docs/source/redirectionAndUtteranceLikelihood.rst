Redirection and Utterance Likelihood
====================================

The `Redirection` transformer measures the extent to which utterances 
redirect the flow of the conversation, 
as described in this
`paper <https://www.cs.cornell.edu/~cristian/Redirection_in_Therapy.html>`_.
The redirection effect of an utterance is determined by comparing the likelihood 
of its reply given the immediate conversation context vs. a reference context 
representing the previous direction of the conversation.

The `UtteranceLikelihood` transformer is a more generalized module that just 
implements log-likelihoods of utterances given a defined conversation context. 

Example usage: `redirection in supreme court oral arguments <https://github.com/CornellNLP/ConvoKit/tree/master/convokit/redirection/redirectionDemo.ipynb>`_

.. automodule:: convokit.redirection.redirection
    :members:

.. automodule:: convokit.redirection.likelihoodModel
    :members:

.. automodule:: convokit.utterance_likelihood.utteranceLikelihood
    :members: