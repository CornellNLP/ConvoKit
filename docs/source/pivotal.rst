Pivotal Measure
====================================

The `PIV` transformer identifes pivotal moments in conversations as described in 
this paper: Hanging in the Balance: Pivotal Moments in Crisis 
Counseling Conversations. 

We consider a moment in a conversation *pivotal* if the next response is expected 
to have a large impact on the conversation’s eventual outcome. Our method relies on 
two main components: an `utteranceSimulatorModel` for generating possible responses 
and a `forecasterModel` for forecasting the eventual outcome of the conversation.

We also provide a general `utteranceSimulator` interface to `utteranceSimulatorModel` 
models that abstracts away the implementation details into a standard fit-transform 
interface.

Example usage: `pivotal moments in conversations gone awry <https://github.com/CornellNLP/ConvoKit/tree/master/convokit/pivotal_framework/pivotal_demo.ipynb>`_

.. automodule:: convokit.pivotal_framework.pivotal
    :members:

.. automodule:: convokit.pivotal_framework.simulator.utteranceSimulatorModel
    :members:

.. automodule:: convokit.pivotal_framework.simulator.utteranceSimulator
    :members: