Pivotal Moment Measure
====================================

The `PivotalMomentMeasure` transformer identifies pivotal moments in conversations 
as described in this `paper <http://www.cs.cornell.edu/~cristian/>`_. 

We consider a moment in a conversation *pivotal* if the next response is expected 
to have a large impact on the conversation’s eventual outcome. Our method relies on 
two main components: an `utteranceSimulatorModel` for generating possible responses 
and a `forecasterModel` for forecasting the eventual outcome of the conversation.

`PivotalMomentMeasure` uses a temporally-ordered stream of conversational data in 
the form of “context tuples” to train and make predictions on. Context tuples 
are generated in chronological order for each utterance in a conversation. 
Each context tuple is defined as a NamedTuple with the following fields: 

* ``context``: a chronological list of Utterances up to and including the most recent Utterance at the time this context was generated
* ``current_utterance``: the most recent utterance at the time this context tuple was generated
* ``future_context``: all Utterances chronologically after the current utterance (or an empty list if this Utterance is the last one)
* ``conversation_id``: the Conversation that this context-reply pair came from

We also provide a general `utteranceSimulator` interface to `utteranceSimulatorModel`
models that abstracts away the implementation details into a standard fit-transform 
interface.

Example usage: `pivotal moments demo in conversations gone awry <https://github.com/CornellNLP/ConvoKit/tree/master/convokit/pivotal_framework/pivotal_demo.ipynb>`_

.. automodule:: convokit.pivotal_framework.pivotal
    :members:

.. automodule:: convokit.utterance_simulator.utteranceSimulatorModel
    :members:

.. automodule:: convokit.utterance_simulator.utteranceSimulator
    :members: