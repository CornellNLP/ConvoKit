Paired Prediction
=================

At a high level, Paired Prediction is a quasi-experimental method that controls for certain priors, see `Cheng et al.
2014 for an illustrated example of PairedPrediction in research <https://cs.stanford
.edu/people/jure/pubs/disqus-icwsm14.pdf>`_.

As an illustrative example, consider the Friends TV series, where we might want to examine how Rachel talks to
Monica and Chandler differently. At one level, we might just look at the differences in the utterances where
Rachel speaks to Monica and Rachel speaks to Chandler. But this inadvertently surfaces differences that might arise
from Rachel interacting with Monica and Chandler separately in different settings and scenarios, and thus highlight
only uninteresting differences in topics discussed.

Instead, we might want to look for subtler differences in speech, controlling for topic perhaps. One way we might to
do this to look only at Conversations where Rachel, Monica, and Chandler are all present. We would then compare
utterances where Rachel speaks to Monica and Rachel speaks to Chandler *within* that Conversation and look
for differences between these paired sets of utterances.

Documentation for the two transformers that do paired prediction task is presented below. PairedPrediction transformer uses corpus object’s metadata features for predictions, while PairedVectorPrediction transformer utilizes vector data associated with the object. Also, see the documentation for :doc:`Pairer transformer <pairer>`, which sets up the pairs needed in paired prediction analysis.

Example usage: `Using Hyperconvo features to predict conversation growth on Reddit in a paired setting <https://github.com/CornellNLP/ConvoKit/blob/master/examples/hyperconvo/predictive_tasks.ipynb>`_

.. automodule:: convokit.paired_prediction.pairedPrediction
    :members:

.. automodule:: convokit.paired_prediction.pairedVectorPrediction
    :members:
