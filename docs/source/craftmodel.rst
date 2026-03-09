CRAFT Model
============

A backend for Forecaster that implements the CRAFT algorithm from the EMNLP 2019 paper "Trouble on the Horizon: Forecasting the Derailment of Conversations as they Develop".

CRAFT is a neural model based on a pre-train-then-fine-tune paradigm. As the purpose of this class is to enable CRAFT to be used as a backend for Forecaster, it uses the author-provided already-trained CRAFT instance. Training a new CRAFT instance from scratch is considered outside the scope of ConvoKit. Users interested in creating their own custom CRAFT models can instead consult `the authors' official implementation <https://github.com/jpwchang/CRAFT>`_

IMPORTANT NOTE: This implementation directly uses the author-provided CRAFT model that was used in the paper's experiments. This model was developed separately from ConvoKit and uses its own tokenization scheme, which differs from ConvoKit's default. Using ConvoKit's tokenization could therefore result in tokens that are inconsistent with what the CRAFT model expects, leading to errors. ConvoKit ships with a workaround in the form of a special tokenizer, craft_tokenize, which implements the tokenization scheme used in the CRAFT model. Users of this class should therefore always use craft_tokenize in place of ConvoKit's default tokenization. See `the CRAFT demo notebook <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/forecaster/CRAFT/demos/craft_demo.ipynb>`_ for an example of how to do this.

.. automodule:: convokit.forecaster.CRAFTModel
    :members:
