Forecaster
==========

The Forecaster class provides a generic interface to *conversational forecasting models*, a class of models designed to computationally capture the trajectory
of conversations in order to predict future events. Though individual conversational forecasting models can get quite complex, the Forecaster API abstracts
away the implementation details into a standard fit-transform interface.

For end users of Forecaster: see the demo notebook which `uses Forecaster to fine-tune the CRAFT forecasting model on the CGA-CMV corpus <https://github.com/CornellNLP/ConvoKit/blob/master/examples/forecaster/CRAFT%20Forecaster%20demo.ipynb>`_

For developers of conversational forecasting models: Forecaster also represents a common framework for conversational forecasting
that you can use, in conjunction with other ML/NLP ecosystems like PyTorch and Huggingface, to streamline the development of your models!
You can create your conversational forecasting model as a subclass of ForecasterModel, which can then be directly "plugged in" to the
Forecaster wrapper which will provide a standard fit-transform interface to your model. At runtime, Forecaster will feed a temporally-ordered
stream of conversational data to your ForecasterModel in the form of "context tuples". Context tuples are generated in chronological order,
simulating the notion that the model is following the conversation as it develops in real time and generating a new prediction every time a
new utterance appears (e.g., in a social media setting, every time a new comment is posted). Each context tuple, in turn, is defined as a
NamedTuple with the following fields:

* ``context``: a chronological list of Utterances up to and including the most recent Utterance at the time this context was generated. Beyond the chronological ordering, no structure of any kind is imposed on the Utterances, so developers of conversational forecasting models are free to perform any structuring of their own that they desire (so yes, if you want, you can build conversational graphs on top of the provided context!)
* ``current_utterance``: the most recent utterance at the time this context tuple was generated. In the vast majority of cases, this will be identical to the last utterance in the context, except in cases where that utterance might have gotten filtered out of the context by the preprocessor (in those cases, current_utterance still reflects the "missing" most recent utterance, in order to provide a reference point for where we currently are in the conversation)
* ``future_context``: during **training only** (i.e., in the fit function), the context tuple also includes this additional field that lists all future Utterances; that is, all Utterances chronologically after the current utterance (or an empty list if this Utterance is the last one). This is meant only to help with data preprocessing and selection during training; for example, CRAFT trains only on the last context in each conversation, so we need to look at future_context to know whether we are at the end of the conversation. It **should not be used as input to the model**, as that would be "cheating" - in fact, to enforce this, future_context is **not available during evaluation** (i.e. in the transform function) so that any model that improperly made use of future_context would crash during evaluation!
* ``conversation_id``: the Conversation that this context-reply pair came from. ForecasterModel also has access to Forecaster's labeler function and can use that together with the conversation_id to look up the label

Illustrative example, a conversation containing utterances ``[a, b, c, d]`` (in temporal order) will produce the following four context tuples, in this exact order:
#. ``(context=[a], current_utterance=a, future_context=[b,c,d])``
#. ``(context=[a,b], current_utterance=b, future_context=[c,d])``
#. ``(context=[a,b,c], current_utterance=c, future_context=[d])``
#. ``(context=[a,b,c,d], current_utterance=d, future_context=[])``


.. automodule:: convokit.forecaster.forecaster
    :members:

.. automodule:: convokit.forecaster.forecasterModel
    :members:

Forecaster Model
================
These are subclasses of ForecasterModel, each implementing forecasting models using different model architectures or families.

.. toctree::
   :maxdepth: 1

   CRAFT Model <craftmodel.rst>
   Transformer Encoder-based Model <transformerencodermodel.rst>
   Transformer Decoder-based Model <transformerdecodermodel.rst>
   Transformer Forecaster Configuration <transformerforecastertraining.rst>

   The following table is the current leaderboard comparing the performance of different forecaster models following a uniform evaluation framework described in Son et al, 2025. If you want to include the performance of another model in this leaderboard, make a pull request with the respective ForecasterModel class and with the version of this `demo <https://github.com/CornellNLP/ConvoKit/blob/master/examples/forecaster/Run%20Transformer%20Fine-tuned%20Models.ipynb>`_ that generates the respective new leaderboard line.

+----------------+-------+------+-------+-------+------+----------+-------------------------+
| Model          | Acc ↑ | P ↑  | R ↑   | F1 ↑  | FPR ↓| Mean H ↑ | Recovery ↑              |
+================+=======+======+=======+=======+======+==========+=========================+
| Gemma2 9B      | 71.0  | 69.1 | 76.1  | 72.3  | 34.2 | 3.9      | +1.8 (8.4 - 6.6)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| Mistral 7B     | 70.7  | 68.8 | 76.0  | 72.1  | 34.6 | 4.0      | +2.9 (8.1 - 5.2)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| Phi4 14B       | 70.5  | 67.7 | 78.4  | 72.6  | 37.5 | 4.0      | +2.0 (7.7 - 5.7)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| LlaMa3.1 8B    | 70.0  | 68.8 | 73.2  | 70.9  | 33.2 | 4.0      | +1.7 (7.3 - 5.6)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| DeBERTaV3-large| 68.9  | 67.3 | 73.7  | 70.3  | 36.0 | 4.2      | +1.1 (7.6 - 6.5)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| RoBERTa-large  | 68.6  | 67.1 | 73.4  | 70.0  | 36.1 | 4.2      | +1.6 (7.5 - 5.9)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| RoBERTa-base   | 68.1  | 67.3 | 70.6  | 68.8  | 34.4 | 4.2      | +0.7 (7.4 - 6.7)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| DeBERTaV3-base | 67.9  | 66.7 | 71.4  | 69.0  | 35.7 | 4.2      | +1.5 (7.2 - 5.7)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| SpanBERT-large | 67.0  | 65.8 | 70.5  | 68.1  | 36.6 | 4.2      | +1.3 (8.3 - 7.0)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| SpanBERT-base  | 66.4  | 64.7 | 72.0  | 68.2  | 39.3 | 4.4      | +1.7 (9.6 - 8.0)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| BERT-large     | 65.7  | 66.0 | 65.4  | 65.5  | 34.1 | 4.2      | +0.4 (7.8 - 7.3)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| BERT-base      | 65.3  | 64.1 | 70.1  | 66.9  | 39.5 | 4.4      | +1.9 (9.7 - 7.8)        |
+----------------+-------+------+-------+-------+------+----------+-------------------------+
| CRAFT          | 62.8  | 59.4 | 81.1  | 68.5  | 55.5 | 4.7      | +4.9 (12.0 - 7.1)       |
+----------------+-------+------+-------+-------+------+----------+-------------------------+

**Table 1: Forecasting derailment on CGA-CMV-large conversations.**
The performance is measured in accuracy (Acc), precision (P), recall (R), F1, false positive rate (FPR), mean horizon (Mean H), and Forecast Recovery (Recovery) along with the correct and incorrect recovery rates. Results are reported as averages over five runs with
different random seeds.
