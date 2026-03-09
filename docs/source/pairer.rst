Pairer
======

The Pairer transformer annotates the Corpus with the pairing information that is needed to run some of the paired prediction analyses (e.g. see the documentation for :doc:`PairedPrediction and PairedVectorPrediction transformers <pairedprediction>`).

To explain how the Pairer works in more detail, consider the example of the Friends TV series, referenced with :doc:`paired prediction transformers <pairedprediction>`. We are interested in examining how differently Rachel talks to Monica and Chandler. When considering all utterances by Rachel to Monica and Chandler in the comparative analysis, the differences we observe may inadvertently be due to different topics of conversations. Thus, in order to control for the variable context of conversations, one might want to focus on utterances from conversations in which the three — Rachel, Monica, and Chandler — are all present. More precisely, we would like to pair Rachel’s utterances directed to Monica with utterances directed to Chandler if they are part of the same conversation.

Next, we are going to show how we can set up this pairing from the example using Pairer transformer:

    - The ``obj_type`` is `“utterance”`, since we compare Rachel’s utterances
    - The ``pairing_func`` is supposed to extract the identifier that would identify the object as part of the pair. In this case, that would be the Utterance's conversation id since we want utterances from the same conversation.
    - We need to distinguish between utterances where Rachel speaks to Monica vs. Chandler. The ``pos_label_func`` and ``neg_label_func`` is how we can specify this (e.g. ``lambda utt: utt.meta['target’]``), where positive instances might be arbitrarily refer to targetting Monica, and negative for targetting Chandler.
    - ``pair_mode`` denotes how many pairs to use per context. For example, a Conversation will likely have Rachel address Monica and Chandler each multiple times. This means that there are multiple positive and negative instances that can be used to form pairs. We could randomly pick one pair of instances ("random"), or the first pair of instances ("first"), or the maximum pairs of instances ("maximize").
Pairer saves this pairing information into the object metadata.

    - ``pair_id`` is the "id" that uniquely identifies a pair of positive and negative instances, and is the output from the pairing_func.
    - ``pair_obj_label`` denotes whether the object is the positive or negative instance of the pair
    - ``pair_orientation`` denotes whether to use the pair itself as a positive or negative data point in a predictive classifier. "pos" means the difference between the objects in the pair should be computed as [+ve obj features] - [-ve obj features], and "neg" means it should be computed as [-ve obj features] - [+ve obj features].


.. automodule:: convokit.paired_prediction.pairer
    :members:
