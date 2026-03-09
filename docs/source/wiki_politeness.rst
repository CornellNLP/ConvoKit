Stanford Politeness Corpus (Wikipedia)
====================================================

A collection of requests from Wikipedia Talk pages, annotated with politeness (4,353 utteranecs).

Distributed together with: A Computational Approach to Politeness with Application to Social Factors. Cristian Danescu-Niculescu-Mizil, Moritz Sudhof, Dan Jurafsky, Jure Leskovec, Christopher Potts. ACL, 2013.

Dataset details
---------------


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to a Wikipedia Talk Page request. For each utterance, we provide:

* id: ID of the request given in the original data release.
* speaker: the author of the utterance
* conversation_id: id of the first utterance in the conversation this utterance belongs to, which in this case is the id of the utterance itself
* reply_to: None. In this dataset, each request is seen as a full conversation, and thus all utterances are at the 'root' of the conversations
* timestamp: "NOT_RECORDED".
* text: textual content of the utterance.

Metadata for each utterance is inherited from the general CMV corpus:

* Normalized Score: Normalized politeness score computed based on annotations.
* Binary: A binarized politeness label where 1="polite", 0="neutral", -1 = "impolite".
* Annotations: the original annotations from Amazon Mechanical Turkers for the given utterance. Ratings are on a 1-25 scale.
* parsed: dependency-parsed version of the utterance text


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("wikipedia-politeness-corpus"))

For some quick stats:

>>> len(corpus.get_utterance_ids())
4353

Data License
^^^^^^^^^^^^

ConvoKit's Stanford Politeness Corpus is governed by the `CC BY license v4.0 <https://creativecommons.org/licenses/by/4.0/>`_. Copyright (C) 2017-2020 The ConvoKit Developers.

Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)
