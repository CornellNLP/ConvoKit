Wikipedia Talk Pages Corpus
===========================

A collection of conversations from `Wikipedia editor's talk pages <http://en.wikipedia.org/wiki/Wikipedia:Talk_page_guidelines>`_, with metadata.

Distributed together with: `Echoes of power: Language effects and power differences in social interaction <https://www.cs.cornell.edu/~cristian/Echoes_of_power.html>`_. Cristian Danescu-Niculescu-Mizil, Lillian Lee, Bo Pang, and Jon Kleinberg. WWW 2012.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are Wikipedia editors; their account names are taken as the speaker names. Additional information include:

* is-admin: whether the speaker is an admin
* edit-count: total number of edits the speaker has made


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id: index of the utterance
* speaker: the speaker who author the utterance
* conversation_id: id of the first utterance in the conversation this utterance belongs to
* reply_to: id of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for each utterance include:

* is-admin: whether the utterance is from an admin

The dataset also comes with the following processed fields, which can be loaded separately via `corpus.load_info('utterance',[list of fields])`:

* parsed: SpaCy dependency parse
* arcs_censored: dependency parse arcs, without nouns


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("wiki-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 38462
Number of Utterances: 391294
Number of Conversations: 125292

Additional notes
----------------

Related links
^^^^^^^^^^^^^

1. A legacy (unmaintained) version of the dataset is available here: https://www.cs.cornell.edu/~cristian/Echoes_of_power_files/wikipedia_conversations_corpus_v1.01.zip

Data License
^^^^^^^^^^^^

This dataset is governed by the `CC BY-SA license v4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`_.


Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)
