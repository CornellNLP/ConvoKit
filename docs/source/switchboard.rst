Switchboard Dialog Act Corpus
===============================

A collection of 1,155 five-minute telephone conversations between two participants, annotated with speech act tags.
In these conversations, callers question receivers on provided topics, such as child care, recycling, and news media.
440 speakers participate in these 1,155 conversations, producing 221,616 utterances (we combine consecutive utterances by the same person into one utterance, so our corpus has 122,646 utterances).

This is a Convokit-formatted version of the Switchboard Dialog Act Corpus (SwDA), originally distributed together with the following paper: Andreas Stolcke, Klaus Ries, Noah Coccaro, Elizabeth Shriberg, Rebecca Bates, Daniel Jurafsky, Paul Taylor, Rachel Martin, Carol Van Ess-Dykema, and Marie Meteer. `Dialogue act modeling for automatic tagging and recognition of conversational speech <https://www.aclweb.org/anthology/J00-3003.pdf>`_. Computational Linguistics, Volume 26, Number 3, September 2000.
The original dataset and additional information can be found `here <http://compprag.christopherpotts.net/swda.html>`_.

Processed dataset
------------------

We also release a processed version of the corpus that removes disfluencies and backchannels. This version is intended to be used for quick back-of-the-envelope analyses, where we want to sidestep challenges arising from using transcribed speech data. In sidestepping these challenges, we emphasize that the version omits a lot of important information, and should only be used for experimentation.

In particular, disfluencies are removed via regex, and backchannels are assumed to be utterances with fewer than 5 tokens; successive turns by the same speaker that are broken up by these inferred-backchannels are merged together, and their tags are merged as well.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

In this dataset, speakers are the participants in the phone conversations (two per conversation). The speaker's ID is the same as the ID used in the original SwDA dataset.

Additional metadata include:

* sex: speaker sex, 'MALE' or 'FEMALE'
* education: the speaker's level of education. Options are 0 (less than high school), 1 (less than college), 2 (college), 3 (more than college), and 9 (unknown).
* birth_year: the speaker's birth year (4-digit year)
* dialect_area: one of the following dialect areas: MIXED, NEW ENGLAND, NORTH MIDLAND, NORTHERN, NYC, SOUTH MIDLAND, SOUTHERN, UNK, WESTERN (where UNK tag is used for speakers of unknown dialect area, and MIXED tag is used for speakers who are of multiple dialect areas).


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to a turn by one speaker.

* id: the unique ID of the utterance. It is formatted as "_conversation_id_"-"_position_of_utterance_". For example, ID 4325-0 is the first utterance in the conversation with ID 4325.
* speaker: the Speaker giving the utterance
* conversation_id: id of the first utterance in the conversation this utterance belongs to. For example, the conversation_id of the utterance with ID 4325-1 would be 4325-0.
* reply_to: id of the utterance this replies to (None if the utterance is not a reply)
* timestamp: timestamp of the utterance (not applicable in SwDA, set to None)
* text: textual content of the utterance

Additional metadata includes:

* tag: a list of [text segment, tag] pairs, where tag refers to the `DAMSL speech act tag <https://web.stanford.edu/~jurafsky/ws97/manual.august1.html>`_.

In the processed dataset, metadata includes:

* alpha_text: version of the text containing alphabetical tokens only
* full_tags: all tags associated with the utterance (including those collected across multiple merged utterances)
* tags: processed, deduplicated version of full_tags
* next_id: id of the utterance replying to this one (None if the utterance has no reply)


Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations are indexed by the id of the utterance that starts the conversation.

Additional metadata include:

* filename: the name of corresponding file in the original SwDA dataset
* talk_day: the date of the conversation
* topic_description: a short description of the conversation prompt
* length: length of the conversation in minutes
* prompt: a long description of the conversation prompt
* from_caller: id of the from-caller (A) of the conversation
* to_caller: id of the to-caller (B) of the conversation

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("switchboard-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 440
Number of Utterances: 122646
Number of Conversations: 1155

The processed dataset can be accessed as:
>>> corpus = Corpus(filename=download("switchboard-processed-corpus"))


Additional note
---------------

* In the original SwDa dataset, utterances are not separated by speaker, but rather by tags. This means that consecutive utterances could have been said by the same speaker. In the ConvoKit Corpus, we changed this so that each utterance in our corpus is a collection of the consecutive sub-utterances said by one person. The metadata on each utterance is combined from the sub-utterances of the original dataset, so that it is clear which DAMSL tags correspond with which parts of each utterance. The original dataset also offers POS and parse tree information for utterances, which are not currently included.

* You should pull the repo at `its original github page <https://github.com/cgpotts/swda>`_ in order to download the dataset and helper functions necessary to create the corpus.

Licensing Information
^^^^^^^^^^^^^^^^^^^^^

The SWDA Switchboard work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License (see source `here <http://compprag.christopherpotts.net/swda.html>`_)


Contact
^^^^^^^

Corpus translated into ConvoKit format by [Nathan Mislang](mailto:ntm39@cornell.edu), [Noam Eshed](mailto:ne236@cornell.edu), and [Sungjun Cho](mailto:sc782@cornell.edu).
