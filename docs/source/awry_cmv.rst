Conversations Gone Awry Dataset - Reddit CMV version (CGA-CMV)
==============================================================

A collection of conversations from the ChangeMyView (CMV) subreddit that derail into personal attacks (6,842 conversations, 42,964 comments).

Summaries of conversation dynamics (SCDs) are available for a subset of the conversations.

Distributed together with: Trouble on the Horizon: Forecasting the Derailment of Online Conversations as they Develop. Jonathan P. Chang and Crisitan Danescu-Niculescu-Mizil. EMNLP 2019.

Summaries of conversation dynamics described in: How Did We Get Here? Summarizing Conversation Dynamics.  Yilun Hua, Nick Chernogor, Yuzhe Gu, Seoyon Julie Jeong, Miranda Luo, Cristian Danescu-Niculescu-Mizil. NAACL 2024.

Example usage of the corpus and summaries: `SCD and Basic Examples <https://github.com/CornellNLP/ConvoKit/blob/master/examples/conversations-gone-awry-cmv/scd-example.ipynb>`_

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are Reddit users; their account names are taken as the user names.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to a Reddit comment. For each utterance, we provide:

* id: Reddit ID of the comment represented by the utterance
* speaker: the speaker who authored the utterance
* conversation_id: id of the first utterance in the conversation this utterance belongs to. Note that this differs from how 'conversation_id' is treated in ConvoKit's general Reddit corpora: in those corpora a conversation is considered to start with a Reddit post utterance, whereas in this corpus a conversation is considered to start with a top-level reply to a post.
* reply_to: Reddit ID of the utterance to which this utterance replies to (None if the utterance represents a top-level comment, i.e., a reply to a post)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for each utterance is inherited from the general CMV corpus:

* score: score (i.e., the number of upvotes minus the number of downvotes) of the content
* top_level_comment: the id of the top level comment (None if the utterance is a post)
* retrieved_on: unix timestamp of the time of when the data is retrieved
* gilded: gilded status of the content
* gildings: gilding information of the content
* stickied: stickied status of the content
* permalink: permanent link of the content
* author_flair_text: flair of the author


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata for each conversation include:

* pair_id: the id of the conversation that this conversation is paired with
* has_removed_comment: whether the final comment in this thread was removed by CMV moderators for violation of Rule 2
* split: which split (train, val, or test) this conversation was used in for the experiments described in "Trouble on the Horizon"
* summary_meta: metadata related to conversation summaries, a list of dictionaries (one per summary available, possibly empty) with the following keys:
    * summary_text: the text of the summary;
    * summary_type: whether the summary is humman written by humans;(human_written_SCD) or generated automatically using the procedural prompt ("machine_generated_SCD") ;
    * up_to_utterance_id: the last utterance considered when creating the summary;
    * truncated_by: the number of utterances the transcript was truncated by when creating the summary (starting from the end);
    * scd_split: whether the summary was in the train/test/validation split in the 2024 Summarizing Conversations Dynamics paper;


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 9548
Number of Utterances: 42964
Number of Conversations: 6842

Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)
