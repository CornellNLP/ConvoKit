Coarse Discourse Sequence Corpus
================================

Data from paper `Characterizing Online Discussion Using Coarse Discourse Sequences. Amy X. Zhang, Bryan Culbertson, and Praveen Paritosh. Proceedings of ICWSM 2017. <https://ai.google/research/pubs/pub46055>`_

Coarse Discourse, the Reddit dataset that contains ~9K threads, with comments annotated with 9 main discourse act labels and an “other” label:

* Question & Request
* Answer
* Announcement
* Agreement
* Appreciation & Positive Reaction
* Disagreement
* Negative Reaction
* Elaboration & FYI
* Humor
* Other

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this Corpus are Reddit users, with their name being their Reddit username. Speakers who deleted their accounts have their name listed as ‘[deleted]’.


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance represents either a top-level Reddit post or a comment on a post. For each utterance, we provide:

* id: unique_id of the utterance. This is the Reddit ID of the post or comment; posts start with t3 and comments with t1
* speaker: author of the post/comment
* conversation_id: id of the first utterance in the conversation this utterance belongs to. For post utterances, the conversation_id is the same as the utterance id
* reply_to: the id of the comment/post that this utterance replies to
* text:  textual content of the utterance, none if there is no body in the text


Additional information including the annotations for discourse actions that are specific to this dataset and the information specific to reddit are contained in the meta data:

* comment_depth: depth of the comment, 0 if the utterance is the top-level post itself.
* majority type: discourse action type by one of the following: question, answer, announcement, agreement,  appreciation, disagreement, elaboration, humor
* annotation_types (list of annotation types by three annotators)
* majority_link : link in relation to previous post, none if no relation with previous comment
* annotation_links (list of annotation links by three annotators)
* ups : number of votes (upvotes - downvotes) for the comment/post


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation has the following metadata:

* subreddit: the name of the subreddit the conversation came from
* url: URL of the original post
* title: title of the post that started this conversation

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("reddit-coarse-discourse-corpus"))

Some stats on the data set:

>>> corpus.print_summary_stats()
Number of Speakers: 63573
Number of Utterances: 115827
Number of Conversations: 9483


Additional notes
----------------
The official dataset distribution from the paper authors contains only comment/post IDs, not text content; the dataset also came with a script to join IDs with text using the Reddit API. This ConvoKit version of the dataset was constructed using that script; however, as some comments may have been deleted in the time between when the paper was published and when the script was run, this Corpus may not correspond 100% to the data used in the paper.

Contact
^^^^^^^
Converted by Ru Zhao, Katy Blumer, Andrew Semmes

Please email any questions to: {rjz46, keb297 , als452} @cornell.edu
