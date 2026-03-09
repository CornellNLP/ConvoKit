Winning Arguments (ChangeMyView) Corpus
=======================================

A metadata-rich subset of conversations made in the r/ChangeMyView subreddit between 1 Jan 2013 - 7 May 2015, with information on the delta (success) of a user's utterance in convincing the poster.

Each ConvoKit Conversation in this dataset is the corresponding full comment thread of an original post made to r/ChangeMyView. Within each full thread are comments made by Redditors (with the objective of the subreddit being to change the opinion of the original poster.) There are 34911 Speakers, 293297 Utterances, and 3051 Conversations.

Original dataset was distributed together with:
`Winning Arguments: Interaction Dynamics and Persuasion Strategies in Good-faith Online Discussions: A new Approach to Understanding Coordination of Linguistic Style in Dialogs <https://chenhaot.com/pages/changemyview.html>`_. Chenhao Tan, Vlad Niculae, Cristian Danescu-Niculescu-Mizil, Lillian Lee.
In Proceedings of the 25th International World Wide Web Conference (WWW'2016).

In the original "Winning Arguments" paper, this corpus was used in a paired prediction task predicting for whether a reply thread (starting from a top-level comment in the comment thread) was successful in convincing the original poster. As stated in Section 4 of the original paper, the threads were paired by first selecting a reply thread that wins a ∆ (i.e. was successful in convincing the OP), then paired with an unsuccessful reply thread in the same discussion tree that did not win a ∆ but was the most “similar” in topic, as measured by Jaccard similarity. The corpus exposes these successful-unsuccessful pairs used in the original paper through Conversation and Utterance level metadata. It additionally provides the other sibling reply threads for context (as part of the full comment thread.)

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are Redditors and are indexed by their Reddit username. There is no other metadata information.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id: index of the utterance (unique comment identification provided by Reddit)
* speaker: the unique id of the user who authored the utterance
* conversation_id: comment identifier of the original post in the thread that this comment was posted in
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: utterance timestamp provided by Reddit API
* text: the full text (in string) of the comment

Metadata for utterances include:

* success: an indicator taking the value of 1 if the comment was part of a successful argument thread (i.e. an argument thread that changed the OP's mind), 0 if unsuccessful, and None if not part of either a successful or unsuccessful thread.
* pair_ids: every successful-unsuccessful argument pair originally compiled by the authors has a unique pair_id. However, it is important to note that not every argument is unique (i.e. a single negative argument within a conversation could have two opposing positive arguments, which necessitates two corresponding pair_ids. Therefore, pair_ids is a list).
* replies: a list of comment ids that respond directly to the current comment. For the OP post in the thread, this was constructed by selecting all comment ids with a "reply_to" field equal to the original post id (this was necessary because the original data provided by the authors did not include all the children of the OP post in their data format). For all comments besides the original post, the "replies" field was originally provided by Reddit API.
* originally provided by Reddit API: author_flair_text, author_flair_css_class, banned_by, controversiality, edited, distinguished, user_reports, ups, downs, subreddit_id, subreddit, score_hidden, score, saved, report_reasons, mod_reports,  num_reports, likes, gilded,approved_by (see the `Reddit API <https://www.reddit.com/dev/api/>`_ for more information)

Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations are indexed by the unique comment identifier of the original post, which started the conversation thread.

Metadata for conversations include:

* op-userID: the Reddit username of the original poster (OP)
* op-text-body: the text of OP's first post (starting the conversation)
* op-title: the title of OP's first post
* pair_ids: a list of all the successful-unsuccessful pairs in the conversation (to subset the successful from unsuccessful Utterances, refer to the "success" metadata indicator in Utterance-level meta (above))
* train, an indicator taking the value of 1 if this conversation was included in the training data subset of the original paper and 0 if it was included in the holdout set.

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("winning-args-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 34911
Number of Utterances: 293297
Number of Conversations: 3051

We provide a `Jupyter notebook <https://github.com/CornellNLP/ConvoKit/blob/master/datasets/winning-args-corpus/stats.ipynb>`_ that demonstrates how to subset the data into successful and unsuccessful arguments (as described in Section 4 of the original paper) and provides basic statistics about their respective comments.

Additional notes
----------------
- To see the original posts in each conversation, note that the 'conversation_id' field of an Utterance (i.e. Reddit comment) would be the same as the 'id' metadata (this only holds true for original posts).
- The main indicator of interest in this data is whether an argument succeeded in changing the original poster's (OP's) view. To denote a successful argument: the "success" field of an utterance takes the value of 1 (these utterances were the comments in the original post that succeeded in changing OP's mind), or the "success" field takes the value of 0 (this collection of utterances were a comment thread that are similar in nature to a successful argument in the full thread(matched on pair_id), but this argument failed to change OP's mind -- See section 4 of the cited paper for selection criteria of successful/unsuccessful arguments. All other comments take a 'success' value of None.
- Note for pair_ids: the successful-unsuccessful argument pairs originally compiled by the authors are not unique at the Conversation-level nor Utterance-level (i.e. the original posts to the ChangeMyView subreddit can have multiple successful-unsuccessful comment pairs in their full-comment threads and some comments can have multiple opposing pairs -- see the relevant metadata fields at Conversation-level and Utterance-level below).
- Note on missing data: 530 'Reddit comments' were included from the original data which did not have text nor an author. These were included for completeness, each utterance has Speaker(name='[missing]').


The original dataset can be downloaded `here <https://chenhaot.com/pages/changemyview.html>`_. Refer to the original README for more explanations on dataset construction.

Contact
^^^^^^^

Corpus converted into ConvoKit format by Andrew Szmurlo and Meir Friedenberg. Please email any questions to: as3934@cornell.edu.
