Reddit Corpus (small)
=====================

A sample of conversations from Reddit from 100 highly active subreddits. From each of these subreddits, we include 100 comments threads that has at least 10 comments each during September, 2018. The complete list of subreddits included can be found `here <https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/subreddits_small_sample.txt>`_.


Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

speakers in this corpus are Reddit speakers, identified by their account names. The corpus includes the following activity statistics:

* num_posts: number of posts from the speaker
* num_comments: number of comments from the speaker


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each individual post or comment is viewed as an utterance. For each utterance, we provide:

* id: index of the utterance
* speaker: the speaker who authored the utterance
* conversation_id: id of the first utterance in the conversation this utterance belongs to
* reply_to: id of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for each utterance may include:

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

Each post with its corresponding comments are considered a conversation. For each conversation, we provide:

* title: title of the post
* num_comments: number of comments following this post
* domain: domain of the post
* subreddit: subreddit this post is retrieved from
* gilded: gilded status of the post
* gildings: gilding information of the post
* stickied: stickied status of the post
* author_flair_text: flair of the author


Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^

* subreddit: the list of subreddits included in this corpus
* num_posts: total number of posts included in this corpus
* num_comments: total number of comments in this corpus
* num_speaker: number of unique speakers in this corpus


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("reddit-corpus-small"))

For some quick stats:

>>> corpus.print_summary_stats()
Number of Utterances: 297132
Number of Speakers: 119889
Number of Conversations: 8286

Additional note
---------------

Refer to :doc:`tutorial` for a quick tutorial with ConvoKit with this corpus.
