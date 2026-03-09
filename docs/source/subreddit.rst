Reddit Corpus (by subreddit)
============================

A collection of Corpuses of Reddit data built from `Pushshift.io Reddit Corpus <https://pushshift.io/>`_. Each Corpus contains posts and comments from an individual subreddit from its inception until Oct 2018.

A total of 948,169 subreddits are included, the list of subreddits included in the dataset can be explored `here <https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/>`_. Note that the directories are ordered lexicographically, with capital letters sorted first, so the subreddit KDC is found in JustinYCult~-~Kanye/ rather than Kanye2024~-~Kemetic/.

We also provide a small sample of the collection (see :doc:`reddit-small`).

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this corpus are Reddit speakers, identified by their account names.

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

A subreddit corpus name is always the name of the subreddit with the prefix "subreddit-". For example, the subreddit `Cornell <https://www.reddit.com/r/Cornell>`_. can be downloaded as follows:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("subreddit-Cornell"))

For some quick stats on this subreddit:

>>> corpus.print_summary_stats()
Number of Speakers: 7568
Number of Utterances: 74467
Number of Conversations: 10744

Combining different subreddits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common use case for the subreddit corpora might be to combine related subreddit corpora for further analysis. This is straightforward with the Corpus's merge functionality, which we demonstrate below.

We use the Cornell and ApplyingToCollege subreddits as we would expect some overlap in Speakers that the merge functionality will take into account.

>>> cornell_corpus = Corpus(filename=download("subreddit-Cornell"))
>>> cornell_corpus.print_summary_stats()
Number of Speakers: 7568
Number of Utterances: 74467
Number of Conversations: 10744
>>> a2c_corpus = Corpus(filename=download("subreddit-ApplyingToCollege"))
>>> a2c_corpus.print_summary_stats()
Number of Speakers: 53067
Number of Utterances: 1148299
Number of Conversations: 121007
>>> merged_corpus = cornell_corpus.merge(a2c_corpus)
>>> merged_corpus.print_summary_stats()
Number of Speakers: 59739
Number of Utterances: 1222766
Number of Conversations: 131751

Notice that the numbers of Utterances and Conversations in the merged corpus are simply the sum of those in the constituent corpora. This is to be expected since the Utterances and Conversations from these two subreddits are distinct and non-overlapping.

However, the number of speakers is not the sum of those of the constituent corpora -- undoubtedly because some Speakers have posted to both r/ApplyingToCollege and r/Cornell.

.. During the merge step, we turned warnings off because there would be warnings printed for every instance of conflicting Speaker metadata.

.. Recall that the Speaker metadata consists of (1) the number of posts the Speaker has made and (2) the number of comments the Speaker has made. A Speaker that is present in both subreddit corpora will likely have very different values for these two metrics, and we would thus expect a large volume of warnings.

.. We illustrate this below:

.. merged_corpus = cornell_corpus.merge(a2c_corpus) # warnings are on by default
.. WARNING: Multiple values found for Speaker([('name', 'Aleeo34152')]) for meta key: num_posts. Taking the latest one found
.. WARNING: Multiple values found for Speaker([('name', 'Aleeo34152')]) for meta key: num_comments. Taking the latest one found
..  WARNING: Multiple values found for Speaker([('name', 'DrowsyTiger22')]) for meta key: num_posts. Taking the latest one found
.. WARNING: Multiple values found for Speaker([('name', 'DrowsyTiger22')]) for meta key: num_comments. Taking the latest one found
.. ...

.. Since the num_posts and num_comments metadata is incorrect for the Speakers now, we can simply update them for this new Corpus as follows:

.. for speaker in merged_corpus.iter_speakers():
..  num_posts = sum(utt.root == utt.id for utt in speaker.iter_utterances())
.. speaker.add_meta("num_posts", num_posts)
.. speaker.add_meta("num_comments", len(speaker.get_utterance_ids()) - num_posts)


Additional notes
----------------

1. Some subreddit corpora are large. If the subreddit of interest is highly active, it is advised to check the size of the compressed subreddit corpus file `here <https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/>`_ prior to downloading.

2. This is a beta version release. Not all subreddits that exist are included, and the completeness of subreddit history is not guaranteed. Note that this also implies that some thread structures may be broken: for some utterances, the reply-to ID may not match any utterance that exists in the current version of the data. We hope to provide a more complete version of the dataset in the next release.

3. In some cases, the speaker activity information (i.e., number of posts/comments) may be inflated by duplicated entries in intermediate processing steps. We plan to release further updates to fix this issue.
