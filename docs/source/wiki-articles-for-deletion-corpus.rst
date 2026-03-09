Wikipedia Articles for Deletion Corpus
======================================
A collection of Wikipedia's Articles for Deletion editor debates that occurred between January 1, 2005 and December 31, 2018. This corpus contains about 3,200,000 contributions by approximately 150,000 Wikipedia editors across almost 400,000 debates.

This is a ConvoKit-formatted version of the Wikipedia Articles for Deletion `dataset <https://github.com/emayfield/AFD_Decision_Corpus>`_, originally distributed with:

Mayfield, Elijah, and Alan W. Black. `"Analyzing Wikipedia Deletion Debates with a Group Decision-Making Forecast Model." <https://dl.acm.org/doi/10.1145/3359308>`_ Proceedings of the ACM on Human-Computer Interaction 3.CSCW (2019): 1-26.


Dataset details
---------------


Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in the dataset are Wikipedia users participating in Article for Deletion discussions. We use a nine digit string "2XXXXXXXX" from the original dataset to index speakers.

For each speaker, the following information is provided as metadata:

* name - username of the Wikipedia user or an IP address for unregistered participants
* editcount - number of edits associated with the user, or None if not available
* signup - date and time of this user's signup on Wikipedia in ISO 8601 format, or None if not available
* gender - gender provided in the user profile, "unknown" if not provided, or None if not available


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Utterances in this dataset are contributions made by Wikipedia users during Article for Deletion debates. There are three types of contributions: nominations (e.g. nomination for article deletion), votes (e.g. keep/delete), and non-voting comments.

For each utterance, we provide:

* id - index of the contribution as given in the original dataset. For each category, the id is a nine-digit string:

	* nominations - "6XXXXXXXX"
	* votes - "4XXXXXXXX"
	* non-voting comment - "5XXXXXXXX"

* speaker - author of the contribution
* conversation_id - a nine digit string ("1XXXXXXXX") as given by the index of the discussion in the original dataset
* reply_to - index of the parent contribution. The original dataset does not provide values for the "parent" of the contribution. Hence, we introduce the following artificial conversation structure:

	* Every first utterance (nomination, vote, or a non-voting comment) we encounter in the discussion does not have a parent utterance (i.e. reply-to is None)
	* Voting comments and nominations (if they are not already the first contribution in the discussion) are replies to the first utterance in the discussion
	* Non-voting comments are replies to either (i) the previous vote or (ii) the first utterance in the discussion if no vote has been cast yet.

* timestamp - time of the contribution, given in Unix timestamp
* text: text of the contribution

Metadata for utterances from each category gives:

* citations - a list of citations to various Wikipedia policies parsed from the text of the contribution.
* type - "nomination", "vote", or "non-voting comment" depending on the category of the contribution
* label - for voting contributions this provides a label for the vote parsed from the text of the contribution (ex. "keep"); for nominations and non-voting comments, the value is None
* raw_label - for voting contributions this provides a raw label for the vote parsed from the text of the contribution; for nominations and non-voting comments, the value is None


Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations are indexed by the id of the discussion. For each conversation we provide information about the article and the outcome of the discussion. Outcome of the discussion is determined by a Wikipedia user with administrative privileges based on the votes in the discussion.

Below is information provided as metadata for conversations:

* outcome_id - a nine digit index string (“3XXXXXXXX”) for the outcome as provided in the original dataset
* outcome_label - label of the discussion outcome determined by Wikipedia admin user (ex. "delete")
* outcome_raw_label - raw label of the discussion outcome determined by Wikipedia admin user
* outcome_decision_maker_id - Wikipedia admin user who decided on the outcome of the Article for Deletion discussion. Here we provide information as a nine digit string "2XXXXXXXX" ID. **Note** that not all outcome decision-makers appear as speakers in this ConvoKit-formatted corpus.
* outcome_timestamp - Unix timestamp for when the outcome was decided
* outcome_rationale - text provided by the Wikipedia admin user to explain/justify the outcome of the discussion


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("wiki-articles-for-deletion-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 161266
Number of Utterances: 3295340
Number of Conversations: 383918


Additional note
---------------

Data License
^^^^^^^^^^^^

The original Wikipedia Articles for Deletion dataset (see source `here <https://github.com/emayfield/AFD_Decision_Corpus>`_) is licensed under the GNU General Public License v3.0.
