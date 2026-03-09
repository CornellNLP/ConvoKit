Tennis Interviews
=================

Transcripts for tennis singles post-match press conferences for major tournaments between 2007 to 2015  (6,467 post-match press conferences).

Distributed together with:
`Tie-breaker: Using language models to quantify gender bias in sports journalism <http://www.cs.cornell.edu/~liye/tennis.html>`_.
Liye Fu, Cristian Danescu-Niculescu-Mizil, Lillian Lee
IJCAI workshop on NLP meets Journalism, 2016.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are tennis professional players, represented by their real names. As this dataset do not contain information about individual reporters, we use a single pseudo user with username "REPORTER" to represent them.

For each player, additional metadata include:

* gender: player gender


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each question or answer is viewed as an utterance. For each utterance, we provide:

* id: index of the utterance
* speaker: the speaker who authored the utterance
* conversation_id: id of the first utterance in the conversation this utterance belongs to
* reply_to: id of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for each utterance include:

* is_answer: whether the utterance is an answer from a player
* is_question: whether the utterance is a question raised by a reporter
* pair_idx: index of the question-answer pair
* parsed: parsed version of the utterance text, represented as a SpaCy Doc


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each round of question-answer pair is considered as a conversation. Metadata associated with conversations include additional information about the match for which the post-match interview is held:

* match_id: id of the match in the original dataset
* opponent: opponent in the match (available only if the opponent has at least one interview recorded in our dataset)
* result: outcome of the match (1 indicates the player being interviewed has won the match; 0 otherwise)
* stage: stage of the tournament (e.g., 'The Final')
* tournament: tournament name
* tournament_type: type of the tournament, indicating tournament prestige
* player_ranking: ranking of the player at the time of the match


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("tennis-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 359
Number of Utterances: 163948
Number of Conversations: 81974

Additional note
---------------

Related links
^^^^^^^^^^^^^

1. Transcripts gathered from `ASAP sports' website <http://www.asapsports.com/>`_.
2. Corresponding match information, such as game outcome and player ranking, are obtained from `Tennis-Data <http://www.tennis-data.co.uk/>`_. Since transcripts data and match results are matched by date and player last name, and we did not manually check for every match, it is possible to have a few matching errors.

Contact
^^^^^^^

Please email any questions to: lf383@cornell.edu (Liye Fu).
