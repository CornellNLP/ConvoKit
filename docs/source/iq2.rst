Intelligence Squared Debates Corpus
====================================

This dataset contains transcripts of debates held as part of Intelligence Squared Debates. There are 108 debates in all, that were held between September 2006 and September 2015.

The original dataset was compiled by and featured in the paper Zhang, Justine, et al. Conversational Flow in Oxford-Style Debates. Proceedings of the 2016 Conference of NAACL, 2016 (`link <http://tisjune.github.io/research/iq2>`_).

Date: October 19, 2019

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are the speakers during the debate. The most prominent speakers are those that are listed in the initial dataset as being either ‘for’ a position, ‘against’ it, or a ‘moderator.’ However, there are also other speakers that do not fit into these categories, such as a host or a panelist. The speaker type is annotated at the utterance level rather than the speaker level.

We provide:

* bio: if available, full length bio of speaker
* bio_short: if available, short bio of speaker

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance is a continuous speaking turn of a single speaker. The original dataset broke down each utterance into paragraphs, but here, such speech is condensed into one single block of text. We have marked the indices of the paragraph breaks in the text as described below to preserve that information (useful for the nontext metadata)

We provide:

* nontext: this is a dictionary of nontextual data in the original transcripts, where keys are descriptions of the type of data (e.g. "laughter" indicating audience laughter), and values are lists of tuples of the form [index of paragraph, index of annotation in words in paragraph]. Ex: {"laughter": [[0, 3]]} indicates that this transcript had audience laughter in paragraph 0 before the fourth word. Given that the text for the utterance is not separated into paragraphs, we have included paragraphbreaks to allow this data to be used in this format.
* paragraphbreaks: a list of indices in the utterance text at which there was originally a paragraph break. Ex: [15, 99] would indicate that there are three paragraphs; the first one is text[0:14], the second is text[15:98], and the last is text[99:].
* segment: which segment of the debate the utterance was spoken during, where 0=intro, 1=discussion, and 2=conclusion section
* speakertype: the type of speaker (which side, moderator, panelist, etc.) who is saying the utterance


Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our conversations are indexed by the first utterance of a conversation. This means each debate is represented as one conversation. We also provide debate-level metadata at the conversation level as they are equivalent for our dataset.

We provide:

* summary: a summary of the debate
* title: the title of the debate
* originalid: the debate id in the original dataset
* date: the date of the debate
* url: the url where the debate can be accessed
* winner: indicates which side won a debate, 'for,' 'against,' or 'tie'
* results: a full breakdown of the starting and ending positions of the audience in the debates. These are all represented as percentages. The 'results' json file contains 3 dicts with the following metadata:
* breakdown: entries in this dictionary take the form position1_position2, where position1 is the position at the start of the debate and position2 is the position at the end of the debate. This breakdown is not available for the earlier debates. The entries are thus:

    * against_against
    * against_for
    * against_undecided
    * undecided_against
    * undecided_for
    * undecided_undecided
    * for_against
    * for_for
    * for_undecided
    * pre: percentage of audience that voted 'for,' 'against,' or 'undecided' at the beginning of the debate. This results breakdown follows the same structure as the original dataset.
    * post: percentage of audience that voted 'for,' 'against,' or 'undecided' at the end of the debate.

* speakers: the official speakers of the debate (note our speakers corpus is more expansive than this definition)

Usage
-----

To download with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("iq2-corpus"))

For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 471
Number of Utterances: 26562
Number of Conversations: 108

Additional notes
----------------

Corpus translated into ConvoKit format by Lucas Van Bramer (ljv32@cornell.edu) and Marianne Aubin Le Quere.
