NewsInterview Corpus
====================

A collection of 500 two-person informational interviews from National Public Radio (NPR) and Cable News Network (CNN), containing 16,396 utterances from 860 speakers. The dataset focuses on journalistic interviews between interviewers and sources, from 2000 to 2020.

A full description of the dataset can be found here:
`NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews <https://aclanthology.org/2025.acl-long.1580/>`_.
Alexander Spangher, Michael Lu, Sriya Kalyan, Hyundong Justin Cho, Tenghao Huang, Weiyan Shi, Jonathan May.
Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are identified by unique IDs. Each speaker has the following metadata:

* display_name: original speaker name as it appears in the transcript
* role: speaker type — one of ``HOST`` (interview host/anchor; 76 speakers), ``GUEST`` (interview subject/interviewee, default if not specified; 738 speakers), or ``BYLINE`` (reporter/correspondent; 46 speakers)
* programs: list of programs this speaker appears in
* num_interviews: total number of interviews the speaker participated in


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to a single speaking turn in an interview. For each utterance, we provide:

* id: unique utterance identifier
* speaker: speaker ID reference
* conversation_id: ID of the interview this utterance belongs to
* reply_to: ID of the previous utterance (for threading)
* timestamp: time marker (if available)
* text: textual content of the utterance

Metadata for each utterance include:

* interview_id: original interview identifier
* turn_order: position in the conversation sequence
* program: NPR/CNN program name
* date: interview broadcast date
* url: source URL (when available)


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation represents a complete interview. Metadata associated with conversations include:

* title: interview title (when available)
* summary: interview summary or description
* program: source program name (63 unique programs total)
* date: broadcast/publication date (ranging from 2000 to 2020)
* url: original source URL
* info_items: extracted information items from the interview
* info_items_dict: structured version of information items
* outlines: interview objectives/outline


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("news-interview"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 860
Number of Utterances: 16396
Number of Conversations: 500


Additional notes
----------------

Data License
^^^^^^^^^^^^

This dataset is shared under the `Creative Commons Attribution 4.0 International License <https://creativecommons.org/licenses/by/4.0/>`_.

Dataset Access
^^^^^^^^^^^^^^

The original dataset can be accessed from the authors' GitHub repository at: `https://github.com/alex2awesome/news-interview-question-generation <https://github.com/alex2awesome/news-interview-question-generation>`_

Contact
^^^^^^^

ConvoKit formatted corpus was created by Axel Bax (adb333@cornell.edu) from the dataset created by Sarkar et al.
Corresponding Author: Rupak Sarkar (rupak@umd.edu).
