Contextual Abuse Dataset (CAD) Corpus
======================================

This corpus contains around 26,500 annotated Reddit entries (1,394 post titles, 1,394 post bodies, and 23,762 comments). Each entry is labeled into one or more of six primary categories: Identity-directed abuse, Affiliation-directed abuse, Person-directed abuse, Counter Speech, Non-hateful Slurs, and Neutral, with additional secondary subcategories like Derogation, Animosity, Threatening, Dehumanization, and Glorification.

The original dataset can be found here:
`Introducing CAD: the Contextual Abuse Dataset <https://aclanthology.org/2021.naacl-main.182/>`_.
Bertie Vidgen, Dong Nguyen, Helen Margetts, Patricia Rossini, and Rebekah Tromble.
Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 2021.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset correspond to Reddit users. Each speaker is identified from the ``meta_author`` field of the original data. If the author value is missing, marked as NA, or deleted, the speaker ID is set to ``[deleted]``.


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to one Reddit entry (a post title, post body, or comment). For each utterance, we provide:

* id: unique utterance identifier, taken from ``info_id``
* speaker: Reddit username of the author
* conversation_id: identifier for the Reddit thread containing this utterance
* reply_to: ID of the parent post or comment (``info_id.parent``), or None if no valid parent exists
* timestamp: Unix timestamp (in seconds) of when the utterance was created
* text: cleaned textual content of the utterance, with ``[linebreak]`` markers replaced by newlines

Metadata for each utterance include:

* annotation_Primary: main abuse category assigned by trained experts — one of ``Identity-directed abuse``, ``Affiliation-directed abuse``, ``Person-directed abuse``, ``Counter Speech``, ``Non-hateful Slurs``, or ``Neutral``
* annotation_Secondary: abuse subtype, e.g., ``Derogation``, ``Animosity``, ``Threatening``, ``Dehumanization``, ``Glorification``
* annotation_Context: whether additional context is required to interpret the label (``Yes`` / ``No`` / ``NA``)
* annotation_Target: the specific individual or group targeted, e.g., ``Women``, ``Immigrants``, ``Political groups``
* annotation_Target_top.level.category: higher-level target category, e.g., ``Identity``, ``Group``, ``Other``
* annotation_highlighted: text span(s) highlighted by annotators as abusive or offensive content; ``"NA"`` if none
* meta_date: UTC date of utterance creation (YYYY-MM-DD)
* meta_created_utc: Unix timestamp of utterance creation
* meta_day: day of utterance creation (YYYY-MM-DD)
* meta_permalink: Reddit permalink to the original post or comment
* info_subreddit: name of the subreddit where the utterance was posted
* info_subreddit_id: Reddit's internal ID for the subreddit
* id: original CAD-assigned ID (e.g., ``cad_1``, ``cad_2``)
* info_id: original identifier for the utterance (with ``-title`` or ``-post`` suffix)
* info_id.parent: identifier of the parent utterance
* info_id.link: identifier of the original submission that started the thread
* info_thread.id: identifier grouping all utterances in the same Reddit thread
* info_order: order of the utterance within its thread
* info_image.saved: whether an image was saved with the utterance (``0`` = no, ``1`` = yes)
* split: the dataset split in the original project — one of ``train``, ``dev``, ``test``, ``exclude_empty``, ``exclude_bot``, ``exclude_lang``, or ``exclude_image``
* subreddit_seen: whether the subreddit was included in the annotation set (``1``) or not (``0``)
* entry_type: type of the utterance — one of ``title``, ``post``, or ``comment``


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each Reddit thread (grouped by ``info_thread.id``) is treated as a conversation. Within each thread, ``reply_to`` relations establish the comment tree structure.


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("contextual-abuse"))

For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 11123
Number of Utterances: 26550
Number of Conversations: 1395

The counts for the primary labels are as follows - 'Neutral': 21935, 'IdentityDirectedAbuse': 2216, 'AffiliationDirectedAbuse': 1111, 'PersonDirectedAbuse': 951, 'CounterSpeech': 210, 'Slur': 127.

Additional notes
----------------

Data License
^^^^^^^^^^^^
This dataset is shared under the `Creative Commons Attribution 4.0 International License <https://creativecommons.org/licenses/by/4.0/>`_.

Contact
^^^^^^^

The original Contextual Abuse Dataset was distributed in the paper `Introducing CAD: the Contextual Abuse Dataset <https://aclanthology.org/2021.naacl-main.182/>`_ (Vidgen et al., NAACL 2021). Corresponding Author: Bertie Vidgen (bvidgen@turing.ac.uk).

The dataset was formatted for Convokit by Hao Wan (hw799@cornell.edu).
The demo on transformer usage and analysis was provided by Jadon Geathers (jag569@cornell.edu).
