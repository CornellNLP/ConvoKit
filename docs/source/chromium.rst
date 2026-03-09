Chromium Conversations Corpus
===============================

A collection of almost 1.5 million conversations and 2.8 million comments posted by developers reviewing proposed code changes in the Chromium project.

Contributed by: Benjamin S. Meyers (bsm9339@rit.edu)

Distributed together with: Benjamin S. Meyers, Nuthan Munaiah, Emily Prud'hommeaux, Andrew Meneely, Cecilia O. Alm, Josephine Wolff, and Pradeep Murukannaiah. **A Dataset for Identifying Actionable Feedback in Collaborative Software Development.** Proceedings of the 2018 Meeting for the Association for Computational Linguistics (ACL). Melbourne, Australia. http://www.aclweb.org/anthology/P18-2021

A full description of the dataset can be found `here <https://zenodo.org/record/2590548>`_.


Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speaker names have been anonymized randomly to 'developer_#' where '#' is a number between 1 and 4842.

Additional metadata includes:

* user_type: either 'developer', the developer who proposed the code change, or 'reviewer', other developers reviewing the code change

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to a comment in the Chromium project.

* id: index of the utterance
* speaker: the speaker who authored the utterance
* conversation_id: id of the first utterance in the conversation this utterance belongs to
* reply_to: id of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Additional metadata includes some associated pre-calculated linguistic metrics:

* yngve_score: The maximum Yngve score of sentences in the code review comment
* frazier_score: The maximum Frazier score of sentences in the code review comment
* pdensity: The Propositional Density score of the code review comment
* cdensity: The Content Density score of the code review comment
* has_doxastic: Binary indicator of presence of a sentence with doxastic uncertainty in the code review comment
* has_epistemic: Binary indicator of presence of a sentence with epistemic uncertainty in the code review comment
* has_conditional: Binary indicator of presence of a sentence with conditional uncertainty in the code review comment
* has_investigative: Binary indicator of presence of a sentence with investigative uncertainty in the code review comment
* has_uncertainty: Binary indicator of presence of a sentence with any uncertainty in the code review comment
* min_formality: Minimum of the formality of sentences in the code review comment
* max_formality: Maximum of the formality of sentences in the code review comment


Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each conversation has the associated metadata:

* review_id: Unique identifier of a code review in the Chromium project. The URL `https://codereview.chromium.org/<review_id>` may be used to access the review online
* patchset_id: Unique identifier of a code review patchset (i.e., collection of changes to the source code) associated with a review
* patch_id: Unique identifier of a code review patch (i.e., individual change to the source code) associated with a patchset
* file_path: The path to the file being modified in the patch
* line_number: The line number in the file at which the comment was posted

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("chromium-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 4842
Number of Utterances: 2853498
Number of Conversations: 1484843


Additional note
---------------

Data License
^^^^^^^^^^^^

Creative Commons Attribution 4.0 International

Contact
^^^^^^^

Please email any questions to: bsm9339@rit.edu (Benjamin S. Meyers)
