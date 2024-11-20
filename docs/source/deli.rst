DeliData Corpus
===============

DeliData is a dataset designed for analyzing deliberation in multi-party problem-solving contexts. It contains information about group discussions, capturing various aspects of participant interactions, message annotations, and team performance.

The corpus is available upon request from the authors, and a ConvoKit-compatible version can be derived using ConvoKit’s conversion tools. ConvoKit also host the ConvoKit-format deli corpus, which can be directly downloaded following instruction in the Usage section.

For a full description of the dataset collection and potential applications, please refer to the original publication: `Karadzhov, G., Stafford, T., & Vlachos, A. (2023). DeliData: A dataset for deliberation in multi-party problem solving. Proceedings of the ACM on Human-Computer Interaction, 7(CSCW2), 1-25.`

Please cite this paper when using DeliData corpus in your research.

Dataset details
---------------

All ConvoKit metadata attributes retain the original names used in the dataset.

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata for each speaker includes the following fields:

* speaker: Identifier or pseudonym of the speaker.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance includes:

* id: Unique identifier for an utterance.
* conversation_id: Identifier for the conversation that the utterance belongs to.
* reply_to: Identifier for the previous utterance in the conversation, if any (null if not a reply).
* speaker: Name or pseudonym of the utterance speaker.
* text: Normalized textual content of the utterance with applied tokenization and masked special tokens.
* timestamp: Null for the entirety of this corpus.

Metadata for each utterance includes:

* annotation_type: Type of utterance deliberation, if annotated (e.g., "Probing" or "Non-probing deliberation"). If unannotated, may be null.
* annotation_target: Target annotation, indicating the intended focus of the message, such as "Moderation" or "Solution." May be null if not annotated.
* annotation_additional: Any additional annotations indicating specific deliberative actions (e.g., "complete_solution"), may be null if not annotated.
* message_type: Type of message, categorized as INITIAL, SUBMIT, or MESSAGE, indicating its function in the dialogue.
* original_text: Original text as said in the collected conversation; For INITIAL type, contains the list of participants and cards presented. For SUBMIT type, contains the cards submitted

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each conversation we provide:

* id: id of the conversation

Metadata for each conversation includes:

* team_performance: Approximate performance of the team based on user submissions and solution mentions, ranging from 0 to 1, where 1 indicates all participants selected the correct solution.
* sol_tracker_message: Extracted solution from the current message content.
* sol_tracker_all: Up-to-date "state-of-mind" for each of the participants, i.e. an approximation of what each participant think the correct solution is at given timestep. This is based on initial solutions, submitted solutions, and solution mentions. team_performance value is calculated based on this column
* performance_change: Change in team performance relative to the previous utterance.

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("deli-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()

* Number of Speakers: 30
* Number of Utterances: 17111
* Number of Conversations: 500

Additionally, if you want to process the original Deli data into ConvoKit format you can use the following script `Converting DeliData to ConvoKit Format <https://github.com/CornellNLP/ConvoKit/blob/master/examples/dataset-examples/DELI/ConvoKit_DeliData_Conversion.ipynb>`_

Additional note
---------------
Data License
^^^^^^^^^^^^

The license of the original distribution applies.

Contact
^^^^^^^

Questions regarding the DeliData corpus should be directed to Georgi Karadzhov (georgi.karadzhov@sheffield.ac.uk).

Files
^^^^^^^

Request the Official Released DeliData Corpus without ConvoKit formatting: https://delibot.xyz/delidata
