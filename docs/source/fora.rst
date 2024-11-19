Fora Corpus
=============
Fora corpus is a dataset of 262 annotated transcripts of multi-person facilitated dialogues regarding issues like education, elections, and public health, primarily through the sharing of personal experience. The corpus is available by request from the authors (`https://github.com/schropes/fora-corpus <https://github.com/schropes/fora-corpus>`_) and ConvoKit contains code for converting the transcripts into ConvoKit format, as detailed below.

A full description of the dataset can be found here: `Schroeder, H., Roy, D., & Kabbara, J. (2024). Fora: A corpus and framework for the study of facilitated dialogue. In L.-W. Ku, A. Martins, & V. Srikumar (Eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 13985–14001). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.754. <https://doi.org/10.18653/v1/2024.acl-long.754>`_
Please cite this paper when using Fora in your research.

Usage
-----

Request Fora Corpus from (transcripts only): `https://github.com/schropes/fora-corpus <https://github.com/schropes/fora-corpus>`_

Convert the Fora Corpus into ConvoKit format using this notebook `Converting Fora Corpus to ConvoKit Format <https://github.com/CornellNLP/ConvoKit/blob/master/examples/dataset-examples/FORA/ConvoKit_Fora_Conversion.ipynb>`_

Dataset details
---------------

All ConvoKit metadata attributes preserve the names used in the original corpus, as detailed `here. <https://github.com/schropes/fora-corpus>`_

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

There were 1776 unique participants. The following information is recorded in the speaker level metadata:

Metadata for each speaker include:
    * speaker_name : Usually, first name or pseudonym of the speaker (str).
    * is_fac : Whether the current speaker is a facilitator (boolean).
    * location : Location of the conversation (str).
    * source_type : Information about the type of audio input (str).

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance we provide:

* id: Unique identifier for an utterance.
* conversation_id: Utterance id corresponding to the first utterance of the conversation.
* reply_to: Utterance id of the previous utterance in the conversation.
* speaker: Speaker object corresponding to the author of this utterance.
* text: Textual content of the utterance.

Metadata for each utterance include:

    * original_index: Index of the original entry in the dataset.
    * collection_title: String title of the collection.
    * collection_id: Numeric identifier of the conversation collection.
    * SpeakerTurn: Index of speaker turn within the conversation (1-indexed).
    * audio_start_offset: In number of seconds, offset within the recording at which point the speaker turn begins.
    * audio_end_offset: In number of seconds, offset within the recording at which point the speaker turn ends.
    * duration: In number of seconds, duration of the speaker turn.
    * conversation_id: Unique identifier for the conversation.
    * speaker_id: Unique int identifier of the speaker within the conversation. Speakers who participated in multiple conversations do not have a persistent speaker_id - these are unique to each conversation.
    * speaker_name: Usually, first name or pseudonym of the speaker. This field may have been anonymized in cases where the last name was provided. Overall, it is reliable but may have occasional diarization errors.
    * words: String of all words in the speaker turn.
    * is_fac: Boolean representing whether the current speaker is a facilitator.
    * cofacilitated: Boolean representing whether the current conversation has more than one facilitator.
    * annotated: Boolean representing whether the conversation was annotated by human experts for facilitation strategies and personal sharing.
    * start_time: Date of the conversation start time. Likely reliable as the date the conversation happened, but may be approximate due to potential delay in uploading.
    * source_type: String providing information about the type of audio input (e.g., Zoom, Hearth, iPhone).
    * location: Represents the location of the conversation, typically a town or neighborhood. About 1/3 of conversations do not have a value for this field and are marked "Unknown."
    * Personal story: Binary label representing the presence of a "Personal story" as annotated by a human.
    * Personal experience: Binary label representing the presence of a "Personal experience" as annotated by a human.
    * Express affirmation: Binary label representing the presence of "Express affirmation" as annotated by a human.
    * Specific invitation: Binary label representing the presence of "Specific invitation" as annotated by a human.
    * Provide example: Binary label representing the presence of "Provide example" as annotated by a human.
    * Open invitation: Binary label representing the presence of "Open invitation" as annotated by a human.
    * Make connections: Binary label representing the presence of "Make connections" as annotated by a human.
    * Express appreciation: Binary label representing the presence of "Express appreciation" as annotated by a human.
    * Follow up question: Binary label representing the presence of "Follow up question" as annotated by a human.

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each Conversation, we provide the following metadata:

* collection_id: Numeric identifier of the conversation collection.

* conversation_id: Unique identifier for the conversation.

* cofacilitated: Boolean representing whether the current conversation has more than one facilitator.

* annotated: Boolean representing whether the conversation was annotated by human experts for facilitation strategies and personal sharing.

* start_time: Date of the conversation start time. Likely reliable as the date the conversation happened, but may be approximate due to potential delay in uploading.

* source_type: String providing information about the type of audio input (e.g., Zoom, Hearth, iPhone).

* location: Represents the location of the conversation, typically a town or neighborhood. About 1/3 of conversations do not have a value for this field and are marked "Unknown."


Statistics about the dataset
----------------------------

* Number of Speakers: 1776
* Number of Utterances: 39911
* Number of Conversations: 262

Additional note
---------------
Data License
^^^^^^^^^^^^

ConvoKit is not distributing the corpus separately, and thus no additional data license is applicable.  The license of the original distribution applies.

Contact
^^^^^^^

Questions about the conversion into ConvoKit format should be directed to Sean Zhang <kz88@cornell.edu>

Questions about the Fora corpus should be directed to the corresponding authors Hope Schroeder <hopes@mit.edu>, Deb Roy <dkroy@mit.edu>, and Jad Kabbara <jkabbara@mit.edu> of the original paper.