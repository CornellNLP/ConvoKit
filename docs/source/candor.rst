CANDOR Corpus
=============
CANDOR corpus is a dataset of 1650 conversations that strangers had over video chat with rich metadata information obtaind from pre-conversation and post-conversation surveys.  The corpus is available by request from the authors (`BetterUp CANDOR Corpus <https://betterup-data-requests.herokuapp.com/>`_) and ConvoKit contains code for converting the transcripts into ConvoKit format, as detailed below.

A full description of the dataset can be found here: `Andrew Reece et al. ,The CANDOR corpus: Insights from a large multimodal dataset of naturalistic conversation. Sci. Adv.9,eadf3197(2023). <https://www.science.org/doi/10.1126/sciadv.adf3197>`_
Please cite this paper when using CANDOR in your research.

Data Usage
-----------

Request CANDOR Corpus from (transcripts only): `BetterUp CANDOR Corpus <https://betterup-data-requests.herokuapp.com/>`_

Convert the CANDOR Corpus into ConvoKit format using this notebook `Converting CANDOR Corpus to ConvoKit Format <https://github.com/CornellNLP/ConvoKit/blob/master/examples/dataset-examples/CANDOR/candor_to_convokit.ipynb>`_

You will need pick the transcription type when converting CANDOR corpus to ConvoKit that will impact ConvoKit Utterance metadata. See section Utterance-level information below for more detail.

Dataset details
---------------

All ConvoKit metadata attributes preserve the names used in the original corpus, as detailed here `BetterUp CANDOR Corpus Data Dictionary <https://docs.google.com/spreadsheets/d/1ADoaajRsw63WpM3zS2xyGC1YS5WM_IuhFZ94W84DDls/edit#gid=997152539>`_

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

There were 1454 unique participants from a broad range of backgrounds. The following information is recorded in the speaker level metadata:

Metadata for each speaker include:
    * sex: gender of speaker
    * politics: political persuasion the speaker most identify (from very conservative to very liberal)
    * race: race/ethnicity of speaker
    * edu: highest level of school the speaker have completed or received
    * employ: current employment situation of speaker
    * age: age of speaker

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

According to the paper, utterances are processed in three different algorithms to parse speaker turns into utterances: Audiophile, Cliffhanger, and Backbiter. Please refer back to the paper for more detailed description on how the three algorithms are implemented.

- Audiophile: A turn is when one speaker starts talking until the other speaker starts speaking
- Cliffhanger: A turns is one full sentence said by one speaker based on terminal punctuation marks (periods, question marks, and exclamation points).
- Backbiter: A turn is what one speaker starts talking until the other speaker speaks a non-backchannel words (example backchannel words: "mhm", "yeah", "exactly", etc.)

You can pick the transcript processing algorithms in the ConvoKit conversion code by changing the TRANSCRIPTION_TYPE variable.  Note that, for different algorithms used to process utterances in transcripts, Utterance-level metadata will be different.

For each utterance we provide:

* id: Unique identifier for an utterance.
* conversation_id: Utterance id corresponding to the first utterance of the conversation.
* reply_to: Utterance id of the previous utterance in the conversation.
* speaker: Speaker object corresponding to the author of this utterance.
* text: Textual content of the utterance.

Metadata for each utterance include:

    * turn_id: The id of the turn in the current conversation.
    * speaker: Speaker id of the speaker of this turn.
    * start: The time that the turn starts in the conversation (in seconds).
    * stop: The time that the turn ends in the conversation (in seconds).
    * backchannel: The text of any backchannels that occur during this conversational turn. (For "backbiter" transcription type only)
    * backchannel_count: The number of backchannel instances (as defined in the paper) that occur during this conversational turn. Backchannel instances can be multiple tokens. (Method "backbiter" only)
    * backchannel_speaker: The user_id of the person backchanneling.  (For "backbiter" transcription type only)
    * backchannel_start: The start time of the first backchannel during this turn.  (For "backbiter" transcription type only)
    * backchannel_stop: The end time of the last backchannel during this turn.  (For "backbiter" transcription type only)
    * interval: The time between the end of the last turn and the start of this turn in seconds. Can be negative if turns overlap.
    * delta: The length of the turn (i.e., stop-start) in seconds.
    * questions: The number of question marks that appear in the utterance.
    * end_question: Indicates if the utterance ends with a question mark.
    * overlap: Indicates if interval is negative.
    * n_words: The number of words in the utterance.

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation metadata contains surveys from each participants organized by survey field names, and the values being speakers' answer organized by speaker ids:

For each conversation we provide:

* id: id of the conversation

Metadata for each conversation correspond to the answer the two speakers gave in the surveys before and after that conversation.
For each conversation, we got 1 survey from each conversation participant, and as this conversation is 2 people video calling, we got 2 surveys per conversation. We decided to organize the metadata in the following way:

convo.meta = {"survey field name" : {speaker_id_x : answer by speaker id speaker_id_x, speaker_id_y : answer by speaker id speaker_id_y} ... }

    * i_like_you: How much did you like your conversation partner? 
        * convo.meta['i_like_you'] = {speaker_id_x : answer by speaker id speaker_id_x, speaker_id_y : answer by speaker id speaker_id_y}
    * you_like_me: How much do think your conversation partner liked you?
    * i_am_funny: How funny were you in the conversation you just had?
    * you_are_funny: How funny was your conversation partner?
    * i_am_polite: How polite were you during the conversation?
    * you_are_polite: How polite was your conversation partner?
    * my_isolation_pre_covid: Prior to the Covid-19 outbreak, how socially isolated did you feel?
    * my_isolation_post_covid: SINCE the Covid-19 outbreak, how socially isolated have you felt?
    * in_common: How much did you and your partner have in common with one another?
    * about 200 other survey fileds detailed in the `BetterUp CANDOR Corpus Data Dictionary <https://docs.google.com/spreadsheets/d/1ADoaajRsw63WpM3zS2xyGC1YS5WM_IuhFZ94W84DDls/edit#gid=997152539/>`_ 


Statistics about the dataset
------------------------------

* Number of Speakers: 1454
* Number of Utterances: 527869 (if TRANSCRIPTION_TYPE = "cliffhanger")
* Number of Conversations: 1650

Additional note
---------------
Data License
^^^^^^^^^^^^

ConvoKit is not distributing the corpus separately, and thus no additional data license is applicable.  The license of the original distribution applies.

Contact
^^^^^^^

Questions about the conversion into ConvoKit format should be directed to Sean Zhang <kz88@cornell.edu>

Questions about the CANDOR corpus should be directed to the corresponding authors <andrew.reece@betterup.com(A.R.);guscooney@gmail.com(G.C.)> of the original paper.
