Group Affect and Performance (GAP) Corpus
=========================================

The Group Affect and Performance (GAP) Corpus was collected at University of the Fraser Valley (UFV, Canada).
The original dataset comprises 28 group meetings of two to four group members, with a total of 84 participants.
All the recorded conversations are in English.

Group members completed a Winter Survival Task (WST), a group decision-making exercise where participants must rank 15 items according to their importance in a hypothetical plane crash scenario.
Participants first rank the items individually.
Then, each group was given a maximum of 15 minutes to complete the WST.
The group's conversations and deliberations during this task were recorded as conversations in this dataset.

A full description of the dataset can be found here:

`M. Braley and G. Murray, “The Group Affect and Performance (GAP) corpus,” in Proceedings of the Group Interaction Frontiers in Technology. ACM, 2018, p. 2. <https://www.ufv.ca/media/assets/computer-information-systems/gabriel-murray/publications/main-2.pdf>`_


Please cite this paper when using it in your research:

| @inproceedings{gapcorpus18,
|   title={The Group Affect and Performance (GAP) Corpus},
|   author={Braley, McKenzie and Murray, Gabriel},
|   booktitle={Proceedings of the ICMI 2018 Workshop on Group Interaction Frontiers in Technology (GIFT)},
|   location={Boulder, CO},
|   year={2018},
| }

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Meeting and Group Member Identification (ID) Codes: For all corpus components, meetings and group members are identified with a code.
Meetings are coded with a number based on the temporal order of data collection.
Group members are coded with one of five arbitrarily assigned colors (i.e., blue, green, pink, orange, yellow).

At the end of each group meeting, group members filled out a post-task questionnaire.
In addition to providing basic demographic information, they also responded on a five-point Likert scale (to how strongly they agreed with statements concerning the meeting:

1. Time Expectations: "This task took longer than expected to complete."
2. Worked Well Together: "Our group worked well together."
3. Time Management "Our group used its time wisely."
4. Efficiency: "Our group struggled to work together efficiently on this task."
5. Overall Quality of Work: "Overall, our group did a good job on this task."
6. Leadership: "I helped lead the group during this task."

The speaker-level metadata includes:

- Year at UFV
- Gender
- English: whether English is the native language
- AIS: Absolute Individual Score: calculated by summing the differences between the individual ranking and the expert ranking for
each item
- AII: Absolute Individual Influence
- Ind_TE: Time Expectations
- Ind_WW: Worked Well Together
- Ind_TM: Time Management
- Ind_Eff: Efficiency
- Ind_QW: Overall Quality of Work
- Ind_Sat: Overall Satisfaction: a combined score calculated by combining and averaging answers to (1) - (5)
- Ind_Lead: Leadership
- Group Number: number based on the temporal order of data collection


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the GAP corpus, speech was transcribed verbatim, including grammatical errors, false starts, stutters, and filled pauses (e.g. "hmm").

There are also symbols used to represent non-speech information:

- "$" means laughter,
- "%" represents coughing,
- "#" indicates another noise.

Each utterance from the GAP corpus possesses the following information, which are aligned with the Utterance schema from ConvoKit:

- id: unique speaker utterance, e.g. 1.Green.70
- speaker: speaker name with group number, e.g. 1.Green
- conversation_id: id of the first utterance in the conversation this utterance belongs to, e.g. 1.Pink.1
- reply_to: previous idx, e.g. 1.Blue.105
- timestamp: start time in format MM:SS.ds, where ds is deciseconds (only ever one decimal value), e.g. 02:10.5
- text: sentence of utterance, without punctuation

Additional metadata includes:

- Duration: in seconds and deciseconds
- Sentiment: whether the sentence bears any positive or negative sentiment
- Decision: denotes a group-decision process -- possible values include Proposal, Acceptance, Rejection, and Confirmation
- Private: if the speaker is refering to a private item
- Survival Item: what survival item was mentioned

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation has the following associated metadata:

- Group Number
- Meeting Size: varies from 2-4
- Meeting Length in Minutes: maximum meeting length is 15 minutes
- AGS: Absolute Group Score: calculated by summing the differences between the group ranking and the expert ranking for each item; a lower score reveals greater similarity to the expert and thus, greater decision making performance
- Group_TE: Time Expectations
- Group_WW: Worked Well Together
- Group_TM: Time Management
- Group_Eff: Efficiency
- Group_QW: Overall Quality of Work
- Group_Sat: Overall Satisfaction

The last six items were assessed at an individual-level (see "Speaker-level information") and at a group-level.
To derive the group-level scores, we calculated the average of the group member responses in each meeting.


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> gap_corpus = Corpus(filename=download("gap-corpus"))


For some quick stats:

>>> gap_corpus.print_summary_stats()
Number of Speakers: 84
Number of Utterances: 8009
Number of Conversations: 28


Additional note
---------------

Data License
^^^^^^^^^^^^

The GAP corpus is released under the `Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license <https://creativecommons.org/licenses/by-nc/4.0/>`_.

Contact
^^^^^^^

Please email any questions to the contributor of this dataset: Uliyana Kubasova (uliyana.kubasova@student.ufv.ca)
