CaSiNo Corpus
=============
CaSiNo (stands for CampSite Negotiations) is a novel dataset of 1030 negotiation dialogues. Two participants take the role of campsite neighbors and negotiate for Food, Water, and Firewood packages, based on their individual preferences and requirements. This design keeps the task tractable, while still facilitating linguistically rich and personal conversations. This helps to overcome the limitations of prior negotiation datasets such as Deal or No Deal and Craigslist Bargain. Each dialogue consists of rich meta-data including participant demographics, personality, and their subjective evaluation of the negotiation in terms of satisfaction and opponent likeness.

A full description of the dataset can be found here:
`Kushal Chawla, Jaysa Ramirez, Rene Clever, Gale Lucas, Jonathan May, and Jonathan Gratch. 2021. CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3167–3185, Online. Association for Computational Linguistics. <https://aclanthology.org/2021.naacl-main.254/>`_

Please cite this paper when using it in your research:

| @inproceedings{chawla2021casino,
| 	title={CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems},
| 	author={Chawla, Kushal and Ramirez, Jaysa and Clever, Rene and Lucas, Gale and May, Jonathan and Gratch, Jonathan},
| 	booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
| 	pages={3167–3185},
| 	year={2021}}

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers are essentially Turkers on the Amazon Mechanical Turk platform who engaged in a negotiation with a randomly paired-up partner via a chat interface. In total, there were 846 unique participants.

For each speaker, the following additional information is provided as metadata:

* demographics: Demographic information of the speaker collected in a pre-survey (based on self-report).
	 * age: Age of the speaker (numeric).
	 * gender: Gender of the speaker.
	 * ethnicity: Ethnicity of the speaker.
	 * education: The highest level of education of the speaker.
* personality: Personality attributes collected in the pre-survey based on standard tests from the psychology literature.
   * svo: Social Value Orientation of the speaker (Prosocial vs Proself).
   * big-five: a dictionary of scores (ranging from 1 to 7) for each of the big-five personality traits (extraversion, agreeableness, conscientiousness, emotional-stability, openness-to-experiences).


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a total of 14,297 utterances in the dataset. Each utterance object includes:

* id: Unique identifier for an utterance.
* conversation_id: Utterance id corresponding to the first utterance of the conversation.
* reply_to: Utterance id of the previous utterance in the conversation.
* speaker: Speaker object corresponding to the author of this utterance.
* text: Textual content of the utterance.
* meta:
    * annotations: Comma-separated list of negotiation strategies used in the given utterance (only available for ~40% of the data).
    * speaker_internal_id: An internal id for the speaker (mturk_agent_1/mturk_agent_2).
    * speaker_id: Speaker Id for the speaker of this utterance.
    * data: (only available for some utterances) Indicates special utterances which involve acceptance or rejection of a deal.
    * issue2youget: (only available for some utterances) Valid only when a deal is submitted. Indicates the quantities that the speaker of this utterance would get, based on this submitted deal.
    * issue2theyget: (only available for some utterances) Valid only when a deal is submitted. Indicates the quantities that the other speaker would get, based on this submitted deal.


Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations are identified by their first utterance. Each conversation also has meta-information about the two participants, including their negotiation context and outcomes:

* dialogue_id: id of the dialogue (different from conversation id above)
* participant_info:
    * mturk_agent_1:
		    * value2issue: Defines the preference order for this specific speaker among Food, Water, and Firewood.
		    * value2reason: The arguments provided by the speaker herself, aligned to the above preference order.
		    * outcomes:
				    * points_scored: Objective outcome indicating the total points scored by this speaker. This is computed on the final deal that the two speakers agree on at the end of the negotiation.
				    * satisfaction: Answer to "How satisfied are you with the negotiation outcome?" (5 levels)
				    * opponent_likeness: Answer to "How much do you like your opponent?" (5 levels)
    * mturk_agent_2: Follows the same structure as above.


Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^

This includes:

* name: A descriptive name of the dataset.
* dataset_url: URL to the original Github repository that hosts the dataset.
* paper_url: URL to the NAACL 2021 paper that released the dataset.


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("casino-corpus"))


Statistics about the dataset:

* Number of unique speakers: 846
* Number of dialogues: 1030
* Number of dialogues annotated: 396
* Number of utterances: 14,297
* Avg length of an utterance: 22 tokens

Additional note
---------------
Data License
^^^^^^^^^^^^
The project is licensed under the `Creative Commons Attribution 4.0 International (CC BY 4.0) license <https://creativecommons.org/licenses/by/4.0/>`_.

Contact
^^^^^^^

Name: Kushal Chawla (Corresponding Author)

Email: kchawla@usc.edu

Affiliation: University of Southern California
