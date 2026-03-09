Emotional Support Conversation Corpus
===============================================

A dataset of approximately 1,300 conversations between emotional support seekers and supporters, annotated with strategy labels, emotion types, and survey scores. The dataset explores the emotional support task, which has applications in areas like mental health support and customer service chats.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation involves exactly two roles: a seeker and a supporter. Speakers are bound to their respective conversation and identified accordingly. Speaker metadata include:

* role: either ``seeker`` (the person sharing their problem) or ``supporter`` (the person providing emotional support)
* dialog_index: index of the conversation this speaker is associated with


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to one turn in a conversation dialog. For each utterance, we provide:

* id: unique utterance identifier, formatted as ``utterance_{conversation_id}_{turn_index}``
* speaker: the speaker who authored the utterance
* conversation_id: ID of the conversation this utterance belongs to
* reply_to: ID of the previous utterance (None if the utterance is not a reply)
* timestamp: not provided in the original dataset
* text: textual content of the utterance

Metadata for each utterance include:

* annotation: researcher-provided annotation for the utterance, including strategy labels (e.g., ``Question``, ``Restatement or Paraphrasing``, ``Emotional Support``) for supporter turns and optional feedback scores for seeker turns


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation corresponds to a single support session. Metadata associated with conversations include:

* experience_type: the type of personal experience described (e.g., ``Previous Experience``)
* emotion_type: the primary emotion type expressed by the seeker (e.g., ``anxiety``, ``depression``)
* problem_type: the category of problem discussed (e.g., ``job crisis``, ``family``)
* situation: a brief description of the seeker's situation
* survey_score: post-conversation survey scores from both parties, including initial and final emotion intensity, empathy, and relevance ratings
* seeker_question1: open-ended post-conversation response from the seeker (question 1)
* seeker_question2: open-ended post-conversation response from the seeker (question 2)
* supporter_question1: open-ended post-conversation response from the supporter (question 1)
* supporter_question2: open-ended post-conversation response from the supporter (question 2)


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("emotional-support"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 2600
Number of Utterances: 38365
Number of Conversations: 1300


Additional notes
----------------

Data License
^^^^^^^^^^^^

This dataset is shared under the `Creative Commons Attribution-NonCommercial 4.0 International License <https://creativecommons.org/licenses/by-nc/4.0/>`_.

Dataset Access
^^^^^^^^^^^^^^

The original dataset is available `here <https://github.com/thu-coai/Emotional-Support-Conversation>`_.
