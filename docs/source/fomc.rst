Federal Open Market Committee (FOMC) Corpus
===========================================

Transcripts of recurring meetings of the Federal Reserve’s Open Market Committee (FOMC), where important aspects of U.S. monetary policy are decided, covering the period 1977-2008. (108,504 conversational exchanges between 364 speakers of FOMC board members in 268 meetings).

Distributed together with:
`Talk it up or play it down? (Un)expected correlations between (de-)emphasis and recurrence of discussion points in consequential U.S. economic policy meetings <https://chenhaot.com/papers/de-emphasis-fomc.html>`_. Chenhao Tan and Lillian Lee. Presented in Text As Data 2016.

Please cite this paper when using this corpus in your research.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are FOMC members, indexed by their name as recorded in the transcripts.
    * id: name of the speaker
    * chair: (boolean) is speaker FOMC Chair
    * vice_chair: (boolean) is speaker FOMC Vice-Chair

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:
    * id: index of the utterance (concatenating the meeting date with the utterance’s sequence position)
    * speaker: the speaker who authored the utterance
    * conversation_id: ID of meeting
    * reply_to: id of the sequentially prior utterance (None for the first utterance of a meeting)
    * text: textual content of the utterance
    * timestamp: calculated value based off the date of the meeting and the speech index

Metadata for utterances include:
    * speech_index: index of utterance in the context of the conversation
    * parsed: parsed version of the utterance text, represented as a SpaCy Doc

Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations are indexed by a string representing the meeting date.

Usage
-----------

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("fomc-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 364
Number of Utterances: 108504
Number of Conversations: 268

Additionally, if you want to process the original FOMC data into ConvoKit format you can use the following script `Converting FOMC Corpus to ConvoKit Format <https://github.com/CornellNLP/ConvoKit/blob/master/examples/dataset-examples/FOMC/fomc_to_convokit.ipynb>`_

Additional note
---------------

The original dataset can be downloaded `here <https://chenhaot.com/pages/de-emphasis-fomc.html>`_. Refer to the original README for more explanations on dataset construction.

Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil).
