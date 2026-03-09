Parliament Question Time Corpus
===============================

A collections of questions and answers from parliamentary question periods in the British House of Commons from May 1979 to December 2016 (433,787 utterances), scraped from `They Work For You <https://www.theyworkforyou.com/`>_.

Distributed together with:
`Asking Too Much? The Rhetorical Role of Questions in Political Discourse <https://www.cs.cornell.edu/~cristian/Asking_too_much.html>`_. Justine Zhang, Arthur Spirling, Cristian Danescu-Niculescu-Mizil. EMNLP 2017.


Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

The speakers in the dataset are members of the Parliament (MP). For each MP, the dataset further includes the following metadata:

* name: name of the MP
* member_start: start date of the MP as the member of the Parliament
* member_end: end date of the MP as the member of the Parliament (set to year 3020 if the MP was still in Parliament as of Dec 2016)
* parties: a list of parties that the MP has belonged to in the past
* first_govt: first government (by Prime Minister) in which the MP was in office
* first_govt_coarse: first government in which the MP was in office. here, consecutive governments of the same party (e.g., thatcher+major) are grouped together.

Note that some of the metadata information may be missing, especially for MPs active before the Blair government.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each question or answer is viewed as an utterance. For each utterance, we provide:

* id: index of the utterance
* speaker: the MP who spoke the utterance
* conversation_id: id of the first utterance in the conversation this utterance belongs to
* reply_to: id of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Additional metadata include:

* next_id: id of the utterance replying to this one (None if the utterance has no reply)
* is_question: whether the utterance is a question
* is_answer: whether the utterance is an answer to a question
* pair_idx: index of the question-answer pair
* is_incumbent: whether the MP is incumbent (i.e., a member of the government party)
* is_minister: whether the MP is a Minister
* is_oppn: whether the MP is from the official opposition party
* party: party affiliation of the MP
* tenure: the number of years that the MP has been in office at the time of the utterance
* govt: current government (by Prime Minister) at the time of the utterance
* govt_coarse: current government (by Prime Minister) at the time of the utterance. here, consecutive governments of the same party (e.g., thatcher+major) are grouped together.
* pair_has_features: whether the pair to which the utterance belongs has a question that contains at least one `q_arc` term and an answer that contains at least one `arcs` term.
* dept_name: the name of the department to which the answering minister for the question or answer belongs. inferred from the raw HTML (see dept_name_raw for an un-processed version of the same attribute).
* dept_name_coarse: department name, listed as `other` for departments other than the 10 containing the most question-answer pairs.

Note that some of the metadata information may be missing, especially for utterances dating back to before the Blair government.

The dataset also comes with the following processed fields, which can be loaded separately via `corpus.load_info('utterance',[list of fields])`:

* parsed: SpaCy dependency parse
* arcs: dependency parse arcs, without nouns
* q_arcs: dependency parse arcs for questions only, without nouns

The latter two fields are used in the original publication to represent utterances.






Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("parliament-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 1978
Number of Utterances: 433787
Number of Conversations: 216894


Additional note
---------------

See this `example notebook <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/parliament_demo.ipynb>`_ for an example of how to group questions in this dataset according to their rhetorical roles.
