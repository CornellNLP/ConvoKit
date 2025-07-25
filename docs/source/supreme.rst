Supreme Court Oral Arguments Corpus
=======================================


A collection of cases from the U.S. Supreme Court, along with transcripts of oral arguments. Contains approximately 1,800,000 utterances over 8,300 oral arguments transcripts from 8,000 cases.

The data comes from two sources: transcripts were scraped from the `Oyez <https://www.oyez.org/>`_ website, while voting information comes from the `Supreme Court Database <http://scdb.wustl.edu/index.php>`_ (SCDB). 

Along with the entire corpus, we release another version split up into different years spanning 1955 to 2023, each named "supreme-(year)". Additional metadata are also included for each case `here <https://zissou.infosci.cornell.edu/convokit/datasets/supreme-corpus/cases.jsonl>`_. 

The following examples use this corpus: 

* `computing the orientation of utterances <https://github.com/CornellNLP/ConvoKit/tree/master/examples/orientation>`_
* `computing linguistic coordination <https://github.com/CornellNLP/ConvoKit/blob/master/examples/coordination/examples.ipynb>`_

Some considerations regarding case and voting information
---------------------------------------------------------------

Each case in the data can have multiple conversations, corresponding to multiple sessions of oral arguments heard. For convenience, we include information for each conversation about how justices voted in the  corresponding *case*, meaning that vote information will be repeated across each conversation corresponding to a case. The case metadata file also lists vote information.

The docket ID was used, along with some heuristics, to match cases in Oyez with those in SCDB. While most cases could be matched this way, a few were done manually (by inspecting case names and decision dates) and a few appear to be missing; please let us know of any mistakes you encounter. The case metadata file contains information about which case IDs, in our data, map to which docket IDs in the SCDB dataset.

SCDB makes finer distinctions about justice votes and case outcomes than whether the petitioner or respondent won. This finer-grained information is listed in the case metadata file; for the vote information included per-conversation in the corpus, we map justice vote and case outcome information to whether the vote/case was in favor of the petitioner or respondent. See below description of the case metadata file for details.


Usage
-----

To download the entire corpus:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("supreme-corpus"))

To download a particular year:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("supreme-2019"))

Dataset details
---------------


Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers correspond to justices and lawyers (also referred to as advocates). 

For each Speaker, we provide:

* id: the ID of the Speaker. If provided in Oyez, we use this ID, such that further information about advocates or justices may be found at `oyez.org/advocates/<id>` or `oyez.org/justices/<id>`. Otherwise this is inferred (see below)
* name: the name of the Speaker, as listed in transcripts.
* type: whether the speaker is a justice J, advocate A or unknown U.  

Additional details: 

* When possible, we tried to ensure Speaker information corresponds to information provided in Oyez. Oyez usually provides explicit lists of the speakers involved in each oral argument, especially for more recent cases; earlier ones are missing these explicit lists. Otherwise we tried to follow the Oyez format for converting between names listed in transcripts and IDs (i.e., replacing spaces with underscores and lowercasing).


Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations correspond to different sessions of oral arguments and re-arguments. Importantly, note that a case can have multiple conversations. 

For each Conversation, we provide:

* id: we use the ID of the corresponding transcript, as provided by Oyez.
* case_id: the ID of the case (see below).
* advocates: a dictionary where each entry lists the following information for each lawyer:
	* role: the role that the advocate plays (e.g., "Argued for the petitioner"), as listed by Oyez; "inferred" if no role is listed. 
	* side: the side that the advocate is on: 0 for respondent, 1 for petitioner, 2 for amicus curiae (NOTE that we currently do not differentiate between which side the amicus was supporting), 3 for unknown, None for unknown or inaudible speakers (see below, Utterance-level information). If no role is listed in Oyez, this is inferred via some heuristics (documentation forthcoming).
* votes_side: a dictionary where each entry lists how each justice voted in the case in which the session occurred: 1 for the petitioner and 0 for the respondent. -1 if vote information was not provided or was otherwise unclear. 
* win_side: 1 if the case (in which the session occurred) was decided favorably for the petitioner, 0 if it wasn't; 2 if the decision was unclear, and -1 if this information was unavailable.

See below description on the case metadata file for further details on votes_side and win_side.
	

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id
* text. Oyez seems to separate different sentences into different paragraphs to facilitate its audio-to-text  matching; we've retained this segmentation in the data, where sentences are separated by newline.
* speaker. Note that some utterances have "<INAUDIBLE>" speakers, corresponding to turns listed in the Oyez transcripts without any speaker information, where an interjection was audible but the identity of the speaker couldn't be discerned.
* conversation_id
* case_id: the ID of the case in which the oral argument took place.
* speaker_type: whether the speaker is a justice J, advocate A, or unknown/inaudible U.
* side: the speaker's side (see above, Conversation-level information, and note that this is sometimes inferred from the data if not explicitly listed)
* start_times: the timestamp (as listed in Oyez) of when each sentence in the text starts. There is one entry per sentence, corresponding to newlines in the text.
* stop_times: the timestamp of when each sentence ends.
* timestamp: the timestamp of the first sentence in the utterance.
* reply_to: the ID of the preceding utterance.

The dataset also comes with the following processed fields, which can be loaded separately via `corpus.load_info('utterance',[list of fields])`:

* parsed: dependency parse of each utterance
* arcs: dependency parse arcs for each utterance
* tokens: processed tokens of each utterance

.. Note that at present, each sentence of a parse contains an extra space at the end, due to how Oyez segments different sentences into paragraphs. A todo is to check  that the Oyez segmentation indeed corresponds to sentence breaks (such that the additional newlines can be safely removed).


Case information
^^^^^^^^^^^^^^^^^^^^^

`This file <https://zissou.infosci.cornell.edu/convokit/datasets/supreme-corpus/cases.jsonl>`_ is a list of json objects containing some information about each case, pulled from Oyez and SCDB. 

* id: generally formatted as <year of case>_<docket no>
* year
* title: the name of the case
* petitioner: the name of the petitioner
* respondent: the name of the respondent
* docket_no: the docket number of the case, as listed in Oyez.
* scdb_docket_id: the docket ID of the case, as listed in SCDB.
* citation: the citation of the case from the United States Reports. Note that there appear to be some missing entries and some duplicates.
* url: the url of the Oyez listing
* court: the court that saw the case (corresponding to a particular roster of justices)
* decided_date: the date the case was decided, according to Oyez
* win_side: whether the petitioning party won; also included in the corpus. See the `corresponding listing <http://scdb.wustl.edu/documentation.php?var=partyWinning>`_ in SCDB for details. -1 if no information available.
* win_side_detail: finer-grained label of case outcome. See the `corresponding listing <http://scdb.wustl.edu/documentation.php?var=caseDisposition>`_ in SCDB for details. -1 if no information available.
* advocates: the advocates participating in the case. 
* adv_sides_inferred: While most Oyez transcripts explicitly list advocates and their roles, some don't, so we fill this information in via a set of heuristics. This field is True if at least one advocate had information that was filled in in this way.
* votes: a dictionary of justice to whether they voted with the majority or dissented. See the `corresponding listing <http://scdb.wustl.edu/documentation.php?var=majority>`_ in SCDB for details. -1 if no information available. 
* votes_detail: a dictionary of justice to their vote in the case. See the `corresponding listing <http://scdb.wustl.edu/documentation.php?var=vote>`_ in SCDB for details. -1 if no information available. 
* votes_side: a dictionary of justice to whether they voted for the petitioning party, derived from the win_side and votes_detail information. -1 if no information available; in particular, note that if the vote was equally divided, we cannot infer which side the justice voted for. Also included in the corpus.
* transcripts: a list of transcript names, URLs and IDs (corresponding to the IDs of conversations in the corpus). 

Citation and other versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This corpus extends a `smaller dataset <https://confluence.cornell.edu/display/llresearch/Supreme+Court+Dialogs+Corpus>`_ of oral arguments that we previously released together with `Echoes of power\: Language effects and power differences in social interaction <https://www.cs.cornell.edu/~cristian/Echoes_of_power.html>`_. Cristian Danescu-Niculescu-Mizil, Bo Pang, Lillian Lee and Jon Kleinberg. WWW 2012.  Please cite the Echoes of Powers paper if you use either version of the corpus.  If you use the ConvoKit version 	please additionally cite: `ConvoKit\: A Toolkit for the Analysis of Conversations <https://www.cs.cornell.edu/~cristian/ConvoKit_Demo_Paper_files/convokit-demo-paper.pdf>`_. Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil. Proceedings of SIGDIAL. 2020.

This work extends the original Supreme Court Corpus curated by the ConvoKit team to include data up to the year 2023. We preserve the structure and metadata of the original release while integrating updated transcripts from 2019 where appropriate. We thank Jeeyon Kang for the help with the 2023 extension of the corpus.

**Note:** The original version of the corpus (prior to the 2023 extension) is still available for reproducibility purposes at `link <https://zissou.infosci.cornell.edu/convokit/datasets/supreme-corpus-deprecated/>`_. 


Contact
^^^^^^^

Please email any questions to: jz727@cornell.edu (Justine Zhang).
For issues with the extended version (2023 extension) of the corpus, please email: jk26@williams.edu (Jeeyon Kang).