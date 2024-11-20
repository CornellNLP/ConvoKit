NPR Interview 2P Dataset Corpus
===============================

This corpus contains conversations between NPR show hosts and their guests. The corpus contains dialog from 22,257 speakers with 428,624 utterances and 22,149 conversations total.

This is a Convokit-formatted version of the dataset originally distributed with the following paper:

Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, and Julian McAuley. 2020. `Interview: Large-Scale Modeling of Media Dialog with Discourse Patterns and Knowledge Grounding. <https://www.aclweb.org/anthology/2020.emnlp-main.653>`_ In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 8129–41.

Please cite this paper when using this corpus in your research.

Dataset Details
---------------

Speaker-Level Information
^^^^^^^^^^^^^^^^^^^^^^^^^

In this dataset, each speaker is either a show host or guest. The speaker index is the same as the index given in the original dataset, and the following metadata is also provided:
    * name: the speaker’s name as given in the original dataset
    * type: host or guest, depending on the speaker’s role

Utterance-Level Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following information about each utterance is provided:
    * id: the index of the utterance in the dataset
    * speaker: the speaker who said the utterance
    * reply_to: the id of the utterance which this utterance replies to, or None if none exists.
    * timestamp: null for the entirety of this corpus
    * text: the text of each utterance
    * episode: the id of the episode this utterance appears in
    * order: the index of this utterance within the episode

Conversational-Level Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversations are indexed by the id of the first utterance that appears in the conversation. The follow information about each utterance is provided:
    * program: the name of the NPR radio program this episode appears in
    * title: the title of this episode
    * date: the date this episode aired

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("npr-2p-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 22267
Number of Utterances: 428624
Number of Conversations: 22149

Additionally, if you want to process the original NPR-2P data into ConvoKit format you can use the following script `Converting NPR-2P Corpus to ConvoKit Format <https://github.com/CornellNLP/ConvoKit/blob/master/examples/dataset-examples/NPR-2P/npr_to_convokit.ipynb>`_

Additional note
---------------

Contact
^^^^^^^

Please email any questions to Andrea (aww66@cornell.edu), Lucy (lj287@cornell.edu), or Rebecca (rmh327@cornell.edu).

Files
^^^^^^^

The original dataset can be found on `Kaggle <https://www.kaggle.com/datasets/shuyangli94/interview-npr-media-dialog-transcripts?select=utterances-2sp.csv>`_

Dataset `Access <https://drive.google.com/file/d/1Yle2eq0VFPXCmKGaeht5bSZujasVmdV_/view?usp=sharing>`_

Cleaning/Conversion `Script <https://drive.google.com/file/d/1O8WWYJ6iHSiW7II2yqa3mxi3XO-B0zxf/view?usp=sharing>`_
