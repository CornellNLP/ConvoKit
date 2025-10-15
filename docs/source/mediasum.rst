MediaSum Corpus
===============

A collection of Interview transcripts from CNN / NPR.

Across the 10 seasons there are 463,596 conversations with ~49.4K
NPR transcripts and ~414.2K CNN transcripts. There is a total of 13,919,244 utterances, and 718,483 speakers.

Across 600 pairs of utterances, there are 5,581 annotations done by 112 annotators on whether the second utterance is a paraphrase of the first.

The original dataset is available `here <https://drive.google.com/file/d/1ZAKZM1cGhEw2A4_n4bGGMYyF8iPjLZni/view?usp=sharing>`_. It was originally distributed with `MediaSum: A Large-scale Media Interview Dataset for Dialogue Summarization <https://github.com/zcgzcgzcg1/MediaSum/>`_, Chenguang Zhu, Yang Liu, Jie Mei, Michael Zeng. Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL'21, 2021.

The original annotation of paraphrases is available `here <https://huggingface.co/datasets/AnnaWegmann/Paraphrases-in-Interviews>`_. It was originally distributed with `What's Mine becomes Yours: Detecting Context-Dependent Paraphrases in News Interview Dialogs <https://github.com/nlpsoc/Paraphrases-in-News-Interviews/tree/main>`_, Anna Wegmann, Tijs A. van den Broek, Dong Nguyen, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP'24, 2024.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are participants in an interview. This could be interview hosts or guests. The original dataset provides each speaker's name as a string, e.g. "ED LAVANDERA, CNN CORRESPONDENT". We index Speakers by these strings.

Note: In the speaker list, authors sometimes have non-unique identifiers (e.g., ‘ED LAVANDERA, CNN CORRESPONDENT’, ‘LAVANDERA’ or ‘E. LAVANDERA’ refer to the same speaker). Further, each identifier that is the same in one conversation as in another is considered the same speaker. This might be incorrect for cases like 'UNIDENTIFIED MALE' or 'UNIDENTIFIED FEMALE' that are sometimes used in interviews.


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each Utterance we provide:

- id: ``<str>``, the index of the utterance in the format `BROADCASTER-CONVONBR-UTTNBR`, where *BROADCASTER* is NPR or CNN, *CONVONBR* is the conversation number, *UTTNBR* is the utterance number, (e.g. *CNN-177596-7* or *NPR-4-1*).
- conversation_id: ``<str>``, conversation_id: id in the format `BROADCASTER-CONVONBR` (e.g. *CNN-177596* or *NPR-4*). This corresponds to the original ids in the MediaSum dataset.
- speaker: ``<Speaker>``, the speaker object who authored the utterance, name available via .speaker.id
- reply_to: ``<str>``, the id of the utterance to which this utterance replies to. `None` if the utterance is the first in a conversation.
- timestamp: ``None``. Our dataset does not contain timestamp information for utterances.
- text: ``<str>``, the textual content of the utterance.
- meta: ``<dict>``, a dictionary containing additional metadata about the utterance. See below for details.
- vectors: empty list. No precomputed vectors are provided.

We additionally provide metadata for paraphrase annotations in some 2-person interviews. For 600 utterance pairs (u, v), there are annotations on whether v contains a paraphrase of u. Annotations contains paraphrase spans selected by Prolific annotators. v is always the interview host. The annotations include: character entities (or who is referred to in the utterance), emotion, a tokenized version of the text, caption information, and notes about the transcript, which we describe as follows:

- paraphrase_is_host: ``<bool>``, whether the utterance is spoken by the host of the interview. This also tells us whether the utterance is the second in a pair of utterances annotated for paraphrases.
- paraphrase_number_votes ``<int>``, the number of annotators who annotated this utterance pair.
- paraphrase_votes ``<int>``, the number of annotators who indicated that the second utterance contains a paraphrase of the first.
- paraphrase_ratio ``<float>``, the ratio of paraphrase votes to total votes.
- paraphrase_guest_highlights ``list <[float]>``, a list based on the tokens that can be created from utterance.text.split(). Each entry is between 0 and 1, indicating the ratio of annotators that highlighted that token as part of a paraphrase for the next utterance.
- paraphrase_host_highlights ``list <[float]>``, a list based on the tokens that can be created from utterance.text.split(). Each entry is between 0 and 1, indicating the ratio of annotators that highlighted that token as part of a paraphrase of the previous utterance.
- paraphrase_PROLIFIC_X ``list <int>``, for X in [1, 2, ..., 112], a list of length utterance.text.split() indicating whether annotator X highlighted token i as part of a paraphrase (1) or not (0). If annotator X did not annotate this key does not exist in the dict.

Example of utterance "CNN-177596-7":


>>> print(media_sum_corpus.get_utterance("CNN-177596-7"))
    Utterance(id: 'CNN-177596-7', conversation_id: CNN-177596, reply-to: CNN-177596-6,
        speaker: Speaker(id: 'JOHNS', vectors: [], meta: ConvoKitMeta({'name': 'JOHNS'})),
        timestamp: None, text: 'This is not good.', vectors: [],
        meta: ConvoKitMeta({
            'paraphrase_guest_highlights': [0.5, 0.45, 0.45, 0.45], 'paraphrase_is_host': False,
            'paraphrase_number_votes': 20, 'paraphrase_votes': 10, 'paraphrase_ratio': 0.5,
            'paraphrase_PROLIFIC_1': [0, 0, 0, 0], 'paraphrase_PROLIFIC_2': [1, 1, 1, 1], 'paraphrase_PROLIFIC_3': [0, 0, 0, 0], 'paraphrase_PROLIFIC_4': [0, 0, 0, 0], 'paraphrase_PROLIFIC_5': [0, 0, 0, 0], 'paraphrase_PROLIFIC_6': [1, 1, 1, 1], 'paraphrase_PROLIFIC_7': [1, 0, 0, 0], 'paraphrase_PROLIFIC_8': [1, 1, 1, 1], 'paraphrase_PROLIFIC_9': [0, 0, 0, 0], 'paraphrase_PROLIFIC_10': [0, 0, 0, 0], 'paraphrase_PROLIFIC_11': [1, 1, 1, 1], 'paraphrase_PROLIFIC_12': [0, 0, 0, 0], 'paraphrase_PROLIFIC_13': [1, 1, 1, 1], 'paraphrase_PROLIFIC_14': [0, 0, 0, 0], 'paraphrase_PROLIFIC_15': [0, 0, 0, 0], 'paraphrase_PROLIFIC_16': [1, 1, 1, 1], 'paraphrase_PROLIFIC_17': [0, 0, 0, 0], 'paraphrase_PROLIFIC_18': [1, 1, 1, 1], 'paraphrase_PROLIFIC_19': [1, 1, 1, 1], 'paraphrase_PROLIFIC_20': [1, 1, 1, 1]
        }))

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Conversations represent interviews on NPR and CNN. They are indexed by the id conversation_id: id in the format `BROADCASTER-CONVONBR` (e.g. *CNN-177596* or *NPR-4*). This corresponds to the original ids in the MediaSum dataset.

- program: ``<str>``, the name of the program that the interview is a part of, e.g., 'CNN SATURDAY NIGHT'.
- date: ``<str>``, the date the interview aired, in the format 'YYYY-MM-DD' or 'YYYY-M-DD', e.g., '2003-2-22' or '2007-11-28'.
- summary: ``<str>``, a summary of the interview or the topic of the interview. The level of detail varies, e.g., 'How Much Will War With Iraq Cost?' in "CNN-67148" and 'More than 400 black actors, artists and ministers are bringing the Gospel to life in the audio book, The Bible Experience:The Complete Bible. Farai Chideya talks with producer Kyle Bowser and actress Wendy Raquel Robinson, who lends her voice to the project.' in "NPR-1".
- url: ``<str>``, the URL of the interview transcript on the broadcaster's website, e.g., 'http://transcripts.cnn.com/TRANSCRIPTS/0302/22/stn.02.html' for "CNN-67148".
- title: ``<str>``, the title of the interview, e.g., 'Black Actors Give Bible Star Appeal' in "CNN-67148".
- broadcaster: ``<str>``, the broadcaster of the interview, either 'CNN' or 'NPR'.

Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- name ``<str>``, the name of the corpus, 'media-sum-corpus'.
- paraphrase_pairs ``list<list<list<str>>>``, a list of lists containing the pairs of utterance ids (u, v) that are annotated for paraphrases. There are 600 such pairs in total. An entry in the list of paraphrase pairs can look like this , [['NPR-35922-5', 'NPR-35922-6', 'NPR-35922-7'], ['NPR-35922-8']], i.e., an "utterance" for the annotations can consist of multiple utterances according to the corpus utterance ids. This happens if the same speaker speaks multiple times in a row.
- paraphrase_labels ``list<float>``, a list of floats between 0 and 1 indicating whether the second utterance in each paraphrase pair contains a paraphrase of the first (1) or not (0). The order corresponds to the order of paraphrase_pairs. Values are floats because they represent the ratio of annotators who indicated that the second utterance contains a paraphrase of the first.



Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("mediasum-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 700
Number of Utterances: 67373
Number of Conversations: 3107

Get all paraphrase pairs from the corpus metadata

>>> paraphrase_pairs = corpus.meta['paraphrase_pairs']
>>> print(f"Total paraphrase pairs: {len(paraphrase_pairs)}")
Total paraphrase pairs: 600

useful functions for working with paraphrase pairs

.. code-block:: python

    from itertools import chain

    def get_paraphrase_pair_info(corpus, pair_id):
        """Get text, paraphrase ratio, and highlighting for a paraphrase pair."""
        pairs = corpus.meta['paraphrase_pairs']
        labels = corpus.meta['paraphrase_labels']

        pair = pairs[pair_id]
        group1_text = " ".join([corpus.get_utterance(uid).text for uid in pair[0]])
        group2_text = " ".join([corpus.get_utterance(uid).text for uid in pair[1]])

        # Get highlighting from all utterances in each group
        group1_highlights = list(chain.from_iterable(corpus.get_utterance(uid).meta['paraphrase_guest_highlights'] for uid in pair[0]))
        group2_highlights = list(chain.from_iterable(corpus.get_utterance(uid).meta['paraphrase_host_highlights'] for uid in pair[1]))

        return {
            'pair_id': pairs[pair_id],
            'text1': group1_text,
            'text2': group2_text,
            'paraphrase_ratio': corpus.meta["paraphrase_labels"][pair_id],
            'is_paraphrase': corpus.meta["paraphrase_labels"][pair_id] >= 0.5,
            'guest_highlights': group1_highlights,
            'host_highlights': group2_highlights,
        }
    def print_highlighted_pair(pair_info):
        """Print paraphrase pair with token-level highlighting -- upper casing if >= 0.5 and emphasis if >= 0.4"""

        def highlight_text(text, highlights):
            tokens = text.split()
            return " ".join(
                token.upper() if score >= 0.5
                else f"\033[1m{token}\033[0m" if score >= 0.4
                else token
                for token, score in zip(tokens, highlights)
            )

        print(f"=== Pair {pair_info['pair_id']} ===")
        print(f"Paraphrase ratio: {pair_info['paraphrase_ratio']:.3f} ({'PARAPHRASE' if pair_info['is_paraphrase'] else 'NOT PARAPHRASE'})")
        print(f"\nGuest:\n{highlight_text(pair_info['text1'], pair_info['guest_highlights'])}")
        print(f"\nHost:\n{highlight_text(pair_info['text2'], pair_info['host_highlights'])}\n")

Example use:

.. code-block:: python

    >>> print_highlighted_pair(get_paraphrase_pair_info(media_sum_corpus, 9))
    === Pair [['CNN-350238-7'], ['CNN-350238-8']] ===
    Paraphrase ratio: 0.500 (PARAPHRASE)

    Guest:
    I want to applaud THE WORK OF THE TEXAS RANGERS and the sheriff's office and DPS in bringing this man into custody.

    Host:
    Do you think -- last quick question -- no, it's been EXTRAORDINARY WORK for you guys in Texas. Had this most recent woman not escaped, what are her chances that she could have been next?


Additional note
---------------

Data License
^^^^^^^^^^^^

Research-only

Details:
Regarding license, the `Mediasum paper <https://aclanthology.org/2021.naacl-main.474.pdf>`_ reads: "We have used only the publicly available transcripts data from the media sources and adhere to their only-for-research-purpose guideline." and "Please restrict your usage of this dataset to research purpose only" on `their GitHub <https://github.com/zcgzcgzcg1/MediaSum>`_. The annotations are also shared with a research-only license, see `GitHub <https://github.com/nlpsoc/Paraphrases-in-News-Interviews>`_.


Contact
^^^^^^^

Please email any questions to Anna Wegmann (a.m.wegmann@uu.nl).
