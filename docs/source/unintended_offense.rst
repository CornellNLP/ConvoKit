Unintended Offense Corpus
=========================
The Unintended Offense Corpus is a collection of unintentionally offensive Tweets and replies in which a Tweet in the exchange was offensive to someone, followed by an indication that the poster meant no offense. The data were collected starting with the "cue" post (e.g., "didn't mean to offend") then working backwards to identify the original unintentionally offensive post. ConvoKit contains code for converting the corpus into ConvoKit format, as detailed below.

A full description of the dataset can be found here: `C. W. Tsai, Y.-H. Huang, T.-K. Liao, D. F. S. Estrada, R. Latifah, and Y.-S. Chen, "Leveraging Conflicts in Social Media Posts: Unintended Offense Dataset," Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024. <https://aclanthology.org/2024.emnlp-main.259>`_
Please cite this paper when using the Unintended Offense Corpus in your research.

Usage
-----

Download the original data from: `https://github.com/IDEA-NTHU-Taiwan/unintended-offense-tweets <https://github.com/IDEA-NTHU-Taiwan/unintended-offense-tweets>`_

Convert the Unintended Offense Corpus into ConvoKit format using `the files in this directory <https://github.com/CornellNLP/ConvoKit/blob/master/examples/dataset-examples/news-interview/conversion.ipynb>`_.

Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^

Speakers in this dataset are Twitter users. Each speaker has an associated identifier sourced from the original dataset.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each utterance corresponds to a single Tweet. For each utterance, we provide:

* id: unique utterance identifier, comprised of the conversation ID concatenated with its index in the conversation
* conversation_id: identifier for the conversation this utterance belongs to
* reply_to: ID of the utterance this utterance replies to (None if the utterance is not a reply)
* speaker: the Twitter user who authored the utterance
* timestamp: Unix timestamp of when the Tweet was posted
* text: textual content of the Tweet

Metadata for each utterance include:

    * tweet_type: the "type" of Tweet within the conversation chain

        - ``Context``: initial Tweet(s) that give context for the subsequent interaction
        - ``Target``: the Tweet identified as offensive by someone else
        - ``Follow-up``: the Tweet that appears between the target post and the cue post
        - ``Cue``: the Tweet that signals acknowledgment of unintentional offense (e.g., "I didn't mean to offend you")

    * index: the sequential position of the Tweet within the conversation (e.g., the sole or preliminary context Tweet will always be at index 0)
    * device: the device used to access Twitter and post the Tweet
    * reply_settings: who can reply to the Tweet, based on the poster's account settings
    * retweet_count: number of retweets gained by the Tweet (at time of data extraction)
    * reply_count: number of replies gained by the Tweet (at time of data extraction)
    * like_count: number of likes gained by the Tweet (at time of data extraction)
    * quote_count: number of quotes gained by the Tweet (at time of data extraction)
    * offensiveness: average human-annotated score of perceived offensiveness (0–100) for the target Tweet
    * confidence: average human-annotated confidence in the perceived offensiveness score

Conversation-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each conversation follows the chain structure: context → target → follow-up → cue. By default, the first context Tweet is a reply to no one; any subsequent context Tweets reply to the previous context Tweet. Target Tweets are set as replies to the last context Tweet, follow-up Tweets are replies to the target Tweet, and cue Tweets are replies to the follow-up Tweet.


Statistics about the dataset
------------------------------

* Number of Speakers: 7944
* Number of Utterances: 16423
* Number of Conversations: 4014

Additional notes
----------------

The original dataset is available `here <https://github.com/IDEA-NTHU-Taiwan/unintended-offense-tweets>`_. Corresponding author: Yi-Shin Chen (yishin@gmail.com).

Data License
^^^^^^^^^^^^

ConvoKit is not distributing this corpus directly, so no additional data license is applicable. The license of the original distribution applies.

Dataset Limitations
^^^^^^^^^^^^^^^^^^^

All engagement metadata (likes, replies, retweets, and quotes) was sourced from the original JSON file, which sometimes reflects no replies in the metadata while at least one reply is implied in the conversation chain. Exercise caution when making inferences based on engagement metrics.

Contact
^^^^^^^

This dataset was formatted for ConvoKit by Eden Shaveet. Contact: ems486@cs.cornell.edu.
