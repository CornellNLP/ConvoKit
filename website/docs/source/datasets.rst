Datasets
========

ConvoKit ships with several datasets ready for use "out-of-the-box". These datasets can be downloaded using the ``convokit.download()`` `helper function <https://github.com/CornellNLP/ConvoKit/blob/master/convokit/util.py>`_. Alternatively you can access them directly `here <http://zissou.infosci.cornell.edu/convokit/datasets/>`_.

.. raw:: html

   <div class="dataset-search-container">
     <input type="text" id="dataset-search" placeholder="Search datasets by name or tags..." />
     <div class="filter-label">Filter by tag:</div>
     <div class="tag-filters">
       <!-- Tags will be dynamically populated by JavaScript -->
     </div>
     <button class="clear-filters">Clear Filters</button>
   </div>
   <div id="datasets-container">

Conversations Gone Awry
------------------------

.. raw:: html

   <div class="dataset-card" data-tags="Wikipedia, derailment, online, asynchronous, Reddit, outcome labels, summaries, persuasion, online, medium size, debate, medium conversations, timestamps">

Three related corpora of conversations that derail into antisocial behavior.

**CGA-WIKI:** Wikipedia talk page conversations that derail into personal attacks as labeled by crowdworkers.

* **Download name:** ``conversations-gone-awry-corpus``
* **Tags:** Wikipedia, derailment, online, asynchronous, outcome labels, summaries, persuasion, online, medium size, debate, medium conversations, timestamps
**CGA-CMV:** ChangeMyView discussion threads that derail into rule-violating behavior.

* **Download name:** ``conversations-gone-awry-cmv-corpus``
* **Tags:** Reddit, derailment, online, asynchronous, outcome labels, summaries, persuasion, online, medium size, debate, medium conversations, timestamps

**CGA-CMV-Large:** Expanded version of CGA-CMV dataset.

* **Download name:** ``conversations-gone-awry-cmv-corpus-large``
* **Tags:** Reddit, derailment, online, asynchronous, outcome labels, summaries, persuasion, online, medium size, debate, medium conversations, timestamps

`Documentation <https://convokit.cornell.edu/documentation/awry.html>`_

.. raw:: html

   </div>

Cornell Movie-Dialogs Corpus
-----------------------------

.. raw:: html

   <div class="dataset-card" data-tags="fictional, speaker info, synchronous, large size, medium conversations">

A large metadata-rich collection of fictional conversations extracted from raw movie scripts.

* **Download name:** ``movie-corpus``
* **Tags:** fictional, speaker info, synchronous, large size, medium conversations
* `Documentation <https://convokit.cornell.edu/documentation/movie.html>`_

.. raw:: html

   </div>

Parliament Question Time Corpus
--------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="politics, speaker info, institutional, asymmetric, synchronous, short conversations, large size">

Parliamentary question periods from May 1979 to December 2016.

* **Download name:** ``parliament-corpus``
* **Tags:** politics, speaker info, institutional, asymmetric, synchronous, short conversations, large size
* `Documentation <https://convokit.cornell.edu/documentation/parliament.html>`_

.. raw:: html

   </div>

Supreme Court Corpus
---------------------

.. raw:: html

   <div class="dataset-card" data-tags="institutional, asymmetric, law, speaker info, outcome labels, in person, synchronous, long conversations, large size">

A collection of conversations from U.S. Supreme Court oral arguments.

* **Download name:** ``supreme-corpus``
* **Tags:** institutional, asymmetric, law, speaker info, outcome labels, in person, synchronous, long conversations, large size
* `Documentation <https://convokit.cornell.edu/documentation/supreme.html>`_

.. raw:: html

   </div>

Wikipedia Talk Pages Corpus
----------------------------

.. raw:: html

   <div class="dataset-card" data-tags="online, asynchronous, Wikipedia, outcome labels, medium size, collaboration, medium conversations, timestamps">

A medium-size collection of conversations from Wikipedia editors' talk pages.

* **Download name:** ``wiki-corpus``
* **Tags:** online, asynchronous, Wikipedia, outcome labels, medium size, collaboration, medium conversations, timestamps
* `Documentation <https://convokit.cornell.edu/documentation/wiki.html>`_

.. raw:: html

   </div>

Reddit Corpus
-------------

.. raw:: html

   <div class="dataset-card" data-tags="large size, Reddit, online, asynchronous, timestamps">

Reddit conversations from over 900k subreddits, arranged by subreddit. A `small subset <https://convokit.cornell.edu/documentation/reddit-small.html>`_ sampled from 100 highly active subreddits is also available.

* **Download name:** ``subreddit-<name_of_subreddit>`` or ``reddit-corpus-small``
* **Tags:** large size, Reddit, online, asynchronous, timestamps
* `Documentation <https://convokit.cornell.edu/documentation/subreddit.html>`_

.. raw:: html

   </div>

WikiConv Corpus
---------------

.. raw:: html

   <div class="dataset-card" data-tags="large size, Wikipedia, online, asynchronous, timestamps, collaboration">

Wikipedia talk page conversations from the distinct English, German, Russian, Chinese, and Greek versions of the site, based on the reconstruction described in `this paper <https://www.cs.cornell.edu/~cristian/index_files/wikiconv-conversation-corpus.pdf>`_. Note that due to the large size of the data, every language but Greek is split up by year. We separately provide `block data retrieved directly from the Wikipedia block log <https://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/blocks.json>`_, , for reproducing the `Trajectories of Blocked Community Members <https://www.cs.cornell.edu/~cristian/Recidivism_online_files/recidivism_online.pdf>`_ paper.

* **Download name:** ``wikiconv-<language>-<year>`` for English, German, Russian, amd Chinese datasets, where the language key is the lowercase name of the language. ``wikiconv-<greek>`` for the Greek dataset.
* **Tags:** large size, Wikipedia, online, asynchronous, timestamps, collaboration
* `Documentation <https://convokit.cornell.edu/documentation/wikiconv.html>`_

.. raw:: html

   </div>

Chromium Conversations Corpus
------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="large size, online, asynchronous, utterance labels, speaker info, timestamps, collaboration, short conversations, work">

A collection of almost 1.5 million conversations and 2.8 million comments posted by developers reviewing proposed code changes in the Chromium project.

* **Download name:** ``chromium-corpus``
* **Tags:** large size, online, asynchronous, utterance labels, speaker info, timestamps, collaboration, short conversations, work
* `Documentation <https://convokit.cornell.edu/documentation/chromium.html>`_

.. raw:: html

   </div>

Tennis Interviews
------------------------

.. raw:: html

   <div class="dataset-card" data-tags="short conversations, interviews, sports, speaker info">

Transcripts for tennis singles post-match press conferences for major tournaments between 2007 to 2015 (6,467 post-match press conferences).

* **Download name:** ``tennis-corpus``
* **Tags:** short conversations, interviews, sports, speaker info
* `Documentation <https://convokit.cornell.edu/documentation/tennis.html>`_

.. raw:: html

   </div>

Winning Arguments Corpus
-------------------------

.. raw:: html

   <div class="dataset-card" data-tags="large size, Reddit, asynchronous, online, outcome labels, debate, persuasion, various topics">

A metadata-rich subset of conversations made in the r/ChangeMyView subreddit between 1 Jan 2013 - 7 May 2015, with information on the delta (success) of a speaker's utterance in convincing the poster.

* **Download name:** ``winning-args-corpus``
* **Tags:** large size, Reddit, asynchronous, online, outcome labels, debate, persuasion, various topics
* `Documentation <https://convokit.cornell.edu/documentation/winning.html>`_

.. raw:: html

   </div>

Coarse Discourse Corpus
------------------------

.. raw:: html

   <div class="dataset-card" data-tags="medium size, Reddit, online, asynchronous, utterance labels, various topics">

A subset of Reddit conversations that have been manually annotated with discourse act labels.

* **Download name:** ``reddit-coarse-discourse-corpus``
* **Tags:** medium size, Reddit, online, asynchronous, utterance labels, various topics
* `Documentation <https://convokit.cornell.edu/documentation/coarseDiscourse.html>`_

.. raw:: html

   </div>

Persuasion For Good Corpus
---------------------------

.. raw:: html

   <div class="dataset-card" data-tags="medium size, online, synchronous, speaker info, utterance labels, outcome labels, persuasion, dyadic, medium conversations">

A collection of online conversations generated by Amazon Mechanical Turk workers, where one participant (the persuader) tries to convince the other (the persuadee) to donate to a charity.

* **Download name:** ``persuasionforgood-corpus``
* **Tags:** medium size, online, synchronous, speaker info, utterance labels, outcome labels, persuasion, dyadic, medium conversations
* `Documentation <https://convokit.cornell.edu/documentation/persuasionforgood.html>`_

.. raw:: html

   </div>

Intelligence Squared Debates Corpus
------------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="small size, in person, summaries, media, utterance labels, timestamps, outcome labels, debate, long conversations, various topics, politics">

Transcripts of debates held as part of Intelligence Squared Debates.

* **Download name:** ``iq2-corpus``
* **Tags:** small size, in person, summaries, media, utterance labels, timestamps, outcome labels, debate, long conversations, various topics, politics
* `Documentation <https://convokit.cornell.edu/documentation/iq2.html>`_

.. raw:: html

   </div>

Friends Corpus
--------------

.. raw:: html

   <div class="dataset-card" data-tags="medium size, fictional, group, media, utterance labels, sarcasm">

A collection of all the conversations that occurred over 10 seasons of Friends, a popular American TV sitcom that ran in the 1990s.

* **Download name:** ``friends-corpus``
* **Tags:** medium size, fictional, group, media, utterance labels, sarcasm
* `Documentation <https://convokit.cornell.edu/documentation/friends.html>`_

.. raw:: html

   </div>

Federal Open Market Committee (FOMC) Corpus
--------------------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="small size, in person, timestamps, institutional, speaker info, utterance labels, politics, financial, long conversations, group">

Transcripts of recurring meetings of the Federal Reserve’s Open Market Committee (FOMC), where important aspects of U.S. monetary policy are decided, covering the period 1977-2008.

* **Download name:** ``fomc-corpus``
* **Tags:** small size, in person, timestamps, institutional, speaker info, utterance labels, politics, financial, long conversations
* `Documentation <https://convokit.cornell.edu/documentation/fomc.html>`_

.. raw:: html

   </div>

NPR Interview 2P Dataset Corpus
--------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="large size, in person, dyadic, media, Q&A, interviews, various topics">

This corpus contains conversations between NPR show hosts and their guests.

* **Download name:** ``npr-2p-corpus``
* **Tags:** large size, in person, dyadic, media, Q&A, interviews, various topics
* `Documentation <https://convokit.cornell.edu/documentation/npr-2p.html>`_

.. raw:: html

   </div>

DeliData Dataset Corpus
------------------------

.. raw:: html

   <div class="dataset-card" data-tags="group, synchronous, medium size, summaries, outcome labels, problem solving, collaboration">

This corpus contains conversations in multi-party problem-solving contexts, containing information about group discussions and team performance.

* **Download name:** ``deli-corpus``
* **Tags:** group, synchronous, medium size, summaries, outcome labels, problem solving, collaboration
* `Documentation <https://convokit.cornell.edu/documentation/deli.html>`_

.. raw:: html

   </div>

Switchboard Dialog Act Corpus
------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="synchronous, dyadic, medium size, speaker info, summaries, various topics, debate">

A collection of 1,155 five-minute telephone conversations between two participants, annotated with speech act tags.

* **Download name:** ``switchboard-corpus``
* **Tags:** synchronous, dyadic, medium size, speaker info, summaries, various topics, debate
* `Documentation <https://convokit.cornell.edu/documentation/switchboard.html>`_

.. raw:: html

   </div>

Stanford Politeness Corpus
------------------------

.. raw:: html

   <div class="dataset-card" data-tags="medium size, asynchronous, Wikipedia, Stack Exchange, utterance labels, online, short conversations, politeness">

Two collections of requests (from Wikipedia and Stack Exchange respectively) with politeness annotations

**Stanford Politeness (Wikipedia):** A collection of requests from Wikipedia Talk pages, annotated with politeness (4,353 utteranecs).

* **Download name:** ``wikipedia-politeness-corpus``
* **Tags:** medium size, asynchronous, Wikipedia, utterance labels, online, short conversations, politeness
* `Documentation <https://convokit.cornell.edu/documentation/wiki_politeness.html>`_

**Stanford Politeness (Stack Exchange):** A collection of requests from Stack Exchange, annotated with politeness (6,603 utteranecs).

* **Download name:** ``stack-exchange-politeness-corpus``
* **Tags:** medium size, asynchronous, Stack Exchange, utterance labels, online, short conversations, politeness
* `Documentation <https://convokit.cornell.edu/documentation/stack_politeness.html>`_

.. raw:: html

   </div>

Deception in Diplomacy Conversations
-------------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="medium size, group, speaker info, utterance labels, negotiation, medium conversations, persuasion, collaboration, deception">

Conversational dataset with intended and perceived deception labels. Over 17,000 messages annotated by the sender for their intended truthfulness and by the receiver for their perceived truthfulness.

* **Download name:** ``diplomacy-corpus``
* **Tags:** medium size, group, speaker info, utterance labels, negotiation, medium conversations, persuasion, collaboration, deception
* `Documentation <https://convokit.cornell.edu/documentation/diplomacy.html>`_

.. raw:: html

   </div>

Group Affect and Performance (GAP) Corpus
------------------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="small size, in person, group, speaker info, timestamps, summaries, outcome labels, collaboration">

A conversational dataset comprising group meetings of two to four participants that deliberate in a group decision-making exercise. This dataset contains 28 group meetings with a total of 84 participants.

* **Download name:** ``gap-corpus``
* **Tags:** institution, small size, in person, group, speaker info, timestamps, summaries, outcome labels, collaboration
* `Documentation <https://convokit.cornell.edu/documentation/gap.html>`_

.. raw:: html

   </div>

Wikipedia Articles for Deletion Corpus
---------------------------------------

.. raw:: html

   <div class="dataset-card" data-tags="Wikipedia, large size, online, asynchronous, speaker info, utterance labels, outcome labels, timestamps, debate">

A collection of Wikipedia's Articles for Deletion editor debates that occurred between January 1, 2005 and December 31, 2018. This corpus contains about 3,200,000 contributions by approximately 150,000 Wikipedia editors across almost 400,000 debates.

* **Download name:** ``wiki-articles-for-deletion-corpus``
* **Tags:** Wikipedia, large size, online, asynchronous, speaker info, utterance labels, outcome labels, timestamps, debate
* `Documentation <https://convokit.cornell.edu/documentation/wiki-articles-for-deletion-corpus.html>`_

.. raw:: html

   </div>

CaSiNo Corpus
-------------

.. raw:: html

   <div class="dataset-card" data-tags="medium size, speaker info, utterance labels, negotiation, collaboration">

CaSiNo (stands for CampSite Negotiations) is a novel dataset of 1030 negotiation dialogues. Two participants take the role of campsite neighbors and negotiate for Food, Water, and Firewood packages, based on their individual preferences and requirements.

* **Download name:** ``casino-corpus``
* **Tags:** medium size, speaker info, utterance labels, negotiation, collaboration
* `Documentation <https://convokit.cornell.edu/documentation/casino-corpus.html>`_

.. raw:: html

   </div>

SPOLIN Corpus
-------------

.. raw:: html

   <div class="dataset-card" data-tags="media, large size, online, synchronous, utterance labels">

Selected Pairs of Learnable ImprovisatioN (SPOLIN) is a collection of more than 68,000 "Yes, and" type utterance pairs extracted from the long-form improvisation podcast Spontaneanation by Paul F. Tompkins, the Cornell Movie-Dialogs Corpus, and the SubTle corpus.

* **Download name:** ``spolin-corpus``
* **Tags:** media, large size, online, synchronous, utterance labels, short conversations, various topics
* `Documentation <https://convokit.cornell.edu/documentation/spolin.html>`_

.. raw:: html

   </div>

CANDOR Corpus
-------------

.. raw:: html

   <div class="dataset-card" data-tags="synchronous, medium size, speaker info, timestamps, utterance labels">

CANDOR corpus is a dataset of 1650 conversations that strangers had over video chat with rich metadata information obtaind from pre-conversation and post-conversation surveys. The corpus is available by request from the authors (BetterUp CANDOR Corpus) and ConvoKit contains code for converting the transcripts into ConvoKit format, as detailed in the documentation.

* **Tags:** synchronous, medium size, speaker info, timestamps, utterance labels
* `Documentation <https://convokit.cornell.edu/documentation/candor.html>`_

.. raw:: html

   </div>

Fora Corpus
-------------

.. raw:: html

   <div class="dataset-card" data-tags="small size, speaker info, utterance labels, timestamps, group, in person, various topics">

Fora corpus is a dataset of 262 annotated transcripts of multi-person facilitated dialogues regarding issues like education, elections, and public health, primarily through the sharing of personal experience. The corpus is available by request from the authors (https://github.com/schropes/fora-corpus) and ConvoKit contains code for converting the transcripts into ConvoKit format, as detailed below.

* **Tags:** small size, speaker info, utterance labels, timestamps, group, in person, various topics
* `Documentation <https://convokit.cornell.edu/documentation/fora.html>`_

.. raw:: html

   </div>

   </div>

Custom Datasets
---------------

You can also use ConvoKit with your own custom datasets by loading them into a ``Corpus`` object.
See our `tutorial on converting custom data <https://github.com/CornellNLP/ConvoKit/blob/master/examples/converting_movie_corpus.ipynb>`_.
