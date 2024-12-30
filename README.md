# [ConvoKit](http://convokit.cornell.edu/)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-31-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![pypi](https://img.shields.io/pypi/v/convokit.svg)](https://pypi.org/pypi/convokit/)
[![py\_versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/pypi/convokit/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/CornellNLP/ConvoKit/blob/master/LICENSE.md)
[![Discord Community](https://img.shields.io/static/v1?logo=discord&style=flat&color=red&label=discord&message=community)](https://discord.gg/WMFqMWgz6P)


This toolkit contains tools to extract conversational features and analyze social phenomena in conversations, using a [single unified interface](https://convokit.cornell.edu/documentation/architecture.html) inspired by (and compatible with) scikit-learn.  Several large [conversational datasets](https://github.com/CornellNLP/ConvoKit#datasets) are included together with scripts exemplifying the use of the toolkit on these datasets. The latest version is [3.0.2](https://github.com/CornellNLP/ConvoKit/releases/tag/v3.0.2) (released December 27, 2024); follow the [project on GitHub](https://github.com/CornellNLP/ConvoKit) to keep track of updates.

Join our [Discord community](https://discord.gg/WMFqMWgz6P) to stay informed, connect with fellow developers, and be part of an engaging space where we share progress, discuss features, and tackle issues together.

Read our [documentation](https://convokit.cornell.edu/documentation) or try ConvoKit in our [interactive tutorial](https://colab.research.google.com/github/CornellNLP/ConvoKit/blob/master/examples/Introduction_to_ConvoKit.ipynb).

The toolkit currently implements features for:

### [Linguistic coordination](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) <sub><sup>[(API)](https://convokit.cornell.edu/documentation/coordination.html)</sup></sub>

A measure of linguistic influence (and relative power) between individuals or groups based on their use of function words.
Example: [exploring the balance of power in the U.S. Supreme Court](https://github.com/CornellNLP/ConvoKit/blob/master/examples/coordination/examples.ipynb).

### [Politeness strategies](https://www.cs.cornell.edu/~cristian/Politeness.html) <sub><sup>[(API)](https://convokit.cornell.edu/documentation/politenessStrategies.html)</sup></sub>

A set of lexical and parse-based features correlating with politeness and impoliteness.
Example: [understanding the (mis)use of politeness strategies in conversations gone awry on Wikipedia](https://github.com/CornellNLP/ConvoKit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb).

### [Expected Conversational Context Framework](https://tisjune.github.io/research/dissertation) <sub><sup>[(API)](https://convokit.cornell.edu/documentation/expected_context_model.html)</sup></sub>

A framework for characterizing utterances and terms based on their expected conversational context, consisting of model implementations and wrapper pipelines.
Examples: [deriving question types and other characterizations in British parliamentary question periods](https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/parliament_demo.ipynb),
[exploration of Switchboard dialog acts corpus](https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/switchboard_exploration_demo.ipynb),  [examining Wikipedia talk page discussions](https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/wiki_awry_demo.ipynb) and [computing the orientation of justice utterances in the US Supreme Court](https://github.com/CornellNLP/ConvoKit/blob/master/convokit/expected_context_framework/demos/scotus_orientation_demo.ipynb)

<!-- ### [Prompt types](http://www.cs.cornell.edu/~cristian/Asking_too_much.html) <sub><sup>[(API)](https://convokit.cornell.edu/documentation/promptTypes.html)</sup></sub>

An unsupervised method for grouping utterances and utterance features by their rhetorical role.
Examples: [extracting question types in the U.K. parliament](https://github.com/CornellNLP/ConvoKit/blob/master/examples/prompt-types/prompt-type-wrapper-demo.ipynb), [extended version demonstrating additional functionality](https://github.com/CornellNLP/ConvoKit/blob/master/examples/prompt-types/prompt-type-demo.ipynb), [understanding the use of conversational prompts in conversations gone awry on Wikipedia](https://github.com/CornellNLP/ConvoKit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb).

Also includes functionality to extract surface motifs to represent utterances, used in the above paper [(API)](https://convokit.cornell.edu/documentation/phrasingMotifs.html). -->

### [Hypergraph conversation representation](http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html) <sub><sup>[(API)](https://convokit.cornell.edu/documentation/hyperconvo.html)</sup></sub>
A method for extracting structural features of conversations through a hypergraph representation.
Example: [hypergraph creation and feature extraction, visualization and interpretation on a subsample of Reddit](https://github.com/CornellNLP/ConvoKit/blob/master/examples/hyperconvo/hyperconvo_demo.ipynb).

### [Linguistic diversity in conversations](http://www.cs.cornell.edu/~cristian/Finding_your_voice__linguistic_development.html) <sub><sup>[(API)](https://convokit.cornell.edu/documentation/speakerConvoDiversity.html)</sup></sub>
A method to compute the linguistic diversity of individuals within their own conversations, and between other individuals in a population.
Example: [speaker conversation attributes and diversity example on ChangeMyView](https://github.com/CornellNLP/ConvoKit/blob/master/examples/speaker-convo-attributes/speaker-convo-diversity-demo.ipynb)

### [CRAFT: Online forecasting of conversational outcomes](https://arxiv.org/abs/1909.01362) <sub><sup>[(API)](https://convokit.cornell.edu/documentation/forecaster.html)</sup></sub>
A neural model for forecasting future outcomes of conversations (e.g., derailment into personal attacks) as they develop.
Available as an interactive notebook: [full version (fine-tuning + inference)](https://colab.research.google.com/drive/1SH4iMEHdoH4IovN-b9QOSK4kG4DhAwmb) or [inference-only](https://colab.research.google.com/drive/1GvICZN0VwZQSWw3pJaEVY-EQGoO-L5lH).



## Datasets
ConvoKit ships with several datasets ready for use "out-of-the-box".
These datasets can be downloaded using the `convokit.download()` [helper function](https://github.com/CornellNLP/ConvoKit/blob/master/convokit/util.py).  Alternatively you can access them directly [here](http://zissou.infosci.cornell.edu/convokit/datasets/).

### Conversations Gone Awry Datasets ([Wikipedia](https://convokit.cornell.edu/documentation/awry.html)/[CMV](https://convokit.cornell.edu/documentation/awry_cmv.html))

Two related corpora of conversations that derail into antisocial behavior. One corpus (CGA-WIKI) consists of Wikipedia talk page conversations that derail into personal attacks as labeled by crowdworkers (4,188 conversations containing 30.021 comments). The other (CGA-CMV) consists of discussion threads on the subreddit ChangeMyView (CMV) that derail into rule-violating behavior as determined by the presence of a moderator intervention (6,842 conversations containing 42,964 comments).
Name for download: `conversations-gone-awry-corpus` (for CGA-WIKI) or `conversations-gone-awry-cmv-corpus` (for CGA-CMV)

### [Cornell Movie-Dialogs Corpus](https://convokit.cornell.edu/documentation/movie.html)

A large metadata-rich collection of fictional conversations extracted from raw movie scripts. (220,579 conversational exchanges between 10,292 pairs of movie characters in 617 movies).
Name for download: `movie-corpus`

### [Parliament Question Time Corpus](https://convokit.cornell.edu/documentation/parliament.html)

Parliamentary question periods from May 1979 to December 2016 (216,894 question-answer pairs).
Name for download: `parliament-corpus`

### [Supreme Court Corpus](https://convokit.cornell.edu/documentation/supreme.html)

A collection of conversations from the U.S. Supreme Court Oral Arguments.
Name for download: `supreme-corpus`

### [Wikipedia Talk Pages Corpus](https://convokit.cornell.edu/documentation/wiki.html)

A medium-size collection of conversations from Wikipedia editors' talk pages.
Name for download: `wiki-corpus`

### [Tennis Interviews](https://convokit.cornell.edu/documentation/tennis.html)

Transcripts for tennis singles post-match press conferences for major tournaments between 2007 to 2015 (6,467 post-match press conferences).
Name for download: `tennis-corpus`

### [Reddit Corpus](https://convokit.cornell.edu/documentation/subreddit.html)

Reddit conversations from over 900k subreddits, arranged by subreddit. A [small subset](https://convokit.cornell.edu/documentation/reddit-small.html) sampled from 100 highly active subreddits is also available.

Name for download: `subreddit-<name_of_subreddit>` for the by-subreddit data, `reddit-corpus-small` for the small subset.

### [WikiConv Corpus](https://convokit.cornell.edu/documentation/wikiconv.html)

The full corpus of Wikipedia talk page conversations, based on the reconstruction described in [this paper](http://www.cs.cornell.edu/~cristian/index_files/wikiconv-conversation-corpus.pdf).
Note that due to the large size of the data, it is split up by year.
We separately provide [block data retrieved directly from the Wikipedia block log](https://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/blocks.json), for reproducing the [Trajectories of Blocked Community Members](http://www.cs.cornell.edu/~cristian/Recidivism_online_files/recidivism_online.pdf) paper.

Name for download: `wikiconv-<year>` to download wikiconv data for the specified year.

### [Chromium Conversations Corpus](https://convokit.cornell.edu/documentation/chromium.html)

A collection of almost 1.5 million conversations and 2.8 million comments posted by developers reviewing proposed code changes in the Chromium project.

Name for download: `chromium-corpus`

### [Winning Arguments Corpus](https://convokit.cornell.edu/documentation/winning.html)

A metadata-rich subset of conversations made in the r/ChangeMyView subreddit between 1 Jan 2013 - 7 May 2015, with information on the delta (success) of a speaker's utterance in convincing the poster.

Name for download: `winning-args-corpus`

### [Coarse Discourse Corpus](https://convokit.cornell.edu/documentation/coarseDiscourse.html)

A subset of Reddit conversations that have been manually annotated with discourse act labels.

Name for download: `reddit-coarse-discourse-corpus`

### [Persuasion For Good Corpus](https://convokit.cornell.edu/documentation/persuasionforgood.html)

A collection of online conversations generated by Amazon Mechanical Turk workers, where one participant (the *persuader*) tries to convince the other (the *persuadee*) to donate to a charity.

Name for download: `persuasionforgood-corpus`

### [Intelligence Squared Debates Corpus](https://convokit.cornell.edu/documentation/iq2.html)

Transcripts of debates held as part of Intelligence Squared Debates.

Name for download: `iq2-corpus`

### [Friends Corpus](https://convokit.cornell.edu/documentation/friends.html)

A collection of all the conversations that occurred over 10 seasons of Friends, a popular American TV sitcom that ran in the 1990s.

Name for download: `friends-corpus`

### [Federal Open Market Committee (FOMC) Corpus](https://convokit.cornell.edu/documentation/fomc.html)

Transcripts of recurring meetings of the Federal Reserve’s Open Market Committee (FOMC), where important aspects of U.S. monetary policy are decided, covering the period 1977-2008.

Name for download: `fomc-corpus`

### [NPR Interview 2P Dataset Corpus](https://convokit.cornell.edu/documentation/npr-2p.html)

This corpus contains conversations between NPR show hosts and their guests.

Name for download: `npr-2p-corpus`

### [DeliData Dataset Corpus](https://convokit.cornell.edu/documentation/deli.html)

This corpus contains conversations in multi-party problem-solving contexts, containing information about group discussions and team performance.

Name for download: `deli-corpus`

### [Switchboard Dialog Act Corpus](https://convokit.cornell.edu/documentation/switchboard.html)

A collection of 1,155 five-minute telephone conversations between two participants, annotated with speech act tags.

Name for download: `switchboard-corpus`

### Stanford Politeness Corpus ([Wikipedia](https://convokit.cornell.edu/documentation/wiki_politeness.html)/[Stack Exchange](https://convokit.cornell.edu/documentation/stack_politeness.html))

Two collections of requests (from Wikipedia and Stack Exchange respectively) with politeness annotations. Name for download: `wikipedia-politeness-corpus` (Wikipedia portion), `stack-exchange-politeness-corpus` (Stack Exchange portion).

### [Deception in Diplomacy Conversations](https://convokit.cornell.edu/documentation/diplomacy.html)

Conversational dataset with intended and perceived deception labels. Over 17,000 messages annotated by the sender for their intended truthfulness and by the receiver for their perceived truthfulness.

Name for download: `diplomacy-corpus`

### [Group Affect and Performance (GAP) Corpus](https://convokit.cornell.edu/documentation/gap.html)

A conversational dataset comprising group meetings of two to four participants that deliberate in a group decision-making exercise. This dataset contains 28 group meetings with a total of 84 participants.

Name for download: `gap-corpus`

### [Wikipedia Articles for Deletion Corpus](https://convokit.cornell.edu/documentation/wiki-articles-for-deletion-corpus.html)

A collection of Wikipedia's Articles for Deletion editor debates that occurred between January 1, 2005 and December 31, 2018. This corpus contains about 3,200,000 contributions by approximately 150,000 Wikipedia editors across almost 400,000 debates.

Name for download: `wiki-articles-for-deletion-corpus`

### [CaSiNo Corpus](https://convokit.cornell.edu/documentation/casino-corpus.html)
CaSiNo (stands for CampSite Negotiations) is a novel dataset of 1030 negotiation dialogues. Two participants take the role of campsite neighbors and negotiate for Food, Water, and Firewood packages, based on their individual preferences and requirements.

Name for download: `casino-corpus`

### [SPOLIN Corpus](https://convokit.cornell.edu/documentation/spolin.html)
Selected Pairs of Learnable ImprovisatioN (SPOLIN) is a collection of more than 68,000 "Yes, and" type utterance pairs extracted from the long-form improvisation podcast Spontaneanation by Paul F. Tompkins, the Cornell Movie-Dialogs Corpus, and the SubTle corpus.

Name for download: `spolin-corpus`

### ...And your own corpus!

In addition to the provided datasets, you may also use ConvoKit with your own custom datasets by loading them into a `convokit.Corpus` object. [This example script](https://github.com/CornellNLP/ConvoKit/blob/master/examples/converting_movie_corpus.ipynb) shows how to construct a Corpus from custom data.

## Installation
This toolkit requires Python >= 3.10.

1. Download the toolkit: `pip3 install convokit`
2. Download Spacy's English model: `python3 -m spacy download en`
3. Download NLTK's 'punkt' model: `import nltk; nltk.download('punkt')` (in Python interpreter)

Alternatively, visit our [Github Page](https://github.com/CornellNLP/ConvoKit) to install from source.

**If you encounter difficulties with installation**, check out our **[Troubleshooting Guide](https://convokit.cornell.edu/documentation/troubleshooting.html)** for a list of solutions to common issues.

## Documentation
Documentation is hosted [here](https://convokit.cornell.edu/documentation/). If you are new to ConvoKit, great places to get started are the [Core Concepts tutorial](https://convokit.cornell.edu/documentation/architecture.html) for an overview of the ConvoKit "philosophy" and object model, and the [High-level tutorial](https://convokit.cornell.edu/documentation/tutorial.html) for a walkthrough of how to import ConvoKit into your project, load a Corpus, and use ConvoKit functions.

For an overview, watch our SIGDIAL talk introducing the toolkit:
[![SIGDIAL 2020: Introducing ConvoKit](http://i3.ytimg.com/vi/nofzyxM4h1k/hqdefault.jpg)](https://youtu.be/nofzyxM4h1k "SIGDIAL 2020: Introducing ConvoKit")

## Contributing

We welcome community contributions. To see how you can help out, check the [contribution guidelines](https://github.com/CornellNLP/ConvoKit/blob/master/CONTRIBUTING.md).

## Citing

If you use the code or datasets distributed with ConvoKit please acknowledge the work tied to the respective component (indicated in the documentation) in addition to:

Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil. 2020. "[ConvoKit: A Toolkit for the Analysis of Conversations](https://www.cs.cornell.edu/~cristian/ConvoKit_Demo_Paper_files/convokit-demo-paper.pdf)". Proceedings of SIGDIAL.

[ConvoKit](http://convokit.cornell.edu/)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cristiandnm"><img src="https://avatars.githubusercontent.com/u/8700563?v=4?s=100" width="100px;" alt="Cristian Danescu-Niculescu-Mizil"/><br /><sub><b>Cristian Danescu-Niculescu-Mizil</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=cristiandnm" title="Code">💻</a> <a href="#data-cristiandnm" title="Data">🔣</a> <a href="#ideas-cristiandnm" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-cristiandnm" title="Maintenance">🚧</a> <a href="https://github.com/CornellNLP/ConvoKit/commits?author=cristiandnm" title="Documentation">📖</a> <a href="https://github.com/CornellNLP/ConvoKit/pulls?q=is%3Apr+reviewed-by%3Acristiandnm" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/andrewzhouwang"><img src="https://avatars.githubusercontent.com/u/4683423?v=4?s=100" width="100px;" alt="Andrew Wang"/><br /><sub><b>Andrew Wang</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=qema" title="Code">💻</a> <a href="#data-qema" title="Data">🔣</a> <a href="#ideas-qema" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-qema" title="Maintenance">🚧</a> <a href="https://github.com/CornellNLP/ConvoKit/commits?author=qema" title="Documentation">📖</a> <a href="https://github.com/CornellNLP/ConvoKit/pulls?q=is%3Apr+reviewed-by%3Aqema" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://tisjune.github.io"><img src="https://avatars.githubusercontent.com/u/8534072?v=4?s=100" width="100px;" alt="Justine Zhang"/><br /><sub><b>Justine Zhang</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=tisjune" title="Code">💻</a> <a href="#data-tisjune" title="Data">🔣</a> <a href="#ideas-tisjune" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-tisjune" title="Maintenance">🚧</a> <a href="https://github.com/CornellNLP/ConvoKit/commits?author=tisjune" title="Documentation">📖</a> <a href="https://github.com/CornellNLP/ConvoKit/pulls?q=is%3Apr+reviewed-by%3Atisjune" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://cs.cornell.edu/~jpchang"><img src="https://avatars.githubusercontent.com/u/989906?v=4?s=100" width="100px;" alt="Jonathan Chang"/><br /><sub><b>Jonathan Chang</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=jpwchang" title="Code">💻</a> <a href="#data-jpwchang" title="Data">🔣</a> <a href="#ideas-jpwchang" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-jpwchang" title="Maintenance">🚧</a> <a href="https://github.com/CornellNLP/ConvoKit/commits?author=jpwchang" title="Documentation">📖</a> <a href="https://github.com/CornellNLP/ConvoKit/pulls?q=is%3Apr+reviewed-by%3Ajpwchang" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cs.cornell.edu/~liye/"><img src="https://avatars.githubusercontent.com/u/12224673?v=4?s=100" width="100px;" alt="Liye Fu"/><br /><sub><b>Liye Fu</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=liye" title="Code">💻</a> <a href="#data-liye" title="Data">🔣</a> <a href="#ideas-liye" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-liye" title="Maintenance">🚧</a> <a href="https://github.com/CornellNLP/ConvoKit/commits?author=liye" title="Documentation">📖</a> <a href="https://github.com/CornellNLP/ConvoKit/pulls?q=is%3Apr+reviewed-by%3Aliye" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/calebchiam"><img src="https://avatars.githubusercontent.com/u/14286996?v=4?s=100" width="100px;" alt="calebchiam"/><br /><sub><b>calebchiam</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=calebchiam" title="Code">💻</a> <a href="#data-calebchiam" title="Data">🔣</a> <a href="#ideas-calebchiam" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-calebchiam" title="Maintenance">🚧</a> <a href="https://github.com/CornellNLP/ConvoKit/commits?author=calebchiam" title="Documentation">📖</a> <a href="https://github.com/CornellNLP/ConvoKit/pulls?q=is%3Apr+reviewed-by%3Acalebchiam" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rgangela99"><img src="https://avatars.githubusercontent.com/u/35738132?v=4?s=100" width="100px;" alt="rgangela99"/><br /><sub><b>rgangela99</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=rgangela99" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Khonzoda"><img src="https://avatars.githubusercontent.com/u/26072772?v=4?s=100" width="100px;" alt="Khonzoda Umarova"/><br /><sub><b>Khonzoda Umarova</b></sub></a><br /><a href="#data-Khonzoda" title="Data">🔣</a> <a href="#maintenance-Khonzoda" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mwilbz"><img src="https://avatars.githubusercontent.com/u/14115641?v=4?s=100" width="100px;" alt="mwilbz"/><br /><sub><b>mwilbz</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=mwilbz" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.alexkoen.com"><img src="https://avatars.githubusercontent.com/u/43913902?v=4?s=100" width="100px;" alt="Alex Koen"/><br /><sub><b>Alex Koen</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/issues?q=author%3Aakoen" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://emtseng.me"><img src="https://avatars.githubusercontent.com/u/5270852?v=4?s=100" width="100px;" alt="Emily Tseng"/><br /><sub><b>Emily Tseng</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/issues?q=author%3Aemtseng" title="Bug reports">🐛</a> <a href="#data-emtseng" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ZiggyFloat"><img src="https://avatars.githubusercontent.com/u/41927607?v=4?s=100" width="100px;" alt="Uliyana Kubasova"/><br /><sub><b>Uliyana Kubasova</b></sub></a><br /><a href="#data-ZiggyFloat" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jschluger.github.io/"><img src="https://avatars.githubusercontent.com/u/14956008?v=4?s=100" width="100px;" alt="Jack Schluger"/><br /><sub><b>Jack Schluger</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/issues?q=author%3Ajschluger" title="Bug reports">🐛</a> <a href="https://github.com/CornellNLP/ConvoKit/commits?author=jschluger" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kushalchawla"><img src="https://avatars.githubusercontent.com/u/8416863?v=4?s=100" width="100px;" alt="Kushal Chawla"/><br /><sub><b>Kushal Chawla</b></sub></a><br /><a href="#data-kushalchawla" title="Data">🔣</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sc782"><img src="https://avatars.githubusercontent.com/u/14970930?v=4?s=100" width="100px;" alt="June Cho"/><br /><sub><b>June Cho</b></sub></a><br /><a href="#data-sc782" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/noameshed"><img src="https://avatars.githubusercontent.com/u/40632766?v=4?s=100" width="100px;" alt="Noam Eshed"/><br /><sub><b>Noam Eshed</b></sub></a><br /><a href="#data-noameshed" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/szmurlo"><img src="https://avatars.githubusercontent.com/u/31192340?v=4?s=100" width="100px;" alt="Andrew Szmurlo"/><br /><sub><b>Andrew Szmurlo</b></sub></a><br /><a href="#data-szmurlo" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kcsadow"><img src="https://avatars.githubusercontent.com/u/34074151?v=4?s=100" width="100px;" alt="Katharine Sadowski"/><br /><sub><b>Katharine Sadowski</b></sub></a><br /><a href="#data-kcsadow" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lucasvanbramer"><img src="https://avatars.githubusercontent.com/u/32553676?v=4?s=100" width="100px;" alt="Lucas Van Bramer"/><br /><sub><b>Lucas Van Bramer</b></sub></a><br /><a href="#data-lucasvanbramer" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://mariannealq.com"><img src="https://avatars.githubusercontent.com/u/16949591?v=4?s=100" width="100px;" alt="Marianne Aubin"/><br /><sub><b>Marianne Aubin</b></sub></a><br /><a href="#data-maubinle" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dn273"><img src="https://avatars.githubusercontent.com/u/27926662?v=4?s=100" width="100px;" alt="Di Ni"/><br /><sub><b>Di Ni</b></sub></a><br /><a href="#data-dn273" title="Data">🔣</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gdeng96"><img src="https://avatars.githubusercontent.com/u/8600751?v=4?s=100" width="100px;" alt="gdeng96"/><br /><sub><b>gdeng96</b></sub></a><br /><a href="#data-gdeng96" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/junfrankli"><img src="https://avatars.githubusercontent.com/u/22462584?v=4?s=100" width="100px;" alt="Frank Li"/><br /><sub><b>Frank Li</b></sub></a><br /><a href="#data-junfrankli" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://rujzhao.com"><img src="https://avatars.githubusercontent.com/u/31158748?v=4?s=100" width="100px;" alt="rjz46"/><br /><sub><b>rjz46</b></sub></a><br /><a href="#data-rjz46" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/KatyBlumer"><img src="https://avatars.githubusercontent.com/u/3669069?v=4?s=100" width="100px;" alt="KatyBlumer"/><br /><sub><b>KatyBlumer</b></sub></a><br /><a href="#data-KatyBlumer" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/als452"><img src="https://avatars.githubusercontent.com/u/15838258?v=4?s=100" width="100px;" alt="als452"/><br /><sub><b>als452</b></sub></a><br /><a href="#data-als452" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/KaminskyJ"><img src="https://avatars.githubusercontent.com/u/26395772?v=4?s=100" width="100px;" alt="KaminskyJ"/><br /><sub><b>KaminskyJ</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=KaminskyJ" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ap1075"><img src="https://avatars.githubusercontent.com/u/25790092?v=4?s=100" width="100px;" alt="Armaan Puri"/><br /><sub><b>Armaan Puri</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=Ap1075" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/oscarso2000"><img src="https://avatars.githubusercontent.com/u/20172573?v=4?s=100" width="100px;" alt="Oscar So"/><br /><sub><b>Oscar So</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=oscarso2000" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://justin-cho.com"><img src="https://avatars.githubusercontent.com/u/31977186?v=4?s=100" width="100px;" alt="Justin Cho"/><br /><sub><b>Justin Cho</b></sub></a><br /><a href="#data-wise-east" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/seanzhangkx8"><img src="https://avatars.githubusercontent.com/u/106214464?v=4?s=100" width="100px;" alt="seanzhangkx8"/><br /><sub><b>seanzhangkx8</b></sub></a><br /><a href="https://github.com/CornellNLP/ConvoKit/commits?author=seanzhangkx8" title="Code">💻</a> <a href="#ideas-seanzhangkx8" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-seanzhangkx8" title="Maintenance">🚧</a> <a href="https://github.com/CornellNLP/ConvoKit/commits?author=seanzhangkx8" title="Documentation">📖</a> <a href="https://github.com/CornellNLP/ConvoKit/pulls?q=is%3Apr+reviewed-by%3Aseanzhangkx8" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
