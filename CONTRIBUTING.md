# Contributing to ConvoKit

ConvoKit is constantly growing, and we welcome outside contributions of all kinds, whether in the form of new datasets, code changes, or bug reports and suggestions for new features. This document details some of the ways in which you can lend a hand.

## Reporting issues

Please use the GitHub issue tracker to report any problems you encounter while using the toolkit. To help us find a fix faster, we ask that include the following details in your report:

- Steps to reproduce the error (this can be in written form, or you can provide a code snippet)
- Specifications of the system you were using when you encountered the error. Most importantly, provide your OS, Python version, and type of python installation (system-provided, downloaded from Python.org, or Anaconda)
- The name of the dataset you were using when encountering the error. If you were using a custom dataset, provide a link to it if possible (this includes the case where you have modified one of the provided datasets)

## Feature requests / suggestions

Feature requests and/or suggestions can also be submitted through the GitHub issue tracker; please use the "enhancement" tag when doing so. Keep in mind however that we may not be able to respond to all requests.

## Contributing new datasets

If you want to contribute directly to the toolkit, one way of doing so that doesn't require extensive coding knowledge is to supply new datasets for inclusion in the next release. ConvoKit provides methods for converting raw conversation data into its own Corpus format; see the [example script](https://github.com/CornellNLP/ConvoKit/blob/master/examples/converting_movie_corpus.ipynb) for a walkthrough of how to do this.

Once you have created a Corpus representation of your dataset, follow these steps to request its inclusion in ConvoKit:

* Use `Corpus.dump()` to write the Corpus to disk
* Locate the resulting folder and compress it into a zip file
* Open a new issue on GitHub (title should start with "[Dataset]")
* Include the following information in the issue:
  * the name of the dataset
  * brief description
  * the dataset details, i.e. Speaker-, Utterance-, Conversation-, Corpus-level data and metadata (refer to [this](https://convokit.cornell.edu/documentation/chromium.html) for an example)
  * licensing information
  * publication to be cited with the dataset (if any)
  * contact information (for acknowledgement purposes)
  * a way to access the zipped corpus itself (you can either upload the file to a public file-sharing website and provide the URL, or include the file as an attachment in the issue)
- (Optional, but recommended): Also tell us some statistics about your dataset (e.g., number of conversations and participants) that we can use to advertise it!
- (Optional, but recommended): You may also include an example script or Jupyter notebook demonstrating the use of the dataset. We will then include this in our examples directory.

If we decide to accept your dataset contribution, we will notify you on the submitted issue, and upon the next public release you will be able to access the dataset through `convokit.download()`!


## Contributing code

In general, code contributions will take one of 3 different forms:

- Edits to existing functions/classes to fix a bug or implement new behavior (possibly in response to an open issue)
- Changes to the core ConvoKit object model hierarchy (currently found in `model.py`)
- A new _module_ implementing some new conversational analysis behavior

There are a number of guidelines that contributors should follow for _all_ types of code contributions, which will be listed below. There are also particular guidelines for implementing new modules, which we will cover at the end

### Style and formatting

ConvoKit [Black](https://black.readthedocs.io/en/stable/) to standardize Python code formatting.
Black is configured with `pyproject.toml`.

Developers should also set up pre-commit hooks to re-format any changed code prior to a commit.
The configuration for pre-commit is in `pre-commit-config.yaml` and `pre-commit-hooks.yaml`.

On every push to the repo, we also use GitHub Actions to check for formatting issues using Black. The
configuration options are in `.github/workflows/format.yml`.


To set up pre-commit:

```
python -m pip install -r requirements-dev.txt
pre-commit install
```

To run the formatter manually:

```
black ./convokit
```


### Testing

Please ensure that all contributions have been tested for integration with existing ConvoKit code. This is particularly pertinent for contributions that make changes to existing functions or modules. Unit tests will run automatically on Github when a pull request is opened, and may be run locally with

```
cd convokit/tests/ && python run_all_tests.py
```

after installing `pip install -r requirements.txt` and `pip install .` from the project directory.

### Documentation

All code changes should be fully documented. New public-facing classes or functions should have top-level docstrings written in Sphinx/RST syntax. If your contribution involves adding a new file, please make a corresponding RST file in the docs directory.

### Special guidelines for new modules

In addition to the above guidelines, there are also specific rules to be followed if your contribution involves a new module. They are as follows:

- _What kinds of things belong in a new module?_: In general, a ConvoKit module implements some specific operation that can be done on a conversation or set of conversations. More often than not, these are implementations of techniques introduced in research papers from the fields of natural language processing or computational social science. For example, the Hyperconvo module implements conversational graph features introduced in the paper [Characterizing Online Public Discussions through Patterns of Participant Interactions](http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html). As a general rule, a new module should provide functionality that is _standalone_ and _not derivative_ of functionality that exists elsewhere in the toolkit. For example, if a paper comes out that introduces a new set of linguistic features representing politeness, and you want to implement this in ConvoKit, most likely you should implement the features as a change to the existing PolitenessStrategies module, not as a new module.
- _New modules go in their own file_: New modules should have their own dedicated files; you should not implement you module inside an already existing file even if you think the functionality is conceptually related.
- _New modules should follow the Transformer API_: The main idea behind ConvoKit is to provide a single unified object model and API for the many conversational analysis methods that have been devised over the years. In keeping with this goal, all new modules should implement the Transformer API. From a coding perspective, this consists simply of making sure your class inherits from Transformer, and implementing the core functionality in `fit` and `transform`. We are aware that the Transformer model is not the _optimal_ interface for all techniques, but our main goal is ease of use, and this goal is well-served by having all modules look the same.
- _New modules should come with at least one example script_: When submitting a new module, please also include at least one example script or Jupyter notebook that demonstrates how to use it. This script or notebook should be documented in a way that a newcomer can use it as a tutorial to learn how to use your new module. The presence of an example script for every module also serves as a way to ensure test coverage.

### Submitting your pull request

Once you have ensured that your changes are consistent with the above criteria, you can submit a pull request! In the pull request, please provide a description of the changes that it makes, a brief justification of the change, any expected changes in output to existing example/test scripts, and (if applicable) a link to the issue or feature request that the change responds to.

Thanks for your interest in contributing to ConvoKit!
