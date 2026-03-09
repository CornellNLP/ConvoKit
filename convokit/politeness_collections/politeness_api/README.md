Stanford Politeness API
=======================
#### Version 1.02 (developing)

 This Python implementation uses spacy as parsing tool and no longer requires the user to parse the document.(based on Version 1.01 implemented by Yiqing Hua)

#### Version 2.00 (released March 2017)

 Python3 version is available here: https://github.com/sudhof/politeness/tree/python3 (refactored from Version 1.01 through the kindness of Benjamin Meyers).

 Note: This python3 version was not yet tested by us, nor compared against the results from our paper (listed below).  The code used in the paper is still here in the master branch of this repository (keep on reading).


#### Version 1.01 (released October 2014)

Python implementation of a politeness classifier for requests, based on the work described in:

	A computational approach to politeness with application to social factors.
	Cristian Danescu-Niculescu-Mizil, Moritz Sudhof, Dan Jurafsky, Jure Leskovec, Christopher Potts.
	Proceedings of ACL, 2013.


We release this code hoping that others will use and improve on our work.

NOTE: If you use this API in your work please send an email to cristian@cs.cornell.edu so we can add you to our list of users.  Thanks!


**Further resources:**

    Info about our work: http://cs.cornell.edu/~cristian/Politeness.html

    A web interface to the politeness model: http://politeness.cornell.edu/

    The Stanford Politeness Corpus: http://cs.cornell.edu/~cristian/Politeness_files/Stanford_politeness_corpus.zip


**Using this API you can:**

- classify requests using politeness.model.score  (using the provided pre-trained model)

- train new models on new data using politeness.scripts.train_model

- experiment with new politeness features in politeness.features.vectorizer and politeness.features.politeness_strategies


**Input:** Requests must be a list of text that you want to process.  A sample of the expected format for documents is given in politeness.test_documents.


**Caveat:** This work focuses on requests, not all kinds of utterances. The model's predictions on non-request utterances will be less accurate. As a bonus, our code also includes a very simple heuristic to check whether a document looks like a request (see politeness.request_utils).


**Requirements:**

python package requirements are listed in requirements.txt. We recommend setting up a new python environment using virtualenv and installing the dependencies by running

    pip install -r requirements.txt

Additionally, since the code uses nltk.word_tokenize to tokenize text, you will need to download the  tokenizers/punkt/english.pickle nltk resource. If you've worked with nltk before, there's a good chance you've already downloaded this model. Otherwise, open the python interpreter and run:

    import nltk
    nltk.download()

In the window that opens, navigate to Models and download the Punkt Tokenizer Models.

The package uses spacy for dependency parsing, please download the English language model for spacy:

	python -m spacy download en

If you are using python 2, please install package future:

	pip install future

If you are using python 2.6, install:

	pip install importlib       # for future.standard_library.import_ function only
	pip install unittest2       # to run the test suite
	pip install argparse        # for the backported http.server module from Py3.3



**Sanity Check:**

To make sure everything's working, navigate to the code directory and run

    python model.py

This should print out the politeness probabilities for 4 test documents.


**Contact:** Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil) and sudhof@stanford.edu (Moritz Sudhof)
