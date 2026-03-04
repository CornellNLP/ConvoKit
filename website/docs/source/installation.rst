Installation
============
Quick Install
-------------

This toolkit requires Python >= 3.10.

The fastest way to get started:

.. code-block:: bash

   pip3 install convokit

That's it! You're ready to use ConvoKit. Alternatively, visit our `Github Page <https://github.com/CornellNLP/ConvoKit>`_ to install from source.

Optional Dependencies
---------------------

For specific features, you may need additional packages. ConvoKit relies on NLTK and SpaCy to implement certain NLP functions. If you have not already previously used these packages, they require additional first time setup:

For NLTK, download the punkt tokenizer: ``import nltk; nltk.download('punkt')`` (in a python interactive session)

For SpaCy, download the default English model: ``python3 -m spacy download en_core_web_sm``

Troubleshooting
----------------------

If you encounter difficulties with installation, check out our `Troubleshooting Guide <https://convokit.cornell.edu/documentation/troubleshooting.html>`_ or ask in our `Discord community <https://discord.gg/WMFqMWgz6P>`_.


Basic Usage
------------------------

.. code-block:: python

   from convokit import Corpus, download

   # Load a dataset
   corpus = Corpus(download('conversations-gone-awry-corpus'))

   # print corpus summary stats
   corpus.print_summary_stats()

   # Example: extract politeness features
   from convokit import PolitenessStrategies
   ps = PolitenessStrategies(verbose=5000)
   corpus = ps.transform(corpus)

Next Steps
----------

* :doc:`datasets` - Explore available datasets
* :doc:`features` - Discover analysis features
* `API Documentation <https://convokit.cornell.edu/documentation/>`_ - Detailed API reference
