Installing ConvoKit
-------------------

System Requirements
===================
ConvoKit requires Python 3.10 or above.

Package Installation
====================
ConvoKit can be installed via pip: ``pip3 install convokit``

Post-install Steps
==================
ConvoKit relies on NLTK and SpaCy to implement certain basic NLP functions.
If you have not already previously used these packages, they require additional first-time setup:

#. For NLTK, download the punkt tokenizer: ``import nltk; nltk.download('punkt')`` (in a ``python`` interactive session)

#. For SpaCy, download the default English model: ``python3 -m spacy download en_core_web_sm``

Optional: Choose a Backend
==========================
By default, ConvoKit uses a native Python backend which keeps all data in memory during runtime.
This is suitable for most use cases and does not require any additional setup.
However, certain use cases (including low-memory environments and real-time applications) may prefer the alternative MongoDB backend, which requires additional setup.
For more information on choosing between the two options and setting up the MongoDB backend, please consult the following guides:

.. toctree::
    :maxdepth: 1

    Choosing a Backend: native Python vs MongoDB <storage_options.rst>
    Setting up MongoDB for ConvoKit <db_setup.rst>

Configuration
==================
ConvoKit configurations are stored in "~/.convokit/config.yml", check out our `Configuration Guide <https://convokit.cornell.edu/documentation/config.html>`_ for a list of configuration details.

Troubleshooting
===============
If you run into any issues during or after installation, check out our `Troubleshooting Guide <https://convokit.cornell.edu/documentation/troubleshooting.html>`_ for a list of solutions to common issues.
