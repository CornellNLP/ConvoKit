Installation
============

ConvoKit requires Python 3.10 or higher.

Quick Install
-------------

The fastest way to get started:

.. code-block:: bash

   pip install convokit
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt')"

That's it! You're ready to use ConvoKit.

Detailed Instructions
---------------------

Step 1: Install ConvoKit
~~~~~~~~~~~~~~~~~~~~~~~~~

Using pip (recommended):

.. code-block:: bash

   pip install convokit

From source:

.. code-block:: bash

   git clone https://github.com/CornellNLP/ConvoKit.git
   cd ConvoKit
   pip install -e .

Step 2: Install Spacy Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ConvoKit uses Spacy for linguistic analysis:

.. code-block:: bash

   python -m spacy download en_core_web_sm

For better performance with larger texts, you can install the larger model:

.. code-block:: bash

   python -m spacy download en_core_web_lg

Step 3: Install NLTK Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Download required NLTK datasets:

.. code-block:: bash

   python -c "import nltk; nltk.download('punkt')"

Or from within Python:

.. code-block:: python

   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')  # For newer NLTK versions

Optional Dependencies
---------------------

For specific features, you may need additional packages:

For neural models (CRAFT):
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install torch transformers

For visualization:
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install matplotlib seaborn networkx

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import errors after installation**

If you encounter import errors, ensure you're using Python 3.10+:

.. code-block:: bash

   python --version

**Spacy model not found**

Make sure you've downloaded the Spacy model:

.. code-block:: bash

   python -m spacy download en_core_web_sm

**NLTK data not found**

Download punkt tokenizer:

.. code-block:: python

   import nltk
   nltk.download('punkt')

**Memory issues with large datasets**

For large datasets, consider:

1. Processing in batches
2. Using a machine with more RAM
3. Using the dataset-specific optimized versions

For more troubleshooting help, see our `Troubleshooting Guide <https://convokit.cornell.edu/documentation/troubleshooting.html>`_ or ask in our `Discord community <https://discord.gg/WMFqMWgz6P>`_.

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import convokit
   print(convokit.__version__)

   # Try loading a small corpus
   corpus = convokit.Corpus(filename=convokit.download("friends-corpus"))
   print(f"Loaded {len(list(corpus.iter_utterances()))} utterances")

If this runs without errors, you're all set!

Development Installation
------------------------

If you plan to contribute to ConvoKit:

.. code-block:: bash

   git clone https://github.com/CornellNLP/ConvoKit.git
   cd ConvoKit
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

This installs ConvoKit in editable mode with development dependencies.

Next Steps
----------

* :doc:`datasets` - Explore available datasets
* :doc:`features` - Discover analysis features
* `API Documentation <https://convokit.cornell.edu/documentation/>`_ - Detailed API reference
