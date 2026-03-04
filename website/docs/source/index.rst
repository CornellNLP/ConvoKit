ConvoKit: Conversational Analysis Toolkit
=========================================

.. image:: https://img.shields.io/pypi/v/convokit.svg
   :target: https://pypi.org/pypi/convokit/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.10%2B-blue
   :target: https://pypi.org/pypi/convokit/
   :alt: Python versions

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/CornellNLP/ConvoKit/blob/master/LICENSE.md
   :alt: License

This toolkit contains tools to extract conversational features and analyze social phenomena in conversations, using a `single unified interface <https://convokit.cornell.edu/documentation/architecture.html>`_ inspired by (and compatible with) scikit-learn. Several large conversational datasets are included together with scripts exemplifying the use of the toolkit on these datasets. The latest version is 4.0.0 (released Nov. 3, 2025); follow the project on GitHub to keep track of updates.

The toolkit currently implements features for:

Quick Links
-----------

* :doc:`installation` - Get started with ConvoKit
* :doc:`datasets` - Browse available conversational datasets
* :doc:`features` - Explore analysis features and APIs
* `Documentation <https://convokit.cornell.edu/documentation/>`_
* `GitHub Repository <https://github.com/CornellNLP/ConvoKit>`_
* `Discord Community <https://discord.gg/WMFqMWgz6P>`_

Documentation
-------------

Documentation is hosted `here <https://convokit.cornell.edu/documentation/>`_.

If you are new to ConvoKit, great places to get started are:

* The `Core Concepts tutorial <https://convokit.cornell.edu/documentation/architecture.html>`_ for an overview of the ConvoKit "philosophy" and object model
* The `High-level tutorial <https://convokit.cornell.edu/documentation/tutorial.html>`_ for a walkthrough of how to import ConvoKit into your project, load a Corpus, and use ConvoKit functions

For an overview, watch our SIGDIAL talk introducing the toolkit:

.. raw:: html

   <div style="margin: 2rem 0;">
     <iframe width="560" height="315"
             src="https://www.youtube.com/embed/nofzyxM4h1k"
             title="SIGDIAL 2020: Introducing ConvoKit"
             frameborder="0"
             allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
             allowfullscreen
             style="max-width: 100%;">
     </iframe>
   </div>


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   datasets
   features
   contributors

Community & Support
-------------------

Join our `Discord community <https://discord.gg/WMFqMWgz6P>`_ to:

* Get help with installation and usage
* Stay updated on the latest releases
* Discuss progress, features, and issues
* Share your work and connect with others

Citation
--------

If you use the code or datasets distributed with ConvoKit please acknowledge the work tied to the respective component (indicated in the documentation) in addition to:

   Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil. 2020.
   "ConvoKit: A Toolkit for the Analysis of Conversations". *Proceedings of SIGDIAL*.

Funding
-------

*ConvoKit is funded in part by the U.S. National Science Foundation under Grant No. IIS-1750615 (CAREER). Any opinions, findings, and conclusions in this work are those of the author(s) and do not necessarily reflect the views of Cornell University or the National Science Foundation.*
