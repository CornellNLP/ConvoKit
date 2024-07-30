Configurations
===================

After you import convokit for the first time, a default configuration file will be generated in ~/.convokit/config.yml.
There are currently four variables:

- db_host: database localhost port, default to be "localhost:27017".
- data_directory: local directory for downloaded corpuses, default to be "~/.convokit/saved-corpora".
- model_directory: local directory for downloaded models, default to be "~/.convokit/saved-models".
- default_backend: default ConvoKit backend choice, can be "mem" or "db", default to be "mem". For more information, check `Storage Options <https://convokit.cornell.edu/documentation/storage_options.html>`_
