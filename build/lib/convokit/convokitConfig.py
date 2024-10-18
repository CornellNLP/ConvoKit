import os
from typing import Optional
from yaml import load, Loader


DEFAULT_CONFIG_CONTENTS = (
    "# Default Backend Parameters\n"
    "db_host: localhost:27017\n"
    "data_directory: ~/.convokit/saved-corpora\n"
    "model_directory: ~/.convokit/saved-models\n"
    "default_backend: mem"
)

ENV_VARS = {"db_host": "CONVOKIT_DB_HOST", "default_backend": "CONVOKIT_BACKEND"}


class ConvoKitConfig:
    """
    Utility class providing read-only access to the ConvoKit config file
    """

    def __init__(self, filename: Optional[str] = None):
        if filename is None:
            filename = os.path.expanduser("~/.convokit/config.yml")

        if not os.path.isfile(filename):
            convo_dir = os.path.dirname(filename)
            if not os.path.isdir(convo_dir):
                os.makedirs(convo_dir)
            with open(filename, "w") as f:
                print(
                    f"No configuration file found at {filename}; writing with contents: \n{DEFAULT_CONFIG_CONTENTS}"
                )
                f.write(DEFAULT_CONFIG_CONTENTS)
                self.config_contents = load(DEFAULT_CONFIG_CONTENTS, Loader=Loader)
        else:
            with open(filename, "r") as f:
                self.config_contents = load(f.read(), Loader=Loader)

    def _get_config_from_env_or_file(self, config_key: str, default_val):
        env_val = os.environ.get(ENV_VARS[config_key], None)
        if env_val is not None:
            # environment variable setting takes priority
            return env_val
        return self.config_contents.get(config_key, default_val)

    @property
    def db_host(self):
        return self._get_config_from_env_or_file("db_host", "localhost:27017")

    @property
    def data_directory(self):
        return self.config_contents.get("data_directory", "~/.convokit/saved-corpora")

    @property
    def model_directory(self):
        return self.config_contents.get("model_directory", "~/.convokit/saved-models")

    @property
    def default_backend(self):
        return self._get_config_from_env_or_file("default_backend", "mem")
