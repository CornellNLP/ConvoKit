import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Optional, Any, Union, List, Tuple

import numpy as np

from convokit.util import create_temp_files, delete_files
from .language_model import LanguageModel

try:
    import kenlm
except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError(
        "kenlm is not currently installed; run `pip install convokit[kenlm]` if you "
        "would like to use the Kenlm language model. If kenlm installation fails, please "
        "follow: https://github.com/kpu/kenlm/issues/57 to install kenlm."
    )


class Kenlm(LanguageModel):
    """A language model to compute the deviation of target from context using KenLM.

    Using KenLM library, this language model implements cross-entropy and perplexity language model
    evaluation functions, to be used in evaluating the average deviation of target text from the
    specified context.

    Run `pip install convokit[kenlm]` to install the KenLM library before using this language model
    class. If kenlm installation fails, please follow: https://github.com/kpu/kenlm/issues/57 to
    install the KenLM library.

    :param model_type: The name of the :py:class:`~convokit.Kenlm`, defaults to "kenlm". Note that
        the `model_type` can be accessed using the `type` property (e.g., `lm.type`).
    :param kwargs: Any additional keyword arguments needed in the language model evaluations. This
        language model currently uses the following keyword arguments:

        * `ngram_order`: The order of n-gram language model, when the specified `ngram_order` is
          less than 2 (or unspecified), the `ngram_order` is set to 2, since the KenLM library does
          not support n-gram order below 2 (see: https://github.com/kpu/kenlm/issues/171).
        * `trained_model_filepath`: The filepath to a pre-trained language model that is to be
          persistently used.
        * `is_persistent`: Indicator of model persistence, i.e., the model generated in the first
          pass or that loaded from `trained_model_filepath` is used in all evaluations. When the
          `trained_model_filepath` is specified, persistence is implied. Defaults to `False`.
        * `kenlm_path`: The path to the KenLM library, defaults to the user's home directory.
        * `models_dir`: The folder path to store the (trained) binary KenLM models, defaults to
          `None`, indicating that the trained KenLM models need not be stored.
        * `model_filename`: The filename used in storing model artefacts, defaults to `model_type`.
        * `n_jobs`: The number of concurrent threads to be used for routines that are parallelized
          with `joblib`, defaults to 1.

        The language model configuration can be retrieved using the `config` property of the model
        class object (e.g., `lm.config`).
    """

    def __init__(self, model_type: str = "kenlm", **kwargs: Optional[Any]):
        super().__init__(model_type, **kwargs)

        self._ngram_order = kwargs["ngram_order"] if "ngram_order" in kwargs else 2
        if self._ngram_order < 2:
            warnings.warn(
                f"kenlm does not support n-gram order below 2; setting n-gram order to 2. "
                f"See: https://github.com/kpu/kenlm/issues/171 for specifics."
            )
            self._ngram_order = 2

        self._is_persistent = kwargs["is_persistent"] if "is_persistent" in kwargs else False
        if self._is_persistent or "trained_model_filepath" in kwargs:
            self._is_persistent = True
            self.__kenlm_model = (
                Kenlm.load_kenlm_from_file(kwargs["trained_model_filepath"])
                if "trained_model_filepath" in kwargs
                else None
            )

        if "kenlm_path" not in kwargs:
            self._kenlm_path = os.path.join(str(Path.home()), "kenlm")
            warnings.warn(f"the kenlm_path is unspecified, setting it to {self._kenlm_path}")
        self.__kenlm_bin_path = os.path.join(self._kenlm_path, "build/bin")
        if not os.path.isdir(self.__kenlm_bin_path):
            raise FileNotFoundError(
                f"the build directory for kenlm does not exist at: {self.__kenlm_bin_path}; "
                f"build kenlm {self._kenlm_path} before computing surprise scores"
            )

        self._models_dir = kwargs["models_dir"] if "models_dir" in kwargs else None
        if self._models_dir and not os.path.exists(self._models_dir):
            warnings.warn(f"creating the folder: {self._models_dir} as it does not exist")
            os.makedirs(self._models_dir)
        self._model_filename = (
            kwargs["model_filename"] if "model_filename" in kwargs else self._model_type
        )

    @staticmethod
    def load_kenlm_from_file(trained_model_filepath: str) -> kenlm.Model:
        """Loads the pre-trained KenLM model from the specified filepath.

        :param trained_model_filepath: The path to the pre-trained KenLM model.
        :return: The loaded KenLM model.
        """
        kenlm_model = kenlm.Model(trained_model_filepath)
        return kenlm_model

    def __make_files(self) -> Tuple[str, str, str]:
        """Create (if needed) and return the filenames of intermittent files.

        KenLM language model needs the training data filename, .arpa filename, and the binary model
        filename to generate a KenLM model. If the models are not stored (specified through the
        argument `models_dir` in the constructor), `tempfile` files are used, else, all the files
        are generated in the `models_dir/current_timestamp` folder, using the filename specified in
        the constructor.

        :return: A tuple of filenames of all the intermittent files needed.
        """
        if self._models_dir:
            epoch = str(int(time.time()))
            os.makedirs(os.path.join(self._models_dir, epoch))

            train_filename = os.path.join(self._models_dir, epoch, f"{self._model_filename}.txt")
            arpa_filename = os.path.join(self._models_dir, epoch, f"{self._model_filename}.arpa")
            model_filename = os.path.join(self._models_dir, epoch, f"{self._model_filename}.bin")
        else:
            train_file, arpa_file, model_file = create_temp_files(num_files=3)
            train_filename, arpa_filename, model_filename = (
                train_file.name,
                arpa_file.name,
                model_file.name,
            )
        return train_filename, arpa_filename, model_filename

    @staticmethod
    def __populate_train_file(filepath: str, samples: Union[List[List[str]], np.ndarray]):
        """Writes the specified samples to a file, to be used in KenLM training.

        :param filepath: The filepath to write the samples to.
        :param samples: The samples that are to be written to the file. Each list of samples is
            delimited using a newline (`\n`).
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(f'{" ".join(sample).strip()}\n')

    def _get_kenlm_model(self, context_samples: Union[List[List[str]], np.ndarray]) -> kenlm.Model:
        """Retrieve the KenLM model trained using the specified `context_samples`.

        This method generates the training file using the `context_samples`, which is then used in
        the generation of the .arpa and a binary KenLM trained model files. These intermittent files
        are deleted, unless the specified value of `models_dir` is not `None`, indicating that the
        models are to be stored.

        :param context_samples: The context samples to be used in training the KenLM model.
        :return: The KenLM model trained on the specified `context_samples`.
        """
        train_filename, arpa_filename, model_filename = self.__make_files()

        self.__populate_train_file(train_filename, samples=context_samples)
        kenlm_args = [
            os.path.join(self.__kenlm_bin_path, "lmplz"),
            "-o",
            f"{self._ngram_order}",
            "--text",
            train_filename,
            "--arpa",
            arpa_filename,
            "--discount_fallback",
        ]
        cmd_return = subprocess.run(
            kenlm_args,
            capture_output=False,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        if cmd_return.returncode != 0:
            delete_files([model_filename, arpa_filename, train_filename])
            raise RuntimeError("the kenlm model training was unsuccessful")

        kenlm_args = [
            os.path.join(self.__kenlm_bin_path, "build_binary"),
            "trie",
            arpa_filename,
            model_filename,
        ]
        cmd_return = subprocess.run(
            kenlm_args,
            capture_output=False,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        if cmd_return.returncode != 0:
            delete_files([model_filename, arpa_filename, train_filename])
            raise RuntimeError("the kenlm model (binary) building was unsuccessful")

        kenlm_model = kenlm.Model(model_filename)
        if not self._models_dir:
            delete_files([model_filename, arpa_filename, train_filename])

        return kenlm_model

    def cross_entropy(
        self,
        target: Union[List[str], np.ndarray],
        context: Union[List[str], np.ndarray],
    ) -> float:
        """Implements the base class method to compute the cross-entropy.

        A KenLM model is trained using the specified `context`, and is used to evaluate the `target`
        text. Note that, if model persistence is indicated in the constructor (using the argument
        `is_persistent`), the model generated in the first pass or that loaded from the parameter
        value of `trained_model_filepath` is used in all evaluations. (When `trained_model_filepath`
        is specified, persistence is automatically implied.)

        The KenLM library returns a score of log-probabilities (when `score()` method is used), and
        the cross-entropy is the negative log-likelihood.

        :param target: A list of tokens that make up the target text (P).
        :param context: A list of tokens that make up the context text (Q), used to train the model.
        :return: The cross-entropy score computed using the `kenlm.score()` method.
        """
        if self.__kenlm_model is None or not self._is_persistent:
            self.__kenlm_model = self._get_kenlm_model([context])
        return -self.__kenlm_model.score(" ".join(target).strip())

    def perplexity(
        self, target: Union[List[str], np.ndarray], context: Union[List[str], np.ndarray]
    ) -> float:
        """Implements the base class method to compute perplexity.

        A KenLM model is trained using the specified `context`, and is used to evaluate the `target`
        text. Note that, if model persistence is indicated in the constructor (using the argument
        `is_persistent`), the model generated in the first pass or that loaded from the parameter
        value of `trained_model_filepath` is used in all evaluations. (When `trained_model_filepath`
        is specified, persistence is automatically implied.)

        The KenLM library returns a perplexity score, with the use of `kenlm.perplexity()` method.

        :param target: A list of tokens that make up the target text (P).
        :param context: A list of tokens that make up the context text (Q), used to train the model.
        :return: The perplexity score computed using the `kenlm.perplexity()` method.
        """
        if self.__kenlm_model is None or not self._is_persistent:
            self.__kenlm_model = self._get_kenlm_model([context])
        return self.__kenlm_model.perplexity(" ".join(target).strip())
