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
    """

    :param model_type:
    :param kwargs:
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
        """

        :param trained_model_filepath:
        :return:
        """
        kenlm_model = kenlm.Model(trained_model_filepath)
        return kenlm_model

    def __make_files(self) -> Tuple[str, str, str]:
        """

        :return:
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
        """

        :param filepath:
        :param samples:
        :return:
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(f'{" ".join(sample).strip()}\n')

    def _get_kenlm_model(self, context_samples: Union[List[List[str]], np.ndarray]) -> kenlm.Model:
        """

        :param context_samples:
        :return:
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
        """

        :param target:
        :param context:
        :return:
        """
        if self.__kenlm_model is None or not self._is_persistent:
            self.__kenlm_model = self._get_kenlm_model([context])
        return -self.__kenlm_model.score(" ".join(target).strip())
