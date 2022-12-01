import multiprocessing
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional, Any, Union, List

import kenlm
import numpy as np
from joblib import Parallel, delayed

from .perplexity import Perplexity
from .utils import create_tmp_files, delete_tmp_files


class KenlmPerplexity(Perplexity):
    """

    :param perplexity_type:
    :param kwargs:
    """

    def __init__(self, perplexity_type: str = 'kenlm_perplexity', **kwargs: Optional[Any]):
        super().__init__(perplexity_type, **kwargs)

        self._ngram_order = kwargs['ngram_order'] if 'ngram_order' in kwargs else 2
        if self._ngram_order < 2:
            warnings.warn(f'kenlm does not support n-gram order below 2; setting n-gram order to 2. '
                          f'See: https://github.com/kpu/kenlm/issues/171 for specifics.')
            self._ngram_order = 2

        if 'kenlm_path' not in kwargs:
            self._kenlm_path = os.path.join(str(Path.home()), 'kenlm')
            warnings.warn(f'the kenlm_path is unspecified, setting it to {self._kenlm_path}')
        self.__kenlm_bin_path = os.path.join(self._kenlm_path, 'build/bin')
        if not os.path.isdir(self.__kenlm_bin_path):
            raise FileNotFoundError(f'the build directory for kenlm does not exist at: {self.__kenlm_bin_path}; '
                                    f'build kenlm {self._kenlm_path} before computing surprise scores')

        self._n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else multiprocessing.cpu_count()

    @staticmethod
    def __populate_train_file(filepath: str, samples: Union[List[str], np.ndarray]):
        """

        :param filepath:
        :param samples:
        :return:
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(f'{" ".join(sample)}\n')

    def _get_kenlm_model(self, context_samples: Union[List[str], np.ndarray]) -> kenlm.Model:
        """

        :param context_samples:
        :return:
        """
        train_file, arpa_file, model_file = create_tmp_files(num_files=3)

        self.__populate_train_file(train_file.name, samples=context_samples)
        kenlm_args = [os.path.join(self.__kenlm_bin_path, 'lmplz'), '-o', f'{self._ngram_order}', '--text',
                      train_file.name, '--arpa', arpa_file.name, '--discount_fallback']
        cmd_return = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)
        if cmd_return.returncode != 0:
            delete_tmp_files([model_file, arpa_file, train_file])
            raise RuntimeError('the kenlm model training was unsuccessful')

        kenlm_args = [os.path.join(self.__kenlm_bin_path, 'build_binary'), 'trie', arpa_file.name, model_file.name]
        cmd_return = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)
        if cmd_return.returncode != 0:
            delete_tmp_files([model_file, arpa_file, train_file])
            raise RuntimeError('the kenlm model (binary) building was unsuccessful')

        kenlm_model = kenlm.Model(model_file.name)
        delete_tmp_files([model_file, arpa_file, train_file])

        return kenlm_model

    def perplexity_fn(self, target_samples: Union[List[str], np.ndarray], context_samples: Union[List[str], np.ndarray],
                      **kwargs: Optional[Any]) -> np.ndarray:
        """

        :param target_samples:
        :param context_samples:
        :param kwargs:
        :return:
        """
        self.overwrite_args(list(kwargs.keys()), kwargs)

        kenlm_model = self._get_kenlm_model(context_samples)
        model_scores = Parallel(n_jobs=self._n_jobs, backend='threading')(
            delayed(kenlm_model.score)(' '.join(target_sample)) for target_sample in target_samples)
        return np.nanmean(model_scores)
