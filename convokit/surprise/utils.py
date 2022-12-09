from typing import List, Union, Optional

import numpy as np


def random_sampler(tokens: List[Union[np.ndarray, List[str]]], sample_size: int,
                   n_samples: int) -> Optional[np.ndarray]:
    """

    :param tokens:
    :param sample_size:
    :param n_samples:
    :return:
    """
    if not sample_size:
        assert len(tokens) == 1
        return np.tile(tokens[0], (n_samples, 1))

    tokens_list = np.array([tokens_ for tokens_ in tokens if len(tokens_) >= sample_size])
    if tokens_list.shape[0] == 0:
        return None

    rng = np.random.default_rng()
    sample_idxs = rng.integers(0, tokens_list.shape[0], size=n_samples)
    return np.array([rng.choice(tokens_list[idx], sample_size) for idx in sample_idxs])
