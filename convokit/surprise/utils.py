import tempfile

import numpy as np


def random_sampler(tokens, sample_size, n_samples):
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


def create_tmp_files(num_files):
    """

    :param num_files:
    :return:
    """
    tmp_files = []
    for _ in range(num_files):
        tmp_files.append(tempfile.NamedTemporaryFile('w', delete=True))
    return tmp_files


def delete_tmp_files(tmp_files):
    """

    :param tmp_files:
    :return:
    """
    for tmp_file in tmp_files:
        try:
            tmp_file.close()
        except FileNotFoundError:
            pass
