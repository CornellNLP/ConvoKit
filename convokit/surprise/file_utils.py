import tempfile
from pathlib import Path
from typing import List, IO


def create_tmp_files(num_files: int) -> List[IO]:
    """

    :param num_files:
    :return:
    """
    tmp_files = []
    for _ in range(num_files):
        tmp_files.append(tempfile.NamedTemporaryFile('w', delete=True))
    return tmp_files


def delete_files(tmp_filenames: List[str], remove_parent_dir=True):
    """

    :param tmp_filenames:
    :param remove_parent_dir:
    :return:
    """
    tmp_filepaths = [Path(tmp_filename) for tmp_filename in tmp_filenames]
    parent_dir = tmp_filepaths[0].parents[0]

    for tmp_filepath in tmp_filepaths:
        Path.unlink(tmp_filepath, missing_ok=True)

    if remove_parent_dir and len(list(parent_dir.glob('*'))) == 0:
        Path.rmdir(parent_dir)
