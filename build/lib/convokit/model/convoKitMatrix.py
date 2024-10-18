import pandas as pd
from typing import Optional, List
import pickle
import os
import numpy as np
from convokit.util import warn
from scipy.sparse import issparse, csr_matrix, hstack, vstack
import scipy


class ConvoKitMatrix:
    """
    A ConvoKitMatrix stores the vector representations of some set of Corpus components (i.e. Utterances,
    Conversations, Speakers).

    :param name: descriptive name for the matrix
    :param matrix: numpy or scipy array matrix
    :param ids: optional list of Corpus component object ids, where each id corresponds to each row of the matrix
    :param columns: optional list of names for the columns of the matrix

    :ivar name: name of the matrix
    :ivar matrix: the matrix data
    :ivar ids: ids corresponding to rows
    :ivar columns: names corresponding to columns
    :ivar ids_to_idx: a mapping from id to the row index
    :ivar cols_to_idx: a mapping from column name to the column index
    """

    def __init__(
        self, name, matrix, ids: Optional[List[str]] = None, columns: Optional[List[str]] = None
    ):
        self.name = name
        self._matrix = matrix
        self._sparse = isinstance(self._matrix, scipy.sparse.csr.csr_matrix)
        self.ids = np.arange(matrix.shape[0]) if ids is None else ids
        self.columns = np.arange(matrix.shape[1]) if columns is None else columns
        self.ids_to_idx = {id: idx for idx, id in enumerate(self.ids)}
        self.cols_to_idx = {col: idx for idx, col in enumerate(self.columns)}
        self._initialization_checks()

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, scipy.sparse.csr.csr_matrix):
            self._matrix = value
            self._sparse = isinstance(value, scipy.sparse.csr.csr_matrix)
        else:
            raise ValueError("Matrix must be a numpy ndarray or a scipy csr_matrix.")

    @matrix.deleter
    def matrix(self):
        warn(
            "ConvoKitMatrix's internal matrix cannot be deleted. Use Corpus.delete_vector_matrix() instead."
        )

    def _initialization_checks(self):
        try:
            self.matrix.shape
        except AttributeError:
            raise AttributeError("Input matrix is not a numpy or scipy matrix.")

        try:
            assert len(self.ids) == self.matrix.shape[0]
            if self.columns is not None:
                assert len(self.columns) == self.matrix.shape[1]
        except AssertionError:
            raise ValueError(
                "Input matrix dimensions {} do not match "
                "length of ids and/or columns".format(self.matrix.shape)
            )

    def get_vectors(
        self,
        ids: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        as_dataframe: bool = False,
    ):
        """

        :param ids: optional list of object ids to get vectors for; all by default
        :param columns: optional list of named columns of the vector to include; all by default
        :param as_dataframe: whether to return the vector as a dataframe (True) or in its raw array form (False). False
            by default.
        :return: a vector matrix (either np.ndarray or csr_matrix) or a pandas dataframe
        """
        ids = self.ids if ids is None else ids
        id_indices = [self.ids_to_idx[id] for id in ids]

        cols = self.columns if columns is None else columns
        col_indices = [self.cols_to_idx[col] for col in cols]

        if not as_dataframe:
            return self.matrix[id_indices][:, col_indices]
        else:
            mat = self.matrix.toarray() if issparse(self.matrix) else self.matrix
            return pd.DataFrame(mat[id_indices][:, col_indices], index=ids, columns=cols)

    def to_dict(self):
        if self.columns is None:
            raise ValueError(
                "Matrix columns are missing. Update matrix.columns with a list of column names."
            )
        d = dict()
        for id, idx in self.ids_to_idx.items():
            row = self.matrix[idx]
            d[id] = {self.columns[i]: v for i, v in enumerate(row)}
        return d

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the matrix of vectors into a pandas DataFrame.

        :return: a pandas DataFrame
        """
        index = {idx: id_ for id_, idx in self.ids_to_idx.items()}
        sorted_ids = [index[idx] for idx in sorted(index)]
        matrix = self.matrix.toarray() if issparse(self.matrix) else self.matrix
        return pd.DataFrame(matrix, index=sorted_ids, columns=self.columns)

    @staticmethod
    def from_file(filepath):
        """
        Initialize a ConvoKitMatrix from a file of form "vector.[name].p".

        :param filepath:
        :return:
        """
        with open(filepath, "rb") as f:
            retval: ConvoKitMatrix = pickle.load(f)
            if not retval._sparse:
                retval.matrix = retval.matrix.toarray()
            return retval

    @staticmethod
    def from_dir(dirpath, matrix_name):
        """
        Initialize a ConvoKitMatrix of the specified `matrix_name` from a specified directory `dirpath`.

        :param dirpath: path to Corpus directory
        :param matrix_name: name of vector matrix
        :return: the initialized ConvoKitMatrix
        """
        try:
            with open(os.path.join(dirpath, "vectors.{}.p".format(matrix_name)), "rb") as f:
                retval: ConvoKitMatrix = pickle.load(f)
                if not retval._sparse:
                    retval.matrix = retval.matrix.toarray()
                return retval
        except FileNotFoundError:
            warn("Could not find vector with name: {} at {}.".format(matrix_name, dirpath))
            return None

    def dump(self, dirpath):
        """
        Dumps the ConvoKitMatrix as a pickle file.

        :param dirpath: directory path to Corpus
        :return: None
        """
        if not issparse(self.matrix):
            temp = self.matrix
            self.matrix = csr_matrix(self.matrix)
            with open(os.path.join(dirpath, "vectors.{}.p".format(self.name)), "wb") as f:
                pickle.dump(self, f)
            self.matrix = temp
        else:
            with open(os.path.join(dirpath, "vectors.{}.p".format(self.name)), "wb") as f:
                pickle.dump(self, f)

    def subset(self, ids: Optional[List[str]] = None, columns: Optional[List[str]] = None):
        """
        Get a (subset) copy of the ConvoKitMatrix object according to specified subset of ids and columns
        :param ids: list of ids to be included in the subset; all by default
        :param columns: list of columns to be included in the subset; all by default
        :return: a new ConvoKitMatrix object with the subset of
        """
        ids = ids if ids is not None else self.ids
        columns = columns if columns is not None else self.columns

        submatrix = self.to_dataframe().loc[ids][columns]
        return ConvoKitMatrix(
            name=self.name,
            matrix=csr_matrix(submatrix.values.astype("float64")),
            ids=ids,
            columns=columns,
        )

    @staticmethod
    def hstack(name: str, matrices: List["ConvoKitMatrix"]):
        """
        Combines multiple ConvoKitMatrices into a single ConvoKitMatrix by stacking them horizontally (i.e. each
        constituent matrix must have the same ids).

        :param name: name of new matrix
        :param matrices: constituent ConvoKiMatrices
        :return: a new ConvoKitMatrix
        """
        assert len(matrices) > 1
        stacked = hstack([csr_matrix(m.matrix) for m in matrices]).tocsr()
        columns = []
        for m in matrices:
            columns.extend(m.columns)

        return ConvoKitMatrix(name=name, matrix=stacked, ids=matrices[0].ids, columns=columns)

    @staticmethod
    def vstack(name: str, matrices: List["ConvoKitMatrix"]):
        """
        Combines multiple ConvoKitMatrices into a single ConvoKitMatrix by stacking them horizontally (i.e. each
        constituent matrix must have the same columns).

        :param name: name of new matrix
        :param matrices: constituent ConvoKiMatrices
        :return: a new ConvoKitMatrix
        """
        assert len(matrices) > 1
        stacked = vstack([csr_matrix(m.matrix) for m in matrices]).tocsr()
        ids = []
        for m in matrices:
            ids.extend(list(m.ids))

        return ConvoKitMatrix(name=name, matrix=stacked, ids=ids, columns=matrices[0].columns)

    def __repr__(self):
        return "ConvoKitMatrix('name': {}, 'matrix': {})".format(self.name, repr(self.matrix))

    def __str__(self):
        return repr(self)
