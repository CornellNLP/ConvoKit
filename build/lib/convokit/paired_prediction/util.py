from random import shuffle
from pandas import DataFrame
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from convokit.classifier.util import extract_feats_from_obj


def generate_vectors_paired_X_y(
    corpus, vector_name, pair_orientation_attribute_name, pair_id_to_objs
):
    """
    Generate the X, y matrix for paired prediction and annotate the objects with the pair orientation.

    :param corpus:
    :param vector_name:
    :param pair_orientation_attribute_name:
    :param pair_id_to_objs:
    :return:
    """
    pos_orientation_pair_ids = []
    neg_orientation_pair_ids = []

    for pair_id, (pos_obj, neg_obj) in pair_id_to_objs.items():
        if pos_obj.meta[pair_orientation_attribute_name] == "pos":
            pos_orientation_pair_ids.append(pair_id)
        else:
            neg_orientation_pair_ids.append(pair_id)

    pos_orientation_pos_objs, pos_orientation_neg_objs = zip(
        *[pair_id_to_objs[pair_id] for pair_id in pos_orientation_pair_ids]
    )

    neg_orientation_pos_objs, neg_orientation_neg_objs = zip(
        *[pair_id_to_objs[pair_id] for pair_id in neg_orientation_pair_ids]
    )

    pos_pos_ids = [obj.id for obj in pos_orientation_pos_objs]
    pos_neg_ids = [obj.id for obj in pos_orientation_neg_objs]

    neg_pos_ids = [obj.id for obj in neg_orientation_pos_objs]
    neg_neg_ids = [obj.id for obj in neg_orientation_neg_objs]

    pos_pos_vectors = corpus.get_vectors(vector_name, pos_pos_ids)
    pos_neg_vectors = corpus.get_vectors(vector_name, pos_neg_ids)

    neg_pos_vectors = corpus.get_vectors(vector_name, neg_pos_ids)
    neg_neg_vectors = corpus.get_vectors(vector_name, neg_neg_ids)

    y = np.array([1] * len(pos_orientation_pair_ids) + [0] * len(neg_orientation_pair_ids))

    if issparse(pos_pos_vectors):  # for csr_matrix
        X = vstack([pos_pos_vectors - pos_neg_vectors, neg_neg_vectors - neg_pos_vectors])
    else:
        X = np.vstack([pos_pos_vectors - pos_neg_vectors, neg_neg_vectors - neg_pos_vectors])

    indices = np.arange(X.shape[0])
    shuffle(indices)
    return X[indices], y[indices]


def generate_paired_X_y(pred_feats, pair_orientation_attribute_name, pair_id_to_objs):
    """
    Generate the X, y matrix for paired prediction
    :param pair_id_to_objs: dictionary indexed by the paired feature instance value, with the value
    being a tuple (pos_obj, neg_obj)
    :return: X, y matrix representing the predictive features and labels respectively
    """
    pos_obj_dict = dict()
    neg_obj_dict = dict()
    for pair_id, (pos_obj, neg_obj) in pair_id_to_objs.items():
        pos_obj_dict[pair_id] = extract_feats_from_obj(pos_obj, pred_feats)
        neg_obj_dict[pair_id] = extract_feats_from_obj(neg_obj, pred_feats)
    pos_obj_df = DataFrame.from_dict(pos_obj_dict, orient="index")
    neg_obj_df = DataFrame.from_dict(neg_obj_dict, orient="index")

    X, y = [], []
    pair_ids = list(pair_id_to_objs)
    shuffle(pair_ids)
    for pair_id in pair_ids:
        pos_feats = np.array(pos_obj_df.loc[pair_id]).astype("float64")
        neg_feats = np.array(neg_obj_df.loc[pair_id]).astype("float64")
        orientation = pair_id_to_objs[pair_id][0].meta[pair_orientation_attribute_name]

        assert orientation in ["pos", "neg"]

        if orientation == "pos":
            y.append(1)
            diff = pos_feats - neg_feats
        else:
            y.append(0)
            diff = neg_feats - pos_feats

        X.append(diff)

    return csr_matrix(np.array(X)), np.array(y)


def generate_pair_id_to_objs(
    corpus,
    obj_type,
    selector,
    pair_orientation_attribute_name,
    label_attribute_name,
    pair_id_attribute_name,
):
    pair_id_to_obj = {"pos": dict(), "neg": dict()}
    for obj in corpus.iter_objs(obj_type, selector):
        if obj.meta[pair_orientation_attribute_name] is None:
            continue
        pair_id_to_obj[obj.meta[label_attribute_name]][obj.meta[pair_id_attribute_name]] = obj

    valid_pair_ids = set(pair_id_to_obj["pos"].keys()).intersection(
        set(pair_id_to_obj["neg"].keys())
    )

    # print(set(pair_id_to_obj['pos'].keys()))
    print("Found {} valid pairs.".format(len(valid_pair_ids)))
    pair_id_to_objs = dict()
    for pair_id in valid_pair_ids:
        pair_id_to_objs[pair_id] = (pair_id_to_obj["pos"][pair_id], pair_id_to_obj["neg"][pair_id])
    return pair_id_to_objs
