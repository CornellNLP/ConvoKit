from typing import Callable
from .util import *
from collections import defaultdict, Counter
from random import shuffle, choice

from convokit.util import deprecation
from convokit import Transformer, CorpusComponent, Corpus

import numpy as np
import matplotlib.pyplot as plt


class Pairer(Transformer):
    """
    Pairer transformer sets up pairing to be used for paired prediction analyses.

    :param obj_type: type of Corpus object to classify: ‘conversation’, ‘speaker’, or ‘utterance’
    :param pairing_func: the Corpus object characteristic to pair on, e.g. to pair on the first 10 characters of a
        well-structured id, use lambda obj: obj.id[:10]
    :param pos_label_func: the function to check if the object is a positive instance
    :param neg_label_func: the function to check if the object is a negative instance
    :param pair_mode: 'random': pick a single positive and negative object pair randomly (default), 'maximize': pick the maximum number of positive and negative object pairs possible randomly, or 'first': pick the first positive and negative object pair found.
    :param pair_id_attribute_name: metadata attribute name to use in annotating object with pair id, default: "pair_id". The value is determined by the output of pairing_func. If pair_mode is 'maximize', the value is the output of pairing_func + "_[i]", where i is the ith pair extracted from a given context.
    :param label_attribute_name: metadata attribute name to use in annotating object with whether it is positive or negative, default: "pair_obj_label"
    :param pair_orientation_attribute_name: metadata attribute name to use in annotating object with pair orientation, default: "pair_orientation"
    """

    def __init__(
        self,
        obj_type: str,
        pairing_func: Callable[[CorpusComponent], str],
        pos_label_func: Callable[[CorpusComponent], bool],
        neg_label_func: Callable[[CorpusComponent], bool],
        pair_mode: str = "random",
        pair_id_attribute_name: str = "pair_id",
        pair_id_feat_name=None,
        label_attribute_name: str = "pair_obj_label",
        label_feat_name=None,
        pair_orientation_attribute_name: str = "pair_orientation",
        pair_orientation_feat_name=None,
    ):
        assert obj_type in ["speaker", "utterance", "conversation"]
        self.obj_type = obj_type
        self.pairing_func = pairing_func
        self.pos_label_func = pos_label_func
        self.neg_label_func = neg_label_func
        self.pair_mode = pair_mode
        self.pair_id_attribute_name = (
            pair_id_attribute_name if pair_id_feat_name is None else pair_id_feat_name
        )
        self.label_attribute_name = (
            label_attribute_name if label_feat_name is None else label_feat_name
        )
        self.pair_orientation_attribute_name = (
            pair_orientation_attribute_name
            if pair_orientation_feat_name is None
            else pair_orientation_feat_name
        )

        for deprecated_set in [
            (pair_id_feat_name, "pair_id_feat_name", "pair_id_attribute_name"),
            (label_feat_name, "label_feat_name", "label_attribute_name"),
            (
                pair_orientation_feat_name,
                "pair_orientation_feat_name",
                "pair_orientation_attribute_name",
            ),
        ]:
            if deprecated_set[0] is not None:
                deprecation(f"Pairer's {deprecated_set[1]} parameter", f"{deprecated_set[2]}")

    def _get_pos_neg_objects(self, corpus: Corpus, selector):
        """
        Get positively-labelled and negatively-labelled lists of objects
        :param corpus: target Corpus
        :return: list of positive objects, list of negative objects
        """
        pos_objects = []
        neg_objects = []
        for obj in corpus.iter_objs(self.obj_type, selector):
            if self.pos_label_func(obj):
                pos_objects.append(obj)
            elif self.neg_label_func(obj):
                neg_objects.append(obj)
        return pos_objects, neg_objects

    def _pair_objs(self, pos_objects, neg_objects):
        """
        Generate a dictionary mapping the Corpus object characteristic value (i.e. pairing_func's output) to one positively and negatively labelled object.
        :param pos_objects: list of positively labelled objects
        :param neg_objects: list of negatively labelled objects
        :return: dictionary indexed by the paired feature instance value,
                 with the value being a tuple (pos_obj, neg_obj)
        """
        pair_feat_to_pos_objs = defaultdict(list)
        pair_feat_to_neg_objs = defaultdict(list)

        for obj in pos_objects:
            pair_feat_to_pos_objs[self.pairing_func(obj)].append(obj)

        for obj in neg_objects:
            pair_feat_to_neg_objs[self.pairing_func(obj)].append(obj)

        valid_pairs = set(pair_feat_to_neg_objs).intersection(set(pair_feat_to_pos_objs))

        if self.pair_mode == "first":
            return {
                pair_id: (pair_feat_to_pos_objs[pair_id][0], pair_feat_to_neg_objs[pair_id][0])
                for pair_id in valid_pairs
            }
        elif self.pair_mode == "random":
            return {
                pair_id: (
                    choice(pair_feat_to_pos_objs[pair_id]),
                    choice(pair_feat_to_neg_objs[pair_id]),
                )
                for pair_id in valid_pairs
            }
        elif self.pair_mode == "maximize":
            retval = dict()
            for pair_id in valid_pairs:
                pos_objs = pair_feat_to_pos_objs[pair_id]
                neg_objs = pair_feat_to_neg_objs[pair_id]
                max_pairs = min(len(pos_objs), len(neg_objs))
                shuffle(pos_objs)
                shuffle(neg_objs)
                for idx in range(max_pairs):
                    retval[pair_id + "_" + str(idx)] = (pos_objs[idx], neg_objs[idx])
            return retval
        else:
            raise ValueError("Invalid pair_mode setting: use 'random', 'first', or 'maximize'.")

    @staticmethod
    def _assign_pair_orientations(obj_pairs):
        """
        Assigns the pair orientation (i.e. whether this pair will have a positive or negative label)
        :param obj_pairs: dictionary indexed by the paired feature instance value
        :return: dictionary of paired feature instance values to pair orientation value ('pos' or 'neg')
        """
        pair_ids = list(obj_pairs)
        shuffle(pair_ids)
        pair_orientations = dict()
        flip = True
        for pair_id in pair_ids:
            pair_orientations[pair_id] = "pos" if flip else "neg"
            flip = not flip
        return pair_orientations

    def transform(
        self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True
    ) -> Corpus:
        """
        Annotate corpus objects with pair information (label, pair_id, pair_orientation), with an optional selector indicating which objects should be considered for pairing.
        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns a bool (True = include)
        :return: annotated Corpus
        """
        pos_objs, neg_objs = self._get_pos_neg_objects(corpus, selector)
        obj_pairs = self._pair_objs(pos_objs, neg_objs)
        pair_orientations = self._assign_pair_orientations(obj_pairs)
        for pair_id, (pos_obj, neg_obj) in obj_pairs.items():
            pos_obj.add_meta(self.label_attribute_name, "pos")
            neg_obj.add_meta(self.label_attribute_name, "neg")
            pos_obj.add_meta(self.pair_id_attribute_name, pair_id)
            neg_obj.add_meta(self.pair_id_attribute_name, pair_id)
            pos_obj.add_meta(self.pair_orientation_attribute_name, pair_orientations[pair_id])
            neg_obj.add_meta(self.pair_orientation_attribute_name, pair_orientations[pair_id])

        for obj in corpus.iter_objs(self.obj_type):
            # unlabelled objects include both objects that did not pass the selector
            # and objects that were not selected in the pairing step
            if self.label_attribute_name not in obj.meta:
                obj.add_meta(self.label_attribute_name, None)
                obj.add_meta(self.pair_id_attribute_name, None)
                obj.add_meta(self.pair_orientation_attribute_name, None)

        return corpus

    def summarize(self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True, attributes=None, uniqueness_threshold=0.2, categorical_minperc=0):
        """
        Summarize and visualize meta-level information for pairs created by the Pairer using categorical or numerical plots for positive and negative classes
        :param corpus: target Cropus
        :param selector: a (lambda) function that takes a Corpus object and returns a bool (True = include)
        :param attributes: a parameter to provide meta attributes to be considered for summarization. By default (None) all valid attributes are compared;
        alternatively, desired attributes can be supplied in a list format (attribute names) or a dictionary format (where each attribute name is mapped
        to either 'categorical' or 'numerical' string).
        :param uniqueness_threshold: a parameter to determine whether attribute values are treated for categorical or numerical analyses. If the ratio
        (# unique values)/(# all values) of a metadata attribute is less than uniqueness_threshold, then categorical comparison is chosen.
        :param categorical_minperc: a threshold parameter to determine whether rare values of a metadata attribute are included in a categorical plot.

        :return: a schema with information on which meta attributes were analyzed, what types of data these attributes take,
        and whether a categorical or numercial plot was used for each attribute.
        """

        #summarize function intends to give a quick overview
        if self.obj_type == "speaker":
            meta_index = corpus.meta_index.to_dict()['speakers-index']
        if self.obj_type == "utterance":
            meta_index = corpus.meta_index.to_dict()['utterances-index']
        if self.obj_type == "conversation":
            meta_index = corpus.meta_index.to_dict()['conversations-index']

        UNIQUE_VAL_LIMIT = 30 # limit on the number of distinct categories plotted in categorical plot if uniqueness threshold is not met.
        simple_meta_value_types = ["<class 'int'>", "<class 'float'>", "<class 'str'>", "<class 'bool'>"]
        attributes_to_consider = {} # keeps track of the analysis schema
        values_to_plot = {} # keeps track of values to be plotted
        if attributes is None:
            # go across all simple meta attributes (i.e. string, integer, or float)
            for meta_name in meta_index:
                if len(meta_index[meta_name]) == 1 \
                    and meta_index[meta_name][0] in simple_meta_value_types \
                    and meta_name not in [self.label_attribute_name, self.pair_id_attribute_name, self.pair_orientation_attribute_name]:

                    pos_values = [obj.meta[meta_name] for obj in corpus.iter_objs(self.obj_type, selector=selector) if meta_name in obj.meta and obj.meta[self.label_attribute_name]=='pos']
                    neg_values= [obj.meta[meta_name] for obj in corpus.iter_objs(self.obj_type, selector=selector) if meta_name in obj.meta and obj.meta[self.label_attribute_name]=='neg']
                    total_value_count = len(pos_values)+len(neg_values)
                    unique_values = list(set(pos_values+neg_values))
                    uniqueness_factor  = len(unique_values)/total_value_count

                    if uniqueness_factor < uniqueness_threshold:
                        # for values that satisfy uniqueness threshold AND if the number of categories
                        # for plotting (i.e., categories that satisfy categorical_minperc) does not
                        # exceed UNIQUE_VAL_LIMIT, we proceed with categorical plot
                        pos_counts = Counter(pos_values)
                        neg_counts = Counter(neg_values)
                        categories_for_plotting = [c for c in unique_values if min(pos_counts[c],neg_counts[c]) >= categorical_minperc*total_value_count]
                        if len(categories_for_plotting) <= UNIQUE_VAL_LIMIT:
                            attributes_to_consider[meta_name] = {'type': meta_index[meta_name][0], 'category': 'categorical'}
                            values_to_plot[meta_name] = {'pos': pos_counts, 'neg': neg_counts}

                    elif meta_index[meta_name][0] in ["<class 'int'>", "<class 'float'>"]:
                        # for values that are of type integer or float proceed with numerical plot
                        attributes_to_consider[meta_name] = {'type': meta_index[meta_name][0], 'category': 'numerical'}
                        values_to_plot[meta_name] = {'pos': pos_values, 'neg': neg_values}

                    else:
                        # even if the uniqueness threshold is not satisfied by we have less categories
                        # for plotting (i.e. categories that satisfy categorical_minperc threshold)
                        # than UNIQUE_VAL_LIMIT, we proceed with categorical plot
                        pos_counts = Counter(pos_values)
                        neg_counts = Counter(neg_values)
                        categories_for_plotting = [c for c in unique_values if min(pos_counts[c],neg_counts[c]) >= categorical_minperc*total_value_count]

                        if len(categories_for_plotting) <= UNIQUE_VAL_LIMIT:
                            attributes_to_consider[meta_name] = {'type': meta_index[meta_name][0], 'category': 'categorical'}
                            values_to_plot[meta_name] = {'pos': pos_counts, 'neg': neg_counts}


        elif type(attributes) == list:
            # identify which attribute is of what event_types
            for meta_name in attributes:
                if meta_name in meta_index and len(meta_index[meta_name]) == 1 and meta_index[meta_name][0] in simple_meta_value_types:

                    pos_values = [obj.meta[meta_name] for obj in corpus.iter_objs(self.obj_type, selector=selector) if meta_name in obj.meta and obj.meta[self.label_attribute_name]=='pos']
                    neg_values= [obj.meta[meta_name] for obj in corpus.iter_objs(self.obj_type, selector=selector) if meta_name in obj.meta and obj.meta[self.label_attribute_name]=='neg']
                    total_value_count = len(pos_values)+len(neg_values)
                    unique_values = list(set(pos_values+neg_values))
                    uniqueness_factor  = len(unique_values)/total_value_count

                    if uniqueness_factor < uniqueness_threshold:
                        # for values that satisfy uniqueness threshold AND if the number of categories
                        # for plotting (i.e., categories that satisfy categorical_minperc) does not
                        # exceed UNIQUE_VAL_LIMIT, we proceed with categorical plot
                        pos_counts = Counter(pos_values)
                        neg_counts = Counter(neg_values)
                        categories_for_plotting = [c for c in unique_values if min(pos_counts[c],neg_counts[c]) >= categorical_minperc*total_value_count]
                        if len(categories_for_plotting) <= UNIQUE_VAL_LIMIT:
                            attributes_to_consider[meta_name] = {'type': meta_index[meta_name][0], 'category': 'categorical'}
                            values_to_plot[meta_name] = {'pos': pos_counts, 'neg': neg_counts}
                        else:
                            raise ValueError('Attribute {} has too many unique categories for plotting: {} exceeds UNIQUE_VAL_LIMIT=30. Adjust categorical_minperc to reduce the number of categories.'.format(meta_name, len(categories_for_plotting)))

                    elif meta_index[meta_name][0] in ["<class 'int'>", "<class 'float'>"]:
                        # for values that are of type integer or float proceed with numerical plot
                        attributes_to_consider[meta_name] = {'type': meta_index[meta_name][0], 'category': 'numerical'}
                        values_to_plot[meta_name] = {'pos': pos_values, 'neg': neg_values}

                    else:
                        # for all other values, check how many categories have counts above categorical_minperc
                        # if this number of categories is less than UNIQUE_VAL_LIMIT, then we can plot categorial
                        pos_counts = Counter(pos_values)
                        neg_counts = Counter(neg_values)
                        categories_for_plotting = [c for c in unique_values if min(pos_counts[c],neg_counts[c]) >= categorical_minperc*total_value_count]
                        if len(categories_for_plotting) <= UNIQUE_VAL_LIMIT:
                            attributes_to_consider[meta_name] = {'type': meta_index[meta_name][0], 'category': 'categorical'}
                            values_to_plot[meta_name] = {'pos': pos_counts, 'neg': neg_counts}
                        else:
                            raise ValueError('Attribute {} has too many unique categories for plotting: {} exceeds UNIQUE_VAL_LIMIT=30. Adjust categorical_minperc to reduce the number of categories.'.format(meta_name, len(categories_for_plotting)))

                elif meta_name not in meta_index:
                    raise ValueError('Attribute {} is not part of {} corpus object metadata.'.format(meta_name, self.obj_type))

                elif len(meta_index[meta_name]) != 1:
                    raise ValueError('Attribute {} does not have consistent value types: {}.'.format(meta_name, meta_index[meta_name]))

                else:
                    raise ValueError('Attribute {} has value type of {}, while simple value type is expected: {}.'.format(meta_name, meta_index[meta_name], simple_meta_value_types))

        elif type(attributes) == dict:
            for meta_name in attributes:
                if meta_name in meta_index and len(meta_index[meta_name]) == 1 and meta_index[meta_name][0] in simple_meta_value_types:

                    pos_values = [obj.meta[meta_name] for obj in corpus.iter_objs(self.obj_type, selector=selector) if meta_name in obj.meta and obj.meta[self.label_attribute_name]=='pos']
                    neg_values= [obj.meta[meta_name] for obj in corpus.iter_objs(self.obj_type, selector=selector) if meta_name in obj.meta and obj.meta[self.label_attribute_name]=='neg']
                    total_value_count = len(pos_values)+len(neg_values)
                    unique_values = list(set(pos_values+neg_values))
                    uniqueness_factor  = len(unique_values)/total_value_count
                    desired_category = attributes[meta_name]

                    assert desired_category in ['numerical', 'categorical']
                    if desired_category == 'numerical':
                        # for values that are of type integer or float proceed with numerical plot
                        if meta_index[meta_name][0] in ["<class 'int'>", "<class 'float'>"]:
                            attributes_to_consider[meta_name] = {'type': meta_index[meta_name][0], 'category': 'numerical'}
                            values_to_plot[meta_name] = {'pos': pos_values, 'neg': neg_values}
                        else:
                            raise ValueError('Attribute {} is of type {} while <class \'int\'> or <class \'float\'> are expected for numerical summary.'.format(meta_name, meta_index[meta_name][0]))

                    elif desired_category == 'categorical':
                        # if the number of categories for plotting does not exceed
                        # UNIQUE_VAL_LIMIT proceed with categorical plot
                        pos_counts = Counter(pos_values)
                        neg_counts = Counter(neg_values)
                        categories_for_plotting = [c for c in unique_values if min(pos_counts[c],neg_counts[c]) >= categorical_minperc*total_value_count]
                        if len(categories_for_plotting) <= UNIQUE_VAL_LIMIT:
                            attributes_to_consider[meta_name] = {'type': meta_index[meta_name][0], 'category': 'categorical'}
                            values_to_plot[meta_name] = {'pos': pos_counts, 'neg': neg_counts}
                        else:
                            raise ValueError('Attribute {} has too many unique categories for plotting: {} exceeds UNIQUE_VAL_LIMIT=30. Adjust categorical_minperc to reduce the number of categories.'.format(meta_name, len(categories_for_plotting)))

                elif meta_name not in meta_index:
                    raise ValueError('Attribute {} is not part of {} corpus object metadata.'.format(meta_name, self.obj_type))

                elif len(meta_index[meta_name]) != 1:
                    raise ValueError('Attribute {} does not have consistent value types: {}.'.format(meta_name, meta_index[meta_name]))

                else:
                    raise ValueError('Attribute {} has value type of {}, while simple value type is expected: {}.'.format(meta_name, meta_index[meta_name], simple_meta_value_types))

        else:
            raise ValueError('Value of type <class \'list\'> or <class \'dict\'> is expected for attributes parameter, but value of type {} was provided.'.format(type(attributes)))

        # plot comparisons of relevant metadata attributes
        pos_class_name = "{}='pos'".format(self.label_attribute_name)
        neg_class_name = "{}='neg'".format(self.label_attribute_name)
        for meta_name in attributes_to_consider:
            if attributes_to_consider[meta_name]['category'] == 'categorical':
                plot_categorical_comparison(values_to_plot[meta_name]['pos'], values_to_plot[meta_name]['neg'], meta_name, pos_class_name, neg_class_name, minperc=categorical_minperc)
            else:
                plot_numerical_comparison(values_to_plot[meta_name]['pos'], values_to_plot[meta_name]['neg'], meta_name, pos_class_name, neg_class_name)
                attributes_to_consider[meta_name]['numerical_stats'] = (np.mean(values_to_plot[meta_name]['pos']), np.mean(values_to_plot[meta_name]['neg'])),\
                                                                        (np.std(values_to_plot[meta_name]['pos']), np.std(values_to_plot[meta_name]['neg']))

        return attributes_to_consider


def plot_categorical_comparison(pos_counts, neg_counts, attr_name, pos_class_name='pos_class', neg_class_name='neg_class', minperc=0):
    total_pos_count = 1 if sum(pos_counts.values())==0 else sum(pos_counts.values())
    total_neg_count = 1 if sum(neg_counts.values())==0 else sum(neg_counts.values())
    sorted_x = sorted(list(set(list(pos_counts.keys()) + list(neg_counts.keys()))),
                      key=lambda k: (pos_counts[k]+neg_counts[k], k))
    x_to_plot = [k for k in sorted_x if min(pos_counts[k]/total_pos_count,
                                            neg_counts[k]/total_neg_count) >= minperc]
    bar_width=0.3
    plt.bar(range(len(x_to_plot)), [pos_counts[k] for k in x_to_plot], align='center', width=bar_width, color ='#d62728', label=pos_class_name)
    plt.bar([x+bar_width for x in range(len(x_to_plot))], [neg_counts[k] for k in x_to_plot], align='center', width=bar_width, color ='#1f77b4', label=neg_class_name)
    plt.xticks([x+bar_width/2 for x in range(len(x_to_plot))], x_to_plot, rotation=90)
    plt.legend()
    plt.title('Attribute: {}'.format(attr_name))
    plt.show()


def plot_numerical_comparison(pos_values, neg_values, attr_name, pos_class_name='pos_class', neg_class_name='neg_class'):
    bar_width=0.3
    violin_parts = plt.violinplot([pos_values,neg_values],
                                  showmeans=True,
                                  showextrema=True,
                                  showmedians=True,
                                  widths=bar_width)
    violin_parts['bodies'][0].set_color('red')
    violin_parts['bodies'][1].set_color('blue')
    for l in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
        violin_parts[l].set_color('grey')
    plt.xticks([1,2], [pos_class_name,neg_class_name])
    plt.title('Attribute: {}'.format(attr_name))
    plt.show()
