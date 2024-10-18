import itertools
import json
import os
from collections import defaultdict

from convokit.text_processing import TextProcessor


class PhrasingMotifs(TextProcessor):
    """
    A model that extracts a set of "phrasings" from a Corpus in the fit step, and that identifies which phrasings in this set are present in an utterance in the transform step. Phrasings intuitively correspond to frequently-occurring structures in the dependency trees of utterances, and are operationalized as frequently-occurring sets of dependency-parse arcs (though any other token-like input could work).

    The model expects as input utterances with a field consisting of either a string with space-separated tokens or arcs, or a list of such strings.

    As output in the transform step it produces:
        * a list, one per sentence, of space-separated phrasings contained in each sentence, where each phrasing is represented as a string where components (e.g., arcs) are separated by double underscores '__'.
        * a list of sink phrasings in each sentence -- the most finely-specified phrasing encapsulated by the sentence. (e.g., "do you agree..." is more finely-specified than "...agree...")

    Internally the model contains the following elements:
        * itemset_counts: a dictionary of phrasings to frequencies in the training data
        * downlinks: the graph structure representing the relationship between phrasings. used to later determine which phrasings are contained in a sentence in the transform step.
        * itemset_to_ids: maps phrasings to their de-duplicated forms.
        * min_support: the minimum frequency of a subset that is to be considered a phrasing

    :param output_field: name of attribute to write phrasings to in transform step. sink phrasings will be written to field <output_field>__sink.
    :param fit_field: name of attribute to use as input in fit.
    :param min_support: the minimum frequency of phrasings to return
    :param fit_filter:  a boolean function of signature `fit_filter(utterance)`. during the fit step phrasings will only be computed over utterances where `fit_filter` returns `True`. By default, will always return `True`, meaning that all utterances will be used.
    :param transform_field: name of attribute to use as input in transform; defaults to the same field used in fit.
    :param transform_filter: a boolean function of signature `transform_filter(utterance)`. during the transform step phrasings will only be computed over utterances where `transform_filter` returns `True`. defaults to filter used in fit step.
    :param deduplication_threshold: merges phrasings into a single phrasing if phrasings co-occur above this frequency (i.e., pr(phrasing 1 | phrasing 2) and vice versa)
    :param max_naive_itemset_size: maximum size of subsets to compute. above this size, a variant of the a-priori algorithm will be used in lieu of enumerating all possible subsets.
    :param max_itemset_size: maximum size of subsets to consider as phrasings. setting lower will run faster but miss more complex phrasings.
    :param verbosity: frequency of status messages.
    """

    def __init__(
        self,
        output_field,
        fit_field,
        min_support,
        fit_filter=None,
        transform_field=None,
        transform_filter=None,
        deduplication_threshold=0.9,
        max_naive_itemset_size=5,
        max_itemset_size=10,
        verbosity=0,
    ):
        self.min_support = min_support
        self.deduplication_threshold = deduplication_threshold
        self.max_naive_itemset_size = max_naive_itemset_size
        self.max_itemset_size = max_itemset_size

        self.fit_field = fit_field
        if fit_filter is None:
            self.fit_filter = lambda utt: True
        else:
            self.fit_filter = fit_filter
        if transform_field is None:
            transform_field = fit_field
        if transform_filter is None:
            transform_filter = fit_filter

        self.phrasing_motif_info = {
            "itemset_counts": {},
            "downlinks": {},
            "itemset_to_ids": {},
            "min_support": {},
        }
        TextProcessor.__init__(
            self,
            self._get_phrasing_motifs_wrapper,
            output_field=[output_field, output_field + "__sink"],
            input_field=transform_field,
            input_filter=transform_filter,
            aux_input=self.phrasing_motif_info,
            verbosity=verbosity,
        )

    def fit(self, corpus, y=None):
        """
        Fits a PhrasingMotifs model for a corpus -- that is, extracts all phrasings from the corpus.

        :param corpus: Corpus
        :return: None
        """

        arcset_dict = self._get_sent_arcset_dict(corpus)
        self.phrasing_motif_info = extract_phrasing_motifs(
            arcset_dict,
            self.min_support,
            self.deduplication_threshold,
            self.max_naive_itemset_size,
            self.max_itemset_size,
            self.verbosity,
        )

    def _get_phrasing_motifs_wrapper(self, arcs_per_sent, aux_input):
        return get_phrasing_motifs(arcs_per_sent, self.phrasing_motif_info)

    def _get_sent_arcset_dict(self, corpus):
        sent_dict = {}
        for utterance in corpus.iter_utterances():
            if self.fit_filter(utterance):
                for idx, sent in enumerate(utterance.retrieve_meta(self.input_field)):
                    sent_dict["%s__%d" % (utterance.id, idx)] = sent.split()
        return sent_dict

    def load_model(self, model_dir):
        """
        Loads a saved PhrasingMotifs model from disk.

        :param model_dir: directory to read model from.
        :return: None
        """

        if self.verbosity > 0:
            print("reading itemset counts")
        with open(os.path.join(model_dir, "itemset_counts.json")) as f:
            self.phrasing_motif_info["itemset_counts"] = {
                tuple(k.split("__")): v for k, v in json.load(f).items()
            }

        if self.verbosity > 0:
            print("reading downlinks")
        with open(os.path.join(model_dir, "downlinks.json")) as f:
            self.phrasing_motif_info["downlinks"] = {
                tuple(k.split("__")): set([tuple(x) for x in v]) for k, v in json.load(f).items()
            }
        if self.verbosity > 0:
            print("reading itemset to ids")
        with open(os.path.join(model_dir, "itemset_to_ids.json")) as f:
            self.phrasing_motif_info["itemset_to_ids"] = {
                tuple(k.split("__")): tuple(v.split("__")) for k, v in json.load(f).items()
            }
        if self.verbosity > 0:
            print("reading meta information")
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.phrasing_motif_info["min_support"] = json.load(f)["min_support"]

    def get_model(self):
        """
        Returns the PhrasingMotifs model. See class docstring for description of fields.

        :return: PhrasingMotifs model
        """
        return self.phrasing_motif_info

    def dump_model(self, model_dir):
        """
        Writes the model to disk.

        Will output one json file per model component.

        :param model_dir: directory to write to.
        :return: None
        """

        if self.verbosity > 0:
            print("writing itemset counts")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        with open(os.path.join(model_dir, "itemset_counts.json"), "w") as f:
            json.dump(
                {"__".join(k): v for k, v in self.phrasing_motif_info["itemset_counts"].items()}, f
            )

        if self.verbosity > 0:
            print("writing downlinks")
        with open(os.path.join(model_dir, "downlinks.json"), "w") as f:
            json.dump(
                {"__".join(k): sorted(v) for k, v in self.phrasing_motif_info["downlinks"].items()},
                f,
            )
        if self.verbosity > 0:
            print("writing itemset to ids")
        with open(os.path.join(model_dir, "itemset_to_ids.json"), "w") as f:
            json.dump(
                {
                    "__".join(k): "__".join(v)
                    for k, v in self.phrasing_motif_info["itemset_to_ids"].items()
                },
                f,
            )

        if self.verbosity > 0:
            print("writing meta information")
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump({"min_support": self.phrasing_motif_info["min_support"]}, f)

    def print_top_phrasings(self, k):
        """
        prints the k most frequent phrasings.

        :param k: number of phrasings to print
        :return: None
        """

        sorted_phrasings = sorted(
            self.phrasing_motif_info["itemset_counts"].items(),
            key=lambda x: (-x[1], len(x[0]), x[0]),
        )[:k]
        for phrasing, count in sorted_phrasings:
            print(phrasing, count)


def _print_output(i, verbosity):
    return (verbosity > 0) and (i > 0) and (i % verbosity == 0)


# fit utilities
def _get_sorted_combinations(itemset, k):
    combos = set()
    for set_ in itertools.combinations(itemset, k):
        combos.add(tuple(sorted(set_)))
    return combos


def _get_mini_powerset(itemset, k):
    powerset = set()
    for k in range(1, min(k + 1, len(itemset) + 1)):
        powerset.update(_get_sorted_combinations(itemset, k))
    return powerset


def _count_frequent_itemsets(
    set_dict, min_support, max_naive_itemset_size=5, max_itemset_size=100, verbosity=0
):
    if verbosity > 0:
        print("counting frequent itemsets for %d sets" % len(set_dict))
    itemset_counts = defaultdict(lambda: defaultdict(int))
    key_to_itemsets = defaultdict(lambda: defaultdict(set))

    if verbosity > 0:
        print(
            "\tfirst pass: counting itemsets up to and including %d items large"
            % max_naive_itemset_size
        )
    for idx, (key, set_) in enumerate(sorted(set_dict.items())):
        if _print_output(idx, verbosity):
            print("\tfirst pass: %03d/%03d sets processed" % (idx, len(set_dict)))
        for itemset in _get_mini_powerset(set_, max_naive_itemset_size):
            itemset_counts[len(itemset)][itemset] += 1
            key_to_itemsets[key][len(itemset)].add(itemset)

    for key, count_dicts in sorted(key_to_itemsets.items()):
        for i in range(1, max_naive_itemset_size + 1):
            count_dicts[i] = [
                itemset for itemset in count_dicts[i] if itemset_counts[i][itemset] >= min_support
            ]
    if max_naive_itemset_size >= max_itemset_size:
        return itemset_counts

    if verbosity > 0:
        print("\tsecond pass: counting itemsets more than %d items large" % max_naive_itemset_size)
    remaining_sets = [
        key
        for key, count_dicts in sorted(key_to_itemsets.items())
        if len(count_dicts[max_naive_itemset_size]) > 0
    ]
    itemset_size = max_naive_itemset_size + 1
    while (len(remaining_sets) > 0) and (itemset_size <= max_itemset_size):
        if verbosity > 0:
            print(
                "\tsecond pass: checking %d sets for itemsets of length %d"
                % (len(remaining_sets), itemset_size)
            )
        for idx, key in enumerate(remaining_sets):
            if _print_output(idx, verbosity):
                print(
                    "\tsecond pass: checked %03d/%03d sets for itemsets of length %d"
                    % (idx, len(remaining_sets), itemset_size)
                )
            set_ = set_dict[key]
            if len(set_) < itemset_size:
                continue

            new_itemsets = set()
            for entry in set_:
                if itemset_counts[1].get((entry,), 0) >= min_support:
                    for itemset in key_to_itemsets[key][itemset_size - 1]:
                        if entry not in itemset:
                            new_itemset = tuple(sorted(set(itemset + (entry,))))
                            new_itemsets.add(new_itemset)
            for new_itemset in new_itemsets:
                itemset_counts[itemset_size][new_itemset] += 1
                key_to_itemsets[key][itemset_size].add(new_itemset)
        for key, count_dicts in sorted(key_to_itemsets.items()):
            count_dicts[itemset_size] = [
                itemset
                for itemset in count_dicts[itemset_size]
                if itemset_counts[itemset_size][itemset] >= min_support
            ]
        remaining_sets = [
            key
            for key, count_dicts in sorted(key_to_itemsets.items())
            if len(count_dicts[itemset_size]) > 0
        ]
        itemset_size += 1

    unrolled_itemset_counts = {}
    for itemset_size, count_dict in sorted(itemset_counts.items()):
        for k, v in sorted(count_dict.items()):
            if v >= min_support:
                unrolled_itemset_counts[k] = v
    unrolled_itemset_counts[("*",)] = len(set_dict)

    unrolled_key_to_itemsets = {}
    for key, count_dicts in sorted(key_to_itemsets.items()):
        curr_unrolled = set()
        for _, count_dict in sorted(count_dicts.items()):
            curr_unrolled.update((k for k in count_dict if k in unrolled_itemset_counts))
        unrolled_key_to_itemsets[key] = sorted(curr_unrolled)
    return unrolled_itemset_counts, unrolled_key_to_itemsets


def _make_itemset_tree(itemset_counts, verbosity=0):
    if verbosity > 0:
        print("making itemset tree for %d itemsets" % len(itemset_counts))

    edges = []
    uplinks = {}
    downlinks = defaultdict(set)

    for itemset, count in sorted(itemset_counts.items()):
        parents = []
        itemset_size = len(itemset)
        if itemset_size == 1:
            item = itemset[0]
            # note this is the only part that is sort of dep-parse specific.
            if item.endswith("*"):
                parents.append(("*",))
            elif "_" in item:
                parents.append((item.split("_")[0] + "_*",))
            elif ">" in item:
                parents.append((item.split(">")[0] + ">*",))
        else:
            for idx in range(itemset_size):
                parents.append(itemset[:idx] + itemset[idx + 1 :])
        for parent in parents:
            parent_count = itemset_counts[parent]
            pr_child = count / itemset_counts[parent]
            edges.append({"child": itemset, "parent": parent})
            uplinks[itemset] = parent
            downlinks[parent].add(itemset)
    return uplinks, downlinks


def _deduplicate_itemsets(itemset_counts, itemset_collections, threshold, verbosity=20000):
    if verbosity > 0:
        print("deduplicating itemsets")
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))
    for idx, (key, itemsets) in enumerate(sorted(itemset_collections.items())):
        if _print_output(idx, verbosity):
            print(
                "\tcounting itemset cooccurrences for %03d/%03d collections"
                % (idx, len(itemset_collections))
            )
        itemset_list = list(itemsets)
        for i in range(len(itemset_list)):
            for j in range(i + 1, len(itemset_list)):
                cooccurrence_counts[itemset_list[i]][itemset_list[j]] += 1
                cooccurrence_counts[itemset_list[j]][itemset_list[i]] += 1
    if verbosity > 0:
        print("\tfinding supersets")
    superset_idx = 0
    supersets = defaultdict(set)
    itemset_to_superset = {}
    for idx, (itemset, count) in enumerate(sorted(itemset_counts.items())):
        if _print_output(idx, verbosity):
            print("\tgetting supersets for %03d/%03d itemsets" % (idx, len(itemset_counts)))
        if itemset in itemset_to_superset:
            continue
        itemset_to_superset[itemset] = superset_idx
        supersets[superset_idx].add(itemset)
        stack = [
            itemset2
            for itemset2, count2 in sorted(cooccurrence_counts.get(itemset, {}).items())
            if (count2 / count >= threshold) and (count2 / itemset_counts[itemset2] >= threshold)
        ]
        curr_stack = set(stack)
        while len(stack) > 0:
            neighbor = stack.pop()
            itemset_to_superset[neighbor] = superset_idx
            supersets[superset_idx].add(neighbor)
            neighbor_count = itemset_counts[neighbor]
            to_push = [
                itemset2
                for itemset2, count2 in sorted(cooccurrence_counts.get(neighbor, {}).items())
                if (count2 / neighbor_count >= threshold)
                and (count2 / itemset_counts[itemset2] >= threshold)
                and (itemset2 not in itemset_to_superset)
                and (itemset2 not in curr_stack)
            ]
            curr_stack.update(to_push)
            stack += to_push
        superset_idx += 1
    superset_ids = {}

    for idx, superset in sorted(supersets.items()):
        superset_ids[idx] = sorted(
            superset, key=lambda x: (itemset_counts[x], len(x)), reverse=True
        )[0]
    itemset_to_ids = {k: superset_ids[v] for k, v in sorted(itemset_to_superset.items())}
    supersets_by_id = {superset_ids[k]: list(v) for k, v in sorted(supersets.items())}
    return itemset_to_ids, supersets_by_id


def extract_phrasing_motifs(
    set_dict,
    min_support,
    deduplication_threshold=0.9,
    max_naive_itemset_size=5,
    max_itemset_size=10,
    verbosity=0,
):
    """
    standalone function to extract phrasings -- i.e., frequently-occurring collections of dependency-parse arcs (or other token-like objects).

    :param set_dict: dictionary mapping an ID (e.g., an utterance-sentence ID) to the collection of arcs in the corresponding object.
    :param min_support: minimum frequency of phrasings to return
    :param deduplication_threshold: merges phrasings into a single phrasing if phrasings co-occur above this frequency
    :param max_naive_itemset_size: maximum size of subsets to compute.
    :param max_itemset_size: maximum size of subsets to consider as phrasings.
    :param verbosity: frequency of status messages.

    :return: phrasing motifs model, as a dict containing each component.

    """

    # counts frequently co-occurring subsets
    itemset_counts, itemset_collections = _count_frequent_itemsets(
        set_dict, min_support, max_naive_itemset_size, max_itemset_size, verbosity
    )
    # induces a graph that relates subsets to each other
    uplinks, downlinks = _make_itemset_tree(itemset_counts, verbosity)
    # deduplicates frequently co-occurring subsets
    itemset_to_ids, supersets_by_id = _deduplicate_itemsets(
        itemset_counts, itemset_collections, deduplication_threshold, verbosity
    )
    return {
        "itemset_counts": itemset_counts,
        "downlinks": downlinks,
        "itemset_to_ids": itemset_to_ids,
        "min_support": min_support,
    }


# transform utilities


def _contains_candidate(container, candidate):
    return set(candidate).issubset(container)


def _get_itemset_collection(items, downlinks, itemset_counts, itemset_to_ids):
    items = sorted(set(items))
    fit_itemsets = {}
    itemset_stack = [(x,) for x in items if (x,) in itemset_counts]
    if len(itemset_stack) == 0:
        return {("*",): 0}
    i = 0
    while (len(itemset_stack) > 0) and (i < 1000):
        i += 1
        next_itemset = itemset_stack.pop()
        itemset_count = itemset_counts.get(next_itemset, None)
        if itemset_count:
            children = sorted(downlinks.get(next_itemset, []))
            valid_children = [
                child
                for child in children
                if _contains_candidate(items, child) and (child in itemset_counts)
            ]
            if len(valid_children) == 0:
                fit_itemsets[next_itemset] = 0
            else:
                fit_itemsets[next_itemset] = max(
                    itemset_counts.get(child, 0) for child in valid_children
                )
            itemset_stack += valid_children
    fit_supersets = defaultdict(list)
    for k, v in sorted(fit_itemsets.items()):
        fit_supersets[itemset_to_ids[k]].append(v)
    return {k: min(v) for k, v in sorted(fit_supersets.items())}


def get_phrasing_motifs(arcs_per_sent, phrasing_motif_info):
    """
    standalone function that returns phrasings and sink phrasings given an utterance (consisting of a list of space-separated arcs per each sentence) and a phrasing motif model.

    :param arcs_per_sent: input arcs per sentence
    :param phrasing_motif_info: phrasing motif model
    :return: phrasings and sink phrasings
    """

    phrasings = []
    sink_phrasings = []
    for sent in arcs_per_sent:
        result = _get_itemset_collection(
            sent.split(),
            phrasing_motif_info["downlinks"],
            phrasing_motif_info["itemset_counts"],
            phrasing_motif_info["itemset_to_ids"],
        )
        # phrasings.append({'__'.join(k): v  for k, v in result.items() if k != ('*',)})
        phrasings.append(" ".join(sorted("__".join(k) for k, v in result.items() if k != ("*",))))
        sink_phrasings.append(
            " ".join(
                sorted(
                    "__".join(k)
                    for k, v in result.items()
                    if (k != ("*",)) and (v < phrasing_motif_info["min_support"])
                )
            )
        )
    return phrasings, sink_phrasings
