"""QuestionTypology features
(http://www.cs.cornell.edu/~cristian/Asking_too_much.html)."""

import itertools
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from ast import literal_eval as make_tuple
from collections import defaultdict
from scipy import sparse
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import Normalizer
from spacy.en import English
from spacy.symbols import *
from spacy.tokens.doc import Doc

NP_LABELS = set([nsubj, nsubjpass, dobj, iobj, pobj, attr])

class QuestionTypology:
    """Encapsulates computation of question types from a question-answer corpus.

    :param corpus: the corpus to compute types for.
    :type corpus: Corpus
    :param data_dir: the directory that the data is stored in, and written to
    :param motifs_dir: the directory that the motifs are stored in, if they have been precomputed
    :param num_clusters: the number of question types to find in the clustering algorithm
    :param dataset_name: parliament or tennis

    :ivar corpus: the QuestionTypology object's corpus.
    :ivar data_dir: the directory that the data is stored in, and written to
    :ivar motifs_dir: the directory that the motifs are stored in, if they have been precomputed
    :ivar num_clusters: the number of question types to find in the clustering algorithm
    :ivar mtx_obj: an object that contains information about the QA matrix from the paper
    :ivar km: the Kmeans object that has the labels
    :ivar types_to_data: an object that contains information about motifs, fragments and questions in each type
    :ivar lq: the low dimensional Q matrix
    :ivar a_u: the low dimensional A matrix
    """

    def __init__(self, corpus, data_dir, motifs_dir=None, num_clusters=8, dataset_name="parliament"):
        self.corpus = corpus
        self.data_dir = data_dir
        self.motifs_dir = motifs_dir
        self.num_clusters = num_clusters

        if not self.motifs_dir:
            self.motifs_dir = os.path.join(self.data_dir, dataset_name+'-motifs')
            spacy_file = os.path.join(self.data_dir, 'spacy')
            MotifsExtractor.spacify(self.corpus.iterate_by('both'), spacy_file)
            MotifsExtractor.extract_question_motifs(self.corpus.iterate_by('questions'), spacy_file, self.motifs_dir)
            MotifsExtractor.extract_answer_arcs(self.corpus.iterate_by('answers'), spacy_file, self.motifs_dir)

        self.matrix_dir = os.path.join(self.data_dir, dataset_name+'-matrix')

        QuestionClusterer.build_matrix(self.motifs_dir, self.matrix_dir, question_threshold=50, answer_threshold=50)
        self.km_name = os.path.join(self.data_dir, 'demo_km.pkl')
        self.mtx_obj, self.km, self.types_to_data, self.lq, self.a_u = QuestionClusterer.extract_clusters(self.matrix_dir, self.km_name, k=num_clusters,d=100,num_egs=10)

        self.motif_df, self.aarc_df, self.qdoc_df = QuestionClusterer.assign_clusters(self.km, self.lq, self.a_u, self.mtx_obj, 100)
        for index, row in self.qdoc_df.iterrows():
            cluster = row["cluster"]
            cluster_dist = row["cluster_dist"]
            q_idx = row["q_idx"]
            self.types_to_data[cluster]["questions"].append(q_idx)
            self.types_to_data[cluster]["question_dists"].append(cluster_dist)

    def _get_question_text_from_pair_idx(self, pair_idx):
        for q in self.corpus.utterances.values():
            # print(pair_idx, q.other["pair_idx"], q.other["is_question"])
            if q.other["pair_idx"] == pair_idx and q.other["is_question"]:
                return q.text
        return "No question found"


    def display_question_types(self):
        pass

    def display_questions_for_type(self, type_num, num_egs=5):
        "Displays num_egs number of questions that were assigned type type_num in the cluster"
        target = self.types_to_data[type_num]
        questions = target["questions"]
        questions_len = len(questions)
        num_to_print = min(questions_len, num_egs)
        print('\t%d sample questions that were assigned type %d (%d total questions with this type) :'%(num_to_print, type_num, questions_len))
        for i in range(num_to_print):
            print('\t\t%d.'%(i+1), self._get_question_text_from_pair_idx(questions[i]))

    def display_motifs_for_type(self, type_num, num_egs=5):
        "Displays num_egs number of motifs that were assigned type type_num in the cluster"
        target = self.types_to_data[type_num]
        motifs = target["motifs"]
        motifs_len = len(motifs)
        num_to_print = min(motifs_len, num_egs)
        print('\t%d sample question motifs for type %d (%d total motifs):'%(num_to_print, type_num, motifs_len))
        for i in range(num_to_print):
            print('\t\t%d.'%(i+1), motifs[i])

    def display_answer_fragments_for_type(self, type_num, num_egs=5):
        "Displays num_egs number of answer fragments whose corresponding question motif were assigned type type_num in the cluster"
        target = self.types_to_data[type_num]
        answer_fragments = target["fragments"]
        fragment_len = len(answer_fragments)
        num_to_print = min(fragment_len, num_egs)
        print('\t%d sample answer fragments for type %d (%d total fragments) :'%(num_to_print, type_num, fragment_len))
        for i in range(num_to_print):
            print('\t\t%d.'%(i+1), answer_fragments[i])


    def display_question_type_log_odds_graph(self):
        clusters = [i for i in range(self.num_clusters, 0, -1)]

        num_questions_govt = [0 for i in range(self.num_clusters)]
        num_questions_opp = [0 for i in range(self.num_clusters)]

        for q in self.corpus.utterances.values():
            user_info = q.user._get_info()
            if q.other["is_question"]:
                if "is_minister" not in user_info or not user_info["is_minister"]: continue
                pair_idx = q.other["pair_idx"]
                for i in range(self.num_clusters):
                    if pair_idx in self.types_to_data[i]["questions"]:
                        if user_info["is_oppn"]:
                            num_questions_opp[i] += 1
                        else:
                            num_questions_govt[i] += 1

        govt_log_odds = [np.log(num_questions_govt[i]/len(self.types_to_data[i]["questions"])) - 
            np.log((len(self.types_to_data[i]["questions"]) - num_questions_govt[i])/len(self.types_to_data[i]["questions"])) 
            for i in range(self.num_clusters)]
        opp_log_odds = [np.log(num_questions_opp[i]/len(self.types_to_data[i]["questions"])) - 
            np.log((len(self.types_to_data[i]["questions"]) - num_questions_opp[i])/len(self.types_to_data[i]["questions"])) 
            for i in range(self.num_clusters)]

        govt_data_style = 'bo'
        opp_data_style = 'rs'

        line_x = np.linspace(-1.2, 1.2, self.num_clusters) #for lines

        labels = ['0. issue update', '1. shared concerns', '2. narrow factual', 
        '3. prompt for comment', '4. agreement', '5. self-promotion', 
        '6. concede/accept', '7. condemnatory']
        #plot lines - probably a better way of doing this
        for i in clusters:
            y_i = np.full(self.num_clusters,i)
            plt.plot(line_x, y_i, linestyle='dashed', linewidth=1, color='lightgrey')

        #plot govt
        govt_plot, = plt.plot(govt_log_odds, clusters, govt_data_style, label='government affiliated')

        #plot opposition
        opp_plot, = plt.plot(opp_log_odds, clusters, opp_data_style, label='opposition affiliated')

        #legend
        plt.legend(handles=[govt_plot, opp_plot], loc='lower right')

        #add labels
        plt.yticks(clusters, labels, rotation='horizontal')

        #add central line
        plt.axvline(x=0, color='black')

        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        plt.show() #This can be changed to show or write to file
        # print(govt_log_odds)
        # print(opp_log_odds)

    def display_mean_propensities_graph(self):
        pass
        # B: Mean propensities for each question type, for MPs who switch from
        #being in the opposition to being in the government (top) and vice-versa (bottom)
        #after an election. Stars indicate statistically significant differences at the
        #p < 0.05 (*), p < 0.01 (**) and p < 0.001 (***) levels (Wilcoxon test).

    def classify_question(self, question_answer_pair):
        #for now only takes in a question answer pair given as a list of 2 json dicts

        #create spacy objects
        spacy_NLP = spacy.load('en')
        vocab = MotifsExtractor.load_vocab()
        spacy_q_obj = Doc(vocab).from_bytes(spacy_NLP(question_answer_pair[0]['text']).to_bytes())
        spacy_a_obj = Doc(vocab).from_bytes(spacy_NLP(question_answer_pair[1]['text']).to_bytes())

        #extract question fragments
        for span_idx, span in enumerate(spacy_q_obj.sents):
            curr_arcset = MotifsExtractor.get_arcs(span.root, True)
            fragments = list(curr_arcset)
        print(fragments)

        #extract answer fragments
        for span_idx, span in enumerate(spacy_a_obj.sents):
            curr_arcset = MotifsExtractor.get_arcs(span.root, True)
            answer_fragments = list(curr_arcset)
        print(answer_fragments)

        #extract motifs
        question_tree_outfile = os.path.join(self.motifs_dir, 'question_tree')
        downlinks = MotifsExtractor.read_downlinks(question_tree_outfile + '_downlinks.json')
        node_counts = MotifsExtractor.read_nodecounts(question_tree_outfile + '_arc_set_counts.tsv')
        fit_nodes = MotifsExtractor.fit_question(set(fragments), downlinks, node_counts)

        #build vec for this question
        superset_file = os.path.join(self.motifs_dir, 'question_supersets_arcset_to_super.json')
        question_threshold = 1
        answer_threshold = 1

        question_to_fits = defaultdict(set)
        question_to_leaf_fits = defaultdict(set)
        motif_counts = defaultdict(int)


        super_mappings = {}
        with open(superset_file) as f:
            for line in f.readlines():
                entry = json.loads(line)
                super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])

        i = 0
        for entry in fit_nodes.values():
            motif = tuple(entry['arcset'])
            super_motif = super_mappings[motif]
            if entry['arcset_count'] < question_threshold: continue
            if entry['max_valid_child_count'] < question_threshold:
                question_to_leaf_fits[i].add(super_motif)
                question_to_fits[i].add(super_motif)
                motif_counts[super_motif] += 1
            i += 1

        question_to_fits = {k: [x for x in v if motif_counts[x] >= question_threshold] for k,v in question_to_fits.items()}
        motif_counts = {k:v for k,v in motif_counts.items() if v >= question_threshold}
        question_to_leaf_fits = {k: [x for x in v if motif_counts.get(x,0) >= question_threshold] for k,v in question_to_leaf_fits.items()}

        question_to_arcs = defaultdict(set)
        arc_counts = defaultdict(int)
        
        question_to_arcs[0].update(answer_fragments)
        for arc in answer_fragments:
            arc_counts[arc] += 1
        question_to_arcs = {k: [x for x in v if arc_counts[x] >= answer_threshold] for k,v in question_to_arcs.items()}
        arc_counts = {k:v for k,v in arc_counts.items() if v >= answer_threshold}


        question_term_list = list(motif_counts.keys())
        answer_term_list = list(arc_counts.keys())

        question_term_to_idx = {k:idx for idx,k in enumerate(question_term_list)}
        answer_term_to_idx = {k:idx for idx,k in enumerate(answer_term_list)}

        question_term_idxes = []
        question_leaves = []
        question_doc_idxes = []
        pair_idx_list = []
        answer_term_idxes = []
        answer_doc_idxes = []

        pair_idxes = list(set(question_to_fits.keys()).intersection(set(question_to_arcs.keys())))

        for idx, p_idx in enumerate(pair_idxes):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)

            question_terms = question_to_fits[p_idx]
            answer_terms = question_to_arcs[p_idx]
            pair_idx_list.append(p_idx)

            for term in question_terms:
                term_idx = question_term_to_idx[term]
                question_term_idxes.append(term_idx)
                question_doc_idxes.append(idx)
                question_leaves.append(term in question_to_leaf_fits.get(p_idx,[]))
            for term in answer_terms:
                term_idx = answer_term_to_idx[term]
                answer_term_idxes.append(term_idx)
                answer_doc_idxes.append(idx)


        # create mtx_obj
        mtx_obj = {}
        mtx_obj['p_idxes'] = pair_idx_list
        mtx_obj['q_tidxes'] = question_term_idxes
        mtx_obj['q_leaves'] = question_leaves
        mtx_obj['a_tidxes'] = answer_term_idxes
        mtx_obj['q_didxes'] = question_doc_idxes
        mtx_obj['a_didxes'] = answer_doc_idxes
        mtx_obj['q_terms'] = question_term_list
        mtx_obj['a_terms'] = answer_term_list
        mtx_obj['q_term_counts'] = [motif_counts[term] for term in question_term_list]
        mtx_obj['a_term_counts'] = [arc_counts[term] for term in answer_term_list]


        #create q_mtx
        # N_terms = len(mtx_obj['q_terms'])
        # data = np.ones(len(mtx_obj['q_terms']))
        #Assume not idf
        # Assume leaves only
        # data[~mtx_obj['q_leaves']] = 0
        # q_mtx = sparse.csr_matrix((data, ([0], [0])), shape=(N_terms,1))
        # norm = 'l2'
        # if norm:
        #     q_mtx = Normalizer(norm=norm).fit_transform(q_mtx)


        # #create a_mtx
        # N_terms = len(mtx_obj['a_terms'])
        # data = np.ones(len(mtx_obj['a_terms']))
        # #Assume not idf
        # # Assume leaves only
        # # data[~mtx_obj['a_leaves']] = 0
        # print(data)
        # a_mtx = sparse.csr_matrix((data, ([0], np.zeros_like(data))), shape=(N_terms,1))
        # if norm:
        #     a_mtx = Normalizer(norm=norm).fit_transform(a_mtx)



        # #get lowdim mtx
        # #what value of k? 51? Assume always snip
        # a_u, a_s, a_v = QuestionClusterer.do_sparse_svd(a_mtx, 51)
        # lq = q_mtx * (a_v.T * a_s**-1)
        # print(lq, a_u, a_s, a_v)
        # # km, _ = QuestionClusterer.inspect_kmeans_run(lq,a_u,d,k,mtx_obj['q_terms'], mtx_obj['a_terms'], num_egs=num_egs)




class MotifsExtractor:
    def load_vocab(verbose=False):
        if verbose:
            print('loading spacy vocab')
        return English().vocab

    def iterate_spacy(path, vocab):
        with open(path + '.bin', 'rb') as spacy_file:
            with open(path + '.txt') as key_file:
                for doc_bytes in Doc.read_bytes(spacy_file):
                    try:
                        key = next(key_file)
                        doc = Doc(vocab).from_bytes(doc_bytes)
                        yield key.strip(), doc
                    except:
                        continue

    def get_spacy_dict(path, vocab=None, verbose=5000):
        '''
            gets a dict of (key --> spacy object) from a path (as generated by the spacify function).
            can pass pre-loaded vocabulary to avoid the terrible load time.

            currently this is super-slow anyways, probably because it's reading in the entire dataset.
            in the ideal world, the dataset would be stored in separate chunks, and we could read in parallel.
        '''
        if not vocab:
            vocab = MotifsExtractor.load_vocab(verbose)
        spacy_dict = {}
        iterable_docs = enumerate(MotifsExtractor.iterate_spacy(path,vocab))
        for idx, (key, doc) in iterable_docs:
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            spacy_dict[key] = doc
        return spacy_dict

    def spacify(text_iter, outfile_name, spacy_NLP=None, verbose=5000):
        '''
            spacifies, writes a spacy object = file w/ spacy objects + other files w/ keys to said objects
            text_iter: iterates over text to spacify, yielding index and text
            outfile_name: where to write the spacy file. will write outfile_name.bin, outfile_name.txt
            if you don't want to keep loading spacy NLP objects (which takes a while) then can
                pass an existing spacy_NLP.
        '''
        if not spacy_NLP:
            if verbose:
                print('loading spacy NLP')
            spacy_NLP = spacy.load('en')
        spacy_keys = []
        spacy_objs = []
        for idx,(text_idx, text, pair_idx) in enumerate(text_iter):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            spacy_keys.append(text_idx)
            spacy_objs.append(spacy_NLP(text).to_bytes())
        with open(outfile_name + '.bin','wb') as f:
            [f.write(byte_val) for byte_val in spacy_objs]
        with open(outfile_name + '.txt','w') as f:
            f.write('\n'.join(spacy_keys))

    def deduplicate_motifs(question_fit_file, outfile, threshold=.9, verbose=5000):

        if verbose:
            print('\treading raw fits')
        span_to_fits = defaultdict(set)
        arcset_counts = defaultdict(int)
        with open(question_fit_file) as f:
            for idx,line in enumerate(f.readlines()):
                if verbose and (idx > 0) and (idx % verbose == 0):
                    print('\t%03d' % idx)
                entry = json.loads(line)
                span_to_fits[entry['span_idx']].add(tuple(entry['arcset']))
                arcset_counts[tuple(entry['arcset'])] += 1
        if verbose:
            print('\tcounting cooccs')
        coocc_counts = defaultdict(lambda: defaultdict(int))
        for idx, (span_idx, fit_arcs) in enumerate(span_to_fits.items()):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            fit_arcs = list(fit_arcs)
            for i in range(len(fit_arcs)):
                for j in range(i+1,len(fit_arcs)):
                    arc1 = fit_arcs[i]
                    arc2 = fit_arcs[j]
                    coocc_counts[arc1][arc2] += 1
                    coocc_counts[arc2][arc1] += 1
        if verbose:
            print('\tdeduplicating')
        superset_idx = 0
        supersets = defaultdict(set)
        arcset_to_superset = {}
        for arcset, count in arcset_counts.items():
            if arcset in arcset_to_superset: continue
            arcset_to_superset[arcset] = superset_idx
            supersets[superset_idx].add(arcset)
            stack = [arc2 for arc2,count2 in coocc_counts.get(arcset,{}).items()
                        if (count2/count >= threshold) and (count2/arcset_counts[arc2] >= threshold)]
            while len(stack) > 0:
                neighbour = stack.pop()
                neighbour_count = arcset_counts[neighbour]
                arcset_to_superset[neighbour] = superset_idx
                supersets[superset_idx].add(neighbour)
                stack += [arc2 for arc2,count2 in coocc_counts.get(neighbour,{}).items()
                        if (count2/neighbour_count >= threshold) and (count2/arcset_counts[arc2] >= threshold) and (arc2 not in arcset_to_superset)]
            superset_idx += 1
        superset_ids = {}
        for idx, superset in supersets.items():
            superset_ids[idx] = sorted(superset, key=lambda x: (arcset_counts[x],len(x)), reverse=True)[0]
        arcset_to_ids = {k: superset_ids[v] for k,v in arcset_to_superset.items()}
        supersets_by_id = [{'idx': k, 'id': superset_ids[k], 'items': list(v)} for k,v in supersets.items()]

        if verbose:
            print('\twriting')
        with open(outfile + '_arcset_to_super.json', 'w') as f:
            f.write('\n'.join(json.dumps({'arcset': k, 'super': v}) for k,v in arcset_to_ids.items()))
        with open(outfile + '_sets.json', 'w') as f:
            f.write('\n'.join(json.dumps(entry) for entry in supersets_by_id))

    def postprocess_fits(question_fit_file, question_tree_file, question_superset_file, verbose=5000):
        '''
            this entire file consists of two quite hacky scripts to remove
            redundant motifs (i.e. p(m1|m2), p(m2|m1) > threshold)

        '''
        downlinks = MotifsExtractor.read_downlinks(question_tree_file + '_downlinks.json')
        super_mappings = {}
        with open(question_superset_file) as f:
            for line in f.readlines():
                entry = json.loads(line)
                super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])
        super_counts = defaultdict(int)
        span_to_fits = defaultdict(set)
        with open(question_fit_file) as f:
            for idx,line in enumerate(f.readlines()):
                if verbose and (idx > 0) and (idx % verbose == 0):
                    print('\t%03d' % idx)
                entry = json.loads(line)
                span_to_fits[entry['span_idx']].add(tuple(entry['arcset']))
        for span_idx, fit_set in span_to_fits.items():
            super_fit_set = set([super_mappings[x] for x in fit_set if x != ('*',)])
            for x in super_fit_set:
                super_counts[x] += 1
            #span_to_super_fits[span_idx] = super_fit_set
        if verbose:
            print('\tmaking new entries')
        new_entries = []
        for idx, (span_idx, fit_set) in enumerate(span_to_fits.items()):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            text_idx = QuestionTypologyUtils.get_text_idx(span_idx)
            super_to_superchildren = defaultdict(set)
            for set_ in fit_set:
                if set_ == ('*',): continue
                superset = super_mappings[set_]
                super_to_superchildren[superset].update([super_mappings[child] for child,_ in downlinks.get(set_, []) if child in fit_set])
            for superset, superchildren in super_to_superchildren.items():
                entry = {'arcset': superset, 'arcset_count': super_counts[superset],
                        'text_idx': text_idx, 'span_idx': span_idx}
                if len(superchildren) == 0:
                    entry['max_child_count'] = 0
                else:
                    entry['max_child_count'] = max(super_counts.get(child,0) for child in superchildren)
                new_entries.append(entry)
        with open(question_fit_file + '.super', 'w') as f:
            f.write('\n'.join(json.dumps(entry) for entry in new_entries))

    def contains_candidate(container, candidate):
        return set(candidate).issubset(container)

    def fit_question(arc_set, downlinks, node_counts):
        fit_nodes = {}
        node_stack = [('*',)]
        while len(node_stack) > 0:
            next_node = node_stack.pop()
            node_count = node_counts.get(next_node,None)
            if node_count:
                entry = {'arcset': next_node, 'arcset_count': node_count}
                children = downlinks.get(next_node, [])
                valid_children = [child for child,_ in children if MotifsExtractor.contains_candidate(arc_set, child)]

                if len(valid_children) == 0:
                    entry['max_valid_child_count'] = 0
                else:
                    entry['max_valid_child_count'] = max(node_counts.get(child,0) for child in valid_children)
                node_stack += valid_children
                fit_nodes[next_node] = entry
        return fit_nodes

    def fit_all(arc_file, tree_file, outfile, verbose=5000):
        '''
            figures out which motifs occur in each piece of text.
            arc_file: listing of arcs per text, from extract_arcs
            tree_file: the motif graph, from make_arc_tree. note that
                this doesn't have to come from the same dataset as arc_file, in which case you're basically fitting a new dataset to motifs extracted elsewhere.
            outfile: where to put things.
        '''
        if verbose:
            print('\treading tree')
        arc_sets = QuestionTypologyUtils.read_arcs(arc_file, verbose)

        downlinks = MotifsExtractor.read_downlinks(tree_file + '_downlinks.json')
        node_counts = MotifsExtractor.read_nodecounts(tree_file + '_arc_set_counts.tsv')


        if verbose:
            print('\tfitting arcsets')
        span_fit_entries = []
        for idx, (span_idx,arcs) in enumerate(arc_sets.items()):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            text_idx = QuestionTypologyUtils.get_text_idx(span_idx)
            fit_nodes = MotifsExtractor.fit_question(set(arcs), downlinks, node_counts)
            for fit_info in fit_nodes.values():
                fit_info['span_idx'] = span_idx
                fit_info['text_idx'] = text_idx
                # fit_info['pair_idx'] = pair_idx
                span_fit_entries.append(fit_info)
        if verbose:
            print('\twriting fits')
        with open(outfile, 'w') as f:
            f.write('\n'.join(json.dumps(entry) for entry in span_fit_entries))

    def get_sorted_combos(itemset, k):
        combos = set()
        for set_ in itertools.combinations(itemset,k):
            combos.add(tuple(sorted(set_)))
        return combos
    def get_mini_powerset(itemset,k=5):
        powerset = set()
        for k in range(1,min(k+1,len(itemset)+1)):
            powerset.update(MotifsExtractor.get_sorted_combos(itemset,k))
        return powerset

    def count_frequent_itemsets(arc_sets,min_support,k=5, verbose=5000):
        itemset_counts = defaultdict(lambda: defaultdict(int))
        span_to_itemsets = defaultdict(lambda: defaultdict(set))
        if verbose:
            print('\tfirst pass')
        for idx, (span_idx,arcs) in enumerate(arc_sets.items()):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            for itemset in MotifsExtractor.get_mini_powerset(arcs,k):
                itemset_counts[len(itemset)][itemset] += 1
                span_to_itemsets[span_idx][len(itemset)].add(itemset)

        for span_idx, count_dicts in span_to_itemsets.items():
            for i in range(1,k+1):
                count_dicts[i] = [arcset for arcset in count_dicts[i] if itemset_counts[i][arcset] >= min_support]
        if verbose:
            print('\tand then the rest')
        setsize = k+1
        spans_to_check = [span_idx for span_idx,span_dict in span_to_itemsets.items() if len(span_dict[k]) > 0]
        while len(spans_to_check) > 0:
            if verbose:
                print('\t',setsize,len(spans_to_check))
            for idx, span_idx in enumerate(spans_to_check):
                if verbose and (idx > 0) and (idx % verbose == 0):
                    print('\t%03d' % idx)
                arcs = arc_sets[span_idx]
                if len(arcs) < setsize: continue
                sets_to_check = [set_ for set_ in span_to_itemsets[span_idx].get(setsize-1,[])
                                    if itemset_counts[setsize-1].get(set_,0) >= min_support]
                if len(sets_to_check) == 0: continue

                newsets = set()
                for arc in arcs:
                    if itemset_counts[1].get((arc,),0) >= min_support:
                        for set_ in sets_to_check:
                            newset = tuple(sorted(set(set_+ (arc,))))
                            if len(newset) == setsize:
                                newsets.add(newset)
                for newset in newsets:
                    itemset_counts[setsize][newset] += 1
                    span_to_itemsets[span_idx][setsize].add(newset)
            for span_idx, count_dicts in span_to_itemsets.items():
                count_dicts[setsize] = [arcset for arcset in count_dicts[setsize] if itemset_counts[setsize][arcset] >= min_support]
            spans_to_check = [span_idx for span_idx,span_dict in span_to_itemsets.items() if len(span_dict[setsize]) > 0]
            setsize+=1
        return itemset_counts, span_to_itemsets

    def make_arc_tree(arc_file, outname, min_support=5, verbose=5000):
        '''
            makes the tree of motifs. (G in the paper.)
        '''

        if verbose:
            print('\treading arcs')
        arc_sets = QuestionTypologyUtils.read_arcs(arc_file, verbose)

        if verbose:
            print('\tcounting itemsets')
        itemset_counts, span_to_itemsets = MotifsExtractor.count_frequent_itemsets(arc_sets,min_support)
        new_itemset_counts = {}
        for setsize, size_dict in itemset_counts.items():
            for k,v in size_dict.items():
                if v >= min_support:
                    new_itemset_counts[k] = v
        itemset_counts = new_itemset_counts
        itemset_counts[('*',)] = len(arc_sets)
        if verbose:
            print('\twriting itemsets')
        sorted_counts = sorted(itemset_counts.items(),key=lambda x: (-x[1],len(x[0]),x[0][0]))
        with open(outname + '_arc_set_counts.tsv', 'w') as f:
            for k,v in sorted_counts:
                f.write('%d\t%d\t%s\n' % (v, len(k), '\t'.join(k)))

        if verbose:
            print('\tbuilding tree')
        edges = []
        uplinks = defaultdict(dict)
        downlinks = defaultdict(dict)

        for itemset,count in itemset_counts.items():
            parents = []
            set_size = len(itemset)
            if set_size == 1:
                arc = itemset[0]
                if arc.endswith('*'):
                    parents.append(('*',))
                elif '_' in arc:
                    parents.append((arc.split('_')[0] + '_*',))
                elif '>' in arc:
                    parents.append((arc.split('>')[0] + '>*',))

            else:
                for idx in range(set_size):
                    parents.append(itemset[:idx] + itemset[idx+1:])
            for parent in parents:
                parent_count = itemset_counts[parent]
                pr_child = count / itemset_counts[parent]
                edges.append({'child': itemset, 'child_count': count,
                            'parent': parent, 'parent_count': parent_count,
                            'pr_child': pr_child})
                uplinks[itemset][parent] = {'pr_child': pr_child, 'parent_count': parent_count}
                downlinks[parent][itemset] = {'pr_child': pr_child, 'child_count': count}

        with open(outname + '_edges.json', 'w') as f:
            f.write('\n'.join(json.dumps(edge) for edge in edges))
        with open(outname + '_uplinks.json', 'w') as f:
            uplink_list = []
            for child, parent_dict in uplinks.items():
                uplink_list.append({'child': child, 'parents': sorted(parent_dict.items(),key=lambda x: x[1]['pr_child'])})
            uplink_list = sorted(uplink_list, key=lambda x: itemset_counts[x['child']], reverse=True)
            f.write('\n'.join(json.dumps(up) for up in uplink_list))
        with open(outname + '_downlinks.json', 'w') as f:
            downlink_list = []
            for parent, child_dict in downlinks.items():
                downlink_list.append({'parent': parent, 'children': sorted(child_dict.items(),key=lambda x: x[1]['pr_child'])})
            downlink_list = sorted(downlink_list, key=lambda x: itemset_counts[x['parent']], reverse=True)
            f.write('\n'.join(json.dumps(down) for down in downlink_list))




    def is_noun_ish(word):
        return (word.dep in NP_LABELS) or (word.tag_.startswith('NN') or word.tag_.startswith('PRP')) or (word.tag_.endswith('DT'))

    def has_w_det(token):
        if token.tag_.startswith('W'): return token.text
        first_tok = next(token.subtree)
        if (first_tok.tag_.startswith('W')): return first_tok.text
        return False

    def get_tok(token):
        if MotifsExtractor.is_noun_ish(token):
            has_w = MotifsExtractor.has_w_det(token)
            if has_w:
                return has_w.lower(), True
            else:
                return 'NN', True
        else:
            return token.text.lower(), False

    def get_clean_tok(tok):
        out_tok, is_noun = MotifsExtractor.get_tok(tok)
        return out_tok.replace('--','').strip(), is_noun

    def is_alpha_ish(text):
        return text.isalpha() or text[1:].isalpha()

    def is_usable(text):
        return MotifsExtractor.is_alpha_ish(text) and (text != 'NN')


    def get_arcs(root, follow_conj):

        # todo: could imagine version where nouns allowed
        arcs = set()
        root_tok, _ = MotifsExtractor.get_clean_tok(root)
        if not MotifsExtractor.is_usable(root_tok): return arcs

        arcs.add(root_tok + '_*')
        conj_elems = []
        for idx, kid in enumerate(root.children):
            if kid.dep_ in ['punct','cc']:
                continue
            elif kid.dep_ == 'conj':
                if follow_conj:
                    conj_elems.append(kid)
            else:
                kid_tok, _ = MotifsExtractor.get_clean_tok(kid)
                if MotifsExtractor.is_usable(kid_tok):
                    arcs.add(root_tok + '_' + kid_tok)

        first_elem = next(root.subtree)
        first_tok, _ = MotifsExtractor.get_clean_tok(first_elem)
        if MotifsExtractor.is_usable(first_tok):
            arcs.add(first_tok + '>*')
            try:
                second_elem = first_elem.nbor()
                second_tok, _ = MotifsExtractor.get_clean_tok(second_elem)
                if MotifsExtractor.is_usable(second_tok):
                    arcs.add(first_tok + '>' + second_tok)
            except:
                pass

        for conj_elem in conj_elems:
            arcs.update(MotifsExtractor.get_arcs(conj_elem, follow_conj))
        return arcs

    def is_question(span):
        span_text = span.text.strip()
        return span_text[-1] == '?'

    def extract_arcs(text_iter, spacy_filename, outfile, vocab, use_span=is_question ,
        follow_conj=True, verbose=5000):

        '''
            extracts all arcs going out of the root in a sentence. used to find question motifs.

            text_iter: iterates over text for which arcs are extracted
            spacy_filename: location of spacy objects (from spacy_utils.py)
            outfile: where to write the arcs.
            vocab: pre-loaded spacy vocabulary. if you pass None it will load vocab for you, but that's slow.
            use_span: filter to decide which sentences to use. the function takes in a spacy sentence object.
            follow_conj: whether to follow conjunctions and treat subtrees as sentences too.

        '''

        if verbose:
            print('reading spacy')
        spacy_dict = MotifsExtractor.get_spacy_dict(spacy_filename, vocab)

        arc_entries = []
        for idx, (text_idx,text, pair_idx) in enumerate(text_iter):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)
            spacy_obj = spacy_dict[text_idx]
            for span_idx, span in enumerate(spacy_obj.sents):
                if use_span(span):
                    curr_arcset = MotifsExtractor.get_arcs(span.root, follow_conj)
                    arc_entries.append({'idx': '%s_%d' % (text_idx, span_idx), 'arcs': list(curr_arcset),
                        'pair_idx': '%s_%d' % (pair_idx, span_idx)})
        if verbose:
            print('\twriting arcs')
        with open(outfile, 'w') as f:
            f.write('\n'.join(json.dumps(arc_entry) for arc_entry in arc_entries))


    def is_uppercase_question(x):
        '''
            for reasonably well-formatted datasets like transcripts of some proceedings, i've included this filter that questions start w/ uppercase and end in a question mark. this filter can be varied/swapped out.
        '''
        text = x.text.strip()
        return (text[-1] == '?') and (text[0].isupper())

    def is_uppercase(x):
        '''
            mainly because we otherwise get a bunch of badly parsed half-lines,
            enforce that answer sentences have to start in uppercase (reliable
            provided your data is well-formatted...)
        '''
        text = x.text.strip()
        return text[0].isupper()


    def extract_question_motifs(question_text_iter, spacy_filename, motif_dir,
        question_filter_fn = is_uppercase_question,
        follow_conj=True,
        min_question_itemset_support=5,
        deduplicate_threshold=.9,
        verbose=5000):
        '''
            convenience pipeline to get question motifs. (see pipelines/extract_*_motifs for examples)
            question_text_iter: iterates over all questions
            spacy_filename: location of spacy objects
            motif_dir: directory where all motifs written
            question_filter_fn: only uses sentences in a question which corresponds to a question. can redefine.
            follow_conj: follows conjunctions to compound questions ("why...and how")
            min_question_itemset_support: the minimum number of times an itemset has to show up for the frequent itemset counter to consider it.
            deduplicate_threshold: how often two motifs co-occur (i.e. p(x|y) and p(y|x) for us to consider them redundant)
        '''
        print('running motif extraction pipeline')

        try:
            os.mkdir(motif_dir)
        except:
            print('\tmotif dir %s exists!' % motif_dir)

        print('loading spacy vocab')
        vocab = MotifsExtractor.load_vocab()

        print('getting question arcs')
        question_arc_outfile = os.path.join(motif_dir, 'question_arcs.json')
        MotifsExtractor.extract_arcs(question_text_iter, spacy_filename, question_arc_outfile, vocab, use_span=question_filter_fn, follow_conj=follow_conj, verbose=verbose)

        print('making motif tree')
        question_tree_outfile = os.path.join(motif_dir, 'question_tree')
        MotifsExtractor.make_arc_tree(question_arc_outfile, question_tree_outfile, min_question_itemset_support, verbose=verbose)

        print('fitting motifs to questions')
        question_fit_outfile = os.path.join(motif_dir, 'question_fits.json')
        MotifsExtractor.fit_all(question_arc_outfile, question_tree_outfile, question_fit_outfile, verbose=verbose)

        print('handling redundant motifs')
        question_super_outfile = os.path.join(motif_dir, 'question_supersets')
        MotifsExtractor.deduplicate_motifs(question_fit_outfile, question_super_outfile, deduplicate_threshold, verbose=verbose)
        MotifsExtractor.postprocess_fits(question_fit_outfile, question_tree_outfile, question_super_outfile + '_arcset_to_super.json')

        print('done motif extraction')

    def read_downlinks(downlink_file):
        downlinks = {}
        with open(downlink_file) as f:
            for line in f.readlines():
                entry = json.loads(line)
                downlinks[tuple(entry['parent'])] = [(tuple(x),y) for x,y in entry['children']]
        return downlinks

    def read_nodecounts(nodecount_file):
        node_counts = {}
        with open(nodecount_file) as f:
            for line in f:
                split = line.split('\t')
                count = int(split[0])
                set_size = int(split[1])
                itemset = tuple([x.strip() for x in split[2:]])
                node_counts[itemset] = count
        return node_counts

    def extract_answer_arcs(answer_text_iter, spacy_filename, motif_dir, answer_filter_fn=is_uppercase, follow_conj=True, verbose=5000):
        '''
            convenience pipeline to get answer motifs
        '''

        print('running answer arc pipeline')
        try:
            os.mkdir(motif_dir)
        except:
            print('\tmotif dir %s exists!' % motif_dir)

        print('loading spacy vocab')
        vocab = MotifsExtractor.load_vocab()

        print('getting answer arcs')
        answer_arc_outfile = os.path.join(motif_dir, 'answer_arcs.json')
        MotifsExtractor.extract_arcs(answer_text_iter, spacy_filename, answer_arc_outfile, vocab, use_span=answer_filter_fn, follow_conj=follow_conj, verbose=verbose)

        print('done answer arc extraction')

class QuestionClusterer:

    def read_uplinks(uplink_file):
        uplinks = {}
        with open(uplink_file) as f:
            for line in f.readlines():
                entry = json.loads(line)
                uplinks[tuple(entry['child'])] = [(tuple(x),y) for x,y in entry['parents']]
        return uplinks

    def get_motifs_per_question(question_fit_file, answer_arc_file, superset_file,question_threshold, answer_threshold,  verbose=5000):
        question_to_fits = defaultdict(set)
        question_to_leaf_fits = defaultdict(set)
        motif_counts = defaultdict(int)


        super_mappings = {}
        with open(superset_file) as f:
            for line in f.readlines():
                entry = json.loads(line)
                super_mappings[tuple(entry['arcset'])] = tuple(entry['super'])

        with open(question_fit_file) as f:
            for idx, line in enumerate(f.readlines()):
                if verbose and (idx > 0) and (idx % verbose == 0):
                    print('\t%03d' % idx)
                entry = json.loads(line)
                motif = tuple(entry['arcset'])
                super_motif = super_mappings[motif]
                if entry['arcset_count'] < question_threshold: continue
                if entry['max_child_count'] < question_threshold:
                    question_to_leaf_fits[entry['text_idx']].add(super_motif)
                    #if leaves_only: continue
                question_to_fits[entry['text_idx']].add(super_motif)
                motif_counts[super_motif] += 1
        question_to_fits = {k: [x for x in v if motif_counts[x] >= question_threshold] for k,v in question_to_fits.items()}
        motif_counts = {k:v for k,v in motif_counts.items() if v >= question_threshold}
        question_to_leaf_fits = {k: [x for x in v if motif_counts.get(x,0) >= question_threshold] for k,v in question_to_leaf_fits.items()}

        question_to_arcs = defaultdict(set)
        arc_sets = QuestionTypologyUtils.read_arcs(answer_arc_file)
        arc_counts = defaultdict(int)
        for span_idx, arcs in arc_sets.items():
            question_to_arcs[QuestionTypologyUtils.get_text_idx(span_idx)].update(arcs)
            for arc in arcs:
                arc_counts[arc] += 1
        question_to_arcs = {k: [x for x in v if arc_counts[x] >= answer_threshold] for k,v in question_to_arcs.items()}
        arc_counts = {k:v for k,v in arc_counts.items() if v >= answer_threshold}
        return question_to_fits, question_to_leaf_fits, motif_counts, question_to_arcs, arc_counts

    def build_joint_matrix(question_fit_file, answer_arc_file, superset_file, outfile, question_threshold, answer_threshold, verbose=5000):
        if verbose:
            print('\treading arcs and motifs')

        question_to_fits, question_to_leaf_fits, motif_counts, question_to_arcs, arc_counts =\
             QuestionClusterer.get_motifs_per_question(question_fit_file, answer_arc_file, superset_file, question_threshold, answer_threshold, verbose)
        question_term_list = list(motif_counts.keys())
        answer_term_list = list(arc_counts.keys())

        question_term_to_idx = {k:idx for idx,k in enumerate(question_term_list)}
        answer_term_to_idx = {k:idx for idx,k in enumerate(answer_term_list)}

        if verbose:
            print('\tbuilding matrices')
        question_term_idxes = []
        question_leaves = []
        question_doc_idxes = []
        answer_term_idxes = []
        answer_doc_idxes = []
        pair_idx_list = []

        pair_idxes = list(set(question_to_fits.keys()).intersection(set(question_to_arcs.keys())))

        for idx, p_idx in enumerate(pair_idxes):
            if verbose and (idx > 0) and (idx % verbose == 0):
                print('\t%03d' % idx)

            question_terms = question_to_fits[p_idx]
            answer_terms = question_to_arcs[p_idx]
            pair_idx_list.append(p_idx)

            for term in question_terms:
                term_idx = question_term_to_idx[term]
                question_term_idxes.append(term_idx)
                question_doc_idxes.append(idx)
                question_leaves.append(term in question_to_leaf_fits.get(p_idx,[]))
            for term in answer_terms:
                term_idx = answer_term_to_idx[term]
                answer_term_idxes.append(term_idx)
                answer_doc_idxes.append(idx)
        if verbose:
            print('\twriting stuff')

        np.save(outfile + '.q.tidx.npy', question_term_idxes)
        np.save(outfile + '.q.leaves.npy', question_leaves)
        np.save(outfile + '.a.tidx.npy', answer_term_idxes)
        np.save(outfile + '.q.didx.npy', question_doc_idxes)
        np.save(outfile + '.a.didx.npy', answer_doc_idxes)
        with open(outfile + '.q.terms.txt', 'w') as f:
            f.write('\n'.join('%d\t%s' % (motif_counts[term],term) for term in question_term_list))
        with open(outfile + '.a.terms.txt', 'w') as f:
            f.write('\n'.join('%d\t%s' % (arc_counts[term],term) for term in answer_term_list))
        with open(outfile + '.docs.txt', 'w') as f:
            f.write('\n'.join(pair_idxes))
        with open(outfile + '.pair_idxs.txt', 'w') as f:
            f.write('\n'.join(pair_idx_list))

    def load_joint_mtx(rootname):
        mtx_obj = {}
        #rootname = os.path.join(LATENT_DIR, data_name, feature_name)

        print('reading question tidxes')
        mtx_obj['q_tidxes'] = np.load(rootname + '.q.tidx.npy')
        print('reading question leaves')
        mtx_obj['q_leaves'] = np.load(rootname + '.q.leaves.npy')
        print('reading answer tidxes')
        mtx_obj['a_tidxes'] = np.load(rootname + '.a.tidx.npy')

        print('reading question didxes')
        mtx_obj['q_didxes'] = np.load(rootname + '.q.didx.npy')
        print('reading answer didxes')
        mtx_obj['a_didxes'] = np.load(rootname + '.a.didx.npy')

        print('reading question terms')
        mtx_obj['q_terms'] = []
        mtx_obj['q_term_to_idx'] = {}
        mtx_obj['q_term_counts'] = []
        fname = rootname + '.q.terms.txt'
        with open(fname) as f:
            for idx, line in enumerate(f.readlines()):
                count,term = line.split('\t')
                term = term.strip()
                term = make_tuple(term)
                mtx_obj['q_term_counts'].append(int(count))
                mtx_obj['q_terms'].append(term)
                mtx_obj['q_term_to_idx'][term] = idx
        mtx_obj['q_terms'] = np.array(mtx_obj['q_terms'])
        mtx_obj['q_term_counts'] = np.array(mtx_obj['q_term_counts'])

        print('reading answer terms')
        mtx_obj['a_terms'] = []
        mtx_obj['a_term_to_idx'] = {}
        mtx_obj['a_term_counts'] = []
        fname = rootname + '.a.terms.txt'
        with open(fname) as f:
            for idx, line in enumerate(f.readlines()):
                count,term = line.split('\t')
                term = term.strip()
                mtx_obj['a_term_counts'].append(int(count))
                mtx_obj['a_terms'].append(term)
                mtx_obj['a_term_to_idx'][term] = idx
        mtx_obj['a_terms'] = np.array(mtx_obj['a_terms'])
        mtx_obj['a_term_counts'] = np.array(mtx_obj['a_term_counts'])

        print('reading docs')
        mtx_obj['docs'] = []
        mtx_obj['doc_to_idx'] = {}
        with open(rootname + '.docs.txt') as f:
            for idx, line in enumerate(f.readlines()):
                doc_id = line.strip()
                mtx_obj['docs'].append(doc_id)
                mtx_obj['doc_to_idx'][doc_id] = idx
        mtx_obj['docs'] = np.array(mtx_obj['docs'])

        print('done!')
        return mtx_obj

    def build_mtx(mtx_obj, data_type, norm='l2', idf=False, leaves_only=False):
        N_terms = len(mtx_obj[data_type + '_terms'])
        N_docs = len(mtx_obj['docs'])
        if idf:
            data = np.log(N_docs) - np.log(mtx_obj[data_type + '_term_counts'][mtx_obj[data_type + '_tidxes']])
        else:
            data = np.ones_like(mtx_obj[data_type + '_tidxes'])
            if leaves_only:
                data[~mtx_obj[data_type + '_leaves']] = 0
        mtx = sparse.csr_matrix((data, (mtx_obj[data_type + '_tidxes'], mtx_obj[data_type + '_didxes'])), shape=(N_terms,N_docs))
        if norm:
            mtx = Normalizer(norm=norm).fit_transform(mtx)

        return mtx

    def run_simple_pipe(rootname, verbose=True):
        mtx_obj = QuestionClusterer.load_joint_mtx(rootname)
        q_mtx = QuestionClusterer.build_mtx(mtx_obj, 'q')
        a_mtx = QuestionClusterer.build_mtx(mtx_obj, 'a', idf=True)
        return q_mtx, a_mtx, mtx_obj

    def do_sparse_svd(mtx, k=50):
        u,s,v = sparse.linalg.svds(mtx, k=k) # ugh, right order dammit
        return u[:,::-1],s[::-1],v[::-1,:]

    def run_lowdim_pipe(q_mtx, a_mtx, k=50, snip=True):
        a_u, a_s, a_v = QuestionClusterer.do_sparse_svd(a_mtx,k + int(snip))
        lq = q_mtx * (a_v.T * a_s**-1)
        if snip:
            return QuestionClusterer.snip_first_dim(lq, a_u, a_s, a_v)
        else:
            return lq, a_u, a_s, a_v

    def inspect_latent_space(mtx, names, dim_iter=None, num_dims=5, num_egs=10, which_end=None, skip_first=True, dim_names={},s=None):
        mtx = Normalizer().fit_transform(mtx).T
        if dim_iter is None:
            dim_iter = range(int(skip_first), num_dims + int(skip_first))
        for dim in dim_iter:
            if s is not None:
                print(dim,s[dim])
            else:
                print(dim)
            row = mtx[dim]
            argsorted = np.argsort(row)
            if (not which_end) or (which_end == -1):
                print('\tbottom',dim_names.get((dim,-1), ''))
                for i in range(num_egs):
                    print('\t\t',names[argsorted[i]], '%+.4f' % row[argsorted[i]])
            if (not which_end) or (which_end == 1):
                print('\ttop',dim_names.get((dim,1), ''))
                for i in range(num_egs):
                    print('\t\t',names[argsorted[-1-i]], '%+.4f' % row[argsorted[-1-i]])
            print()

    def run_kmeans(X, in_dim, k):
        km = KMeans(n_clusters=k,max_iter=1000)
        km.fit(X)
        return km

    def inspect_kmeans_run(q_mtx, a_mtx, num_svd_dims, num_clusters, q_terms, a_terms, km=None, remove_first=False, num_egs=10):
        if remove_first:
            q_mtx = q_mtx[:,1:(num_svd_dims + 1)]
            a_mtx = a_mtx[:,1:(num_svd_dims + 1)]
        else:
            q_mtx = q_mtx[:,:num_svd_dims]
            a_mtx = a_mtx[:,:num_svd_dims]
        q_mtx = Normalizer().fit_transform(q_mtx)
        a_mtx = Normalizer().fit_transform(a_mtx)
        types_to_data = {}
        if km:
            q_km = km
        else:
            q_km = QuestionClusterer.run_kmeans(q_mtx, num_svd_dims, num_clusters)
        if num_egs > 0:
            q_dists = q_km.transform(q_mtx)
            q_assigns = q_km.labels_
            a_dists = q_km.transform(a_mtx)
            a_assigns = q_km.predict(a_mtx)
            for cl in range(num_clusters):
                types_to_data[cl] = {
                    "motifs": [],
                    "motif_dists": [],
                    "fragments": [],
                    "fragment_dists": [],
                    "questions": [],
                    "question_dists": [],
                } 
                # print('cluster',cl)
                q_assigned = q_assigns == cl
                median_qdist = np.median(q_dists[:,cl][q_assigned])
                # print('\tq assigns:',q_assigned.sum(),  'median dist:', '%.4f' % median_qdist)
                a_assigned = a_assigns == cl
                median_adist = np.median(a_dists[:,cl][a_assigned])
                # print('\ta assigns:',a_assigned.sum(),  'median dist:', '%.4f' % median_adist)
                if num_egs == 0: continue
                argsorted_qdists = np.argsort(q_dists[:,cl])
                argsorted_qdists = argsorted_qdists[np.in1d(argsorted_qdists, np.where(q_assigned)[0])]
                # print('\tqs:')
                for i in range(q_assigned.sum()):
                    curr_qdist = q_dists[:,cl][argsorted_qdists[i]]
                    if curr_qdist > median_qdist:
                        diststr = '%.4f ~~' %  curr_qdist
                    else:
                        diststr = '%.4f' % curr_qdist
                    # print('\t\t', q_terms[argsorted_qdists[i]], diststr)
                    types_to_data[cl]["motifs"].append(q_terms[argsorted_qdists[i]])
                    types_to_data[cl]["motif_dists"].append(diststr)
                argsorted_adists = np.argsort(a_dists[:,cl])
                argsorted_adists = argsorted_adists[np.in1d(argsorted_adists, np.where(a_assigned)[0])]
                # print('\tas:')
                for i in range(a_assigned.sum()):
                    curr_adist = a_dists[:,cl][argsorted_adists[i]]
                    if curr_adist > median_adist:
                        diststr = '%.4f ~~' %  curr_adist
                    else:
                        diststr = '%.4f' % curr_adist
                    # print('\t\t', a_terms[argsorted_adists[i]], diststr)
                    types_to_data[cl]["fragments"].append(a_terms[argsorted_adists[i]])
                    types_to_data[cl]["fragment_dists"].append(diststr)
                # print('========================')
        return q_km, types_to_data

    def snip_first_dim(lq, a_u, a_s, a_v):
        return lq[:,1:], a_u[:,1:], a_s[1:], a_v[1:]

    def assign_clusters(km, lq, a_u, mtx_obj, n_dims):
        km_qdists = km.transform(Normalizer().fit_transform(lq[:,:n_dims]))
        km_qlabels = km.predict(Normalizer().fit_transform(lq[:,:n_dims]))
        km_adists = km.transform(Normalizer().fit_transform(a_u[:,:n_dims]))
        km_alabels = km.predict(Normalizer().fit_transform(a_u[:,:n_dims]))

        motif_df_entries = []
        for idx, motif in enumerate(mtx_obj['q_terms']):
            entry = {'idx': idx, 'motif': motif, 'cluster': km_qlabels[idx],
                    'count': mtx_obj['q_term_counts'][idx]}
            entry['cluster_dist'] = km_qdists[idx,entry['cluster']]
            motif_df_entries.append(entry)
        motif_df = pd.DataFrame(motif_df_entries).set_index('idx')

        aarc_df_entries = []
        for idx, aarc in enumerate(mtx_obj['a_terms']):
            entry = {'idx': idx, 'aarc': aarc, 'cluster': km_alabels[idx],
                    'count': mtx_obj['a_term_counts'][idx]}
            entry['cluster_dist'] = km_adists[idx,entry['cluster']]
            aarc_df_entries.append(entry)
        aarc_df = pd.DataFrame(aarc_df_entries).set_index('idx')

        q_leaves = QuestionClusterer.build_mtx(mtx_obj,'q',leaves_only=True)
        qdoc_vects = q_leaves.T * Normalizer().fit_transform(lq)
        km_qdoc_dists = km.transform(Normalizer().fit_transform(qdoc_vects[:,:n_dims]))
        km_qdoc_labels = km.predict(Normalizer().fit_transform(qdoc_vects[:,:n_dims]))
        qdoc_df_entries = []
        for idx, qdoc in enumerate(mtx_obj['docs']):
            entry = {'idx': idx, 'q_idx': qdoc, 'cluster': km_qdoc_labels[idx]}
            entry['cluster_dist'] = km_qdoc_dists[idx,entry['cluster']]
            qdoc_df_entries.append(entry)
        qdoc_df = pd.DataFrame(qdoc_df_entries).set_index('idx')

        return motif_df, aarc_df, qdoc_df

    def build_matrix(motif_dir, matrix_dir, question_threshold, answer_threshold):
        '''
            convenience pipeline to build the question answer matrices.
            motif_dir: wherever extract_motifs wrote to
            matrix_dir: where to put the matrices
            question_threshold: minimum # of questions in which a question motif has to occur to be considered
        '''
        print('building q-a matrices')
        question_fit_file = os.path.join(motif_dir, 'question_fits.json.super')
        answer_arc_file = os.path.join(motif_dir, 'answer_arcs.json')
        superset_file = os.path.join(motif_dir, 'question_supersets_arcset_to_super.json')

        try:
            os.mkdir(matrix_dir)
        except:
            print('matrix dir %s exists!' % matrix_dir)

        outfile = os.path.join(matrix_dir, 'qa_mtx')
        QuestionClusterer.build_joint_matrix(question_fit_file, answer_arc_file,superset_file, outfile, question_threshold, answer_threshold, verbose=5000)

    def extract_clusters(matrix_dir,km_file,k=8, d=25,num_egs=10):
        '''
            convenience pipeline to get latent q-a dimensions and clusters.

            km_file: where to write the kmeans object
            k: num clusters
            d: num latent dims

        '''
        matrix_file = os.path.join(matrix_dir, 'qa_mtx')
        q_mtx, a_mtx, mtx_obj = QuestionClusterer.run_simple_pipe(matrix_file)
        lq, a_u, a_s, a_v = QuestionClusterer.run_lowdim_pipe(q_mtx,a_mtx,d)
        km, types_to_data = QuestionClusterer.inspect_kmeans_run(lq,a_u,d,k,mtx_obj['q_terms'], mtx_obj['a_terms'], num_egs=num_egs)

        joblib.dump(km, km_file)
        return mtx_obj, km, types_to_data, lq, a_u

class QuestionTypologyUtils:
    def read_arcs(arc_file, verbose=5000):
        arc_sets = {}
        with open(arc_file) as f:
            for idx,line in enumerate(f.readlines()):
                if (idx > 0) and (idx % verbose == 0):
                    print('\t%03d' % idx)
                entry = json.loads(line)
                arc_sets[entry['pair_idx']] = entry['arcs']
        return arc_sets

    def get_text_idx(span_idx):
        # return '.'.join(span_idx.split('.')[:-1])
        return span_idx[:span_idx.rfind("_")]
        # return span_idx
