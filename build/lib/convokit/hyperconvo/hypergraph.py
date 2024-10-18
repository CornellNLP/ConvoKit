from typing import Tuple, List, Dict, Collection
from collections import defaultdict
from convokit import Utterance, Speaker
import itertools


class Hypergraph:
    """
    Represents a hypergraph, consisting of nodes, directed edges,
    hypernodes (each of which is a set of nodes) and hyperedges (directed edges
    from hypernodes to hypernodes). Contains functionality to extract motifs
    from hypergraphs (Fig 2 of
    http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html)
    """

    def __init__(self):
        # public
        self.nodes: Dict[str, Utterance] = dict()
        self.hypernodes = dict()
        self.speakers = dict()

        # private
        self.adj_out = dict()  # out edges for each (hyper)node
        self.adj_in = dict()  # in edges for each (hyper)node

    @staticmethod
    def init_from_utterances(utterances: List[Utterance]):
        utt_dict = {utt.id: utt for utt in utterances}
        utt_to_speaker_id = {utt.id: utt.speaker.id for utt in utterances}
        hypergraph = Hypergraph()
        speaker_to_utt_ids = dict()
        reply_edges = []
        speaker_to_reply_tos = defaultdict(list)
        speaker_target_pairs = list()

        # nodes (utts)
        for utt in sorted(utterances, key=lambda h: h.timestamp):
            if utt.speaker not in speaker_to_utt_ids:
                speaker_to_utt_ids[utt.speaker] = set()
            speaker_to_utt_ids[utt.speaker].add(utt.id)

            if utt.reply_to is not None and utt.reply_to in utt_dict:
                reply_edges.append((utt.id, utt.reply_to))
                speaker_to_reply_tos[utt.speaker.id].append(utt.reply_to)
                speaker_target_pairs.append(
                    [
                        utt.speaker.id,
                        utt_dict[utt.reply_to].speaker.id,
                        {"utt": utt, "target_speaker": utt_to_speaker_id[utt.reply_to]},
                    ]
                )
            hypergraph.add_node(utt)

        # hypernodes (speakers)
        for speaker, utt_ids in speaker_to_utt_ids.items():
            hypergraph.add_hypernode(speaker, utt_ids)

        # reply edges (utt to utt)
        for speaker_utt_id, target_utt_id in reply_edges:
            hypergraph.add_edge(speaker_utt_id, target_utt_id)

        # hypernode to node response edges
        for speaker, reply_tos in speaker_to_reply_tos.items():
            for reply_to in reply_tos:
                hypergraph.add_edge(speaker, reply_to)

        # hypernode to hypernode response edges
        for speaker, target, utt in speaker_target_pairs:
            hypergraph.add_edge(speaker, target, utt)

        return hypergraph

    def add_node(self, utt: Utterance) -> None:
        self.nodes[utt.id] = utt
        self.adj_out[utt.id] = dict()
        self.adj_in[utt.id] = dict()

    def add_hypernode(self, speaker: Speaker, nodes: Collection[str]) -> None:
        self.hypernodes[speaker.id] = set(nodes)
        self.speakers[speaker.id] = speaker
        self.adj_out[speaker.id] = dict()
        self.adj_in[speaker.id] = dict()

    # edge or hyperedge
    def add_edge(self, u: str, v: str, info=None) -> None:
        assert u in self.nodes or u in self.hypernodes
        assert v in self.nodes or v in self.hypernodes
        # if u in self.hypernodes and v in self.hypernodes:
        #     assert info is not N
        if v not in self.adj_out[u]:
            self.adj_out[u][v] = []
        if u not in self.adj_in[v]:
            self.adj_in[v][u] = []
        if info is None:
            info = dict()
        self.adj_out[u][v].append(info)
        self.adj_in[v][u].append(info)

    def edges(self) -> Dict[Tuple[str, str], List]:
        return dict(((u, v), lst) for u, d in self.adj_out.items() for v, lst in d.items())

    def outgoing_nodes(self, u: str) -> Dict[str, List]:
        assert u in self.adj_out
        return dict((v, lst) for v, lst in self.adj_out[u].items() if v in self.nodes)

    def outgoing_hypernodes(self, u) -> Dict[str, List]:
        assert u in self.adj_out
        return dict((v, lst) for v, lst in self.adj_out[u].items() if v in self.hypernodes)

    def incoming_nodes(self, v: str) -> Dict[str, List]:
        assert v in self.adj_in
        return dict((u, lst) for u, lst in self.adj_in[v].items() if u in self.nodes)

    def incoming_hypernodes(self, v: str) -> Dict[str, List]:
        assert v in self.adj_in
        return dict((u, lst) for u, lst in self.adj_in[v].items() if u in self.hypernodes)

    def outdegrees(self, from_hyper: bool = False, to_hyper: bool = False) -> List[int]:
        retval = []
        from_nodes = self.hypernodes if from_hyper else self.nodes
        to_nodes = self.hypernodes if to_hyper else self.nodes

        for node in from_nodes:
            retval.append(sum([1 for v, l in self.adj_out[node].items() if v in to_nodes]))
        return retval

    def indegrees(self, from_hyper: bool = False, to_hyper: bool = False) -> List[int]:
        retval = []
        from_nodes = self.hypernodes if from_hyper else self.nodes
        to_nodes = self.hypernodes if to_hyper else self.nodes

        for node in to_nodes:
            retval.append(sum([1 for u, l in self.adj_in[node].items() if u in from_nodes]))
        return retval

    def reciprocity_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C1, c1, c2, C1->c2, c2->c1) as in paper
        """
        motifs = []
        for C1, c1_nodes in self.hypernodes.items():
            for c1 in c1_nodes:
                motifs += [
                    (C1, c1, c2, e1, e2)
                    for c2 in self.adj_in[c1]
                    if c2 in self.nodes and c2 in self.adj_out[C1]
                    for e1 in self.adj_out[C1][c2]  # only 1 such e1
                    for e2 in self.adj_out[c2][c1]
                ]  # only 1 such e2
        return motifs

    def external_reciprocity_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C3, c2, c1, C3->c2, c2->c1) as in paper
        """
        motifs = []
        for C3 in self.hypernodes:
            for c2 in self.adj_out[C3]:
                if c2 in self.nodes:
                    motifs += [
                        (C3, c2, c1, e1, e2)
                        for c1 in set(self.adj_out[c2].keys()) - self.hypernodes[C3]
                        if c1 in self.nodes
                        for e1 in self.adj_out[C3][c2]  # there should be only 1 such e1
                        for e2 in self.adj_out[c2][c1]
                    ]  # there should only be 1 such e2
        return motifs

    def dyadic_interaction_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C1, C2, C1->C2, C2->C1) as in paper
        """

        motifs = []
        for C1, C2 in itertools.combinations(self.hypernodes, 2):
            if len(self.adj_out[C1].get(C2, [])) > 0 and len(self.adj_out[C2].get(C1, [])) > 0:
                motifs += [(C1, C2, self.adj_out[C1][C2], self.adj_out[C2][C1])]
        return motifs

    def incoming_triad_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C1, C2, C3, C2->C1, C3->C1) as in paper
        """
        motifs = []
        for C1 in self.hypernodes:
            incoming = [C for C in self.adj_in[C1].keys() if C not in self.nodes]
            for C2, C3 in itertools.combinations(incoming, 2):
                motifs += [(C1, C2, C3, self.adj_out[C2][C1], self.adj_out[C3][C1])]
        return motifs

    def outgoing_triad_motifs(self) -> List[Tuple]:
        """
        :return: List of tuples of form (C1, C2, C3, C1->C2, C1->C3) as in paper
        """
        motifs = []
        for C1 in self.hypernodes:
            outgoing = [C for C in self.adj_out[C1].keys() if C not in self.nodes]
            for C2, C3 in itertools.combinations(outgoing, 2):
                motifs += [(C1, C2, C3, self.adj_out[C1][C2], self.adj_out[C1][C3])]
        return motifs
