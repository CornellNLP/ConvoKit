import unittest

from convokit.model import Corpus
from convokit.tests.general.traverse_convo.traverse_convo_helpers import (
    construct_tree_corpus,
    construct_nonexistent_reply_to_corpus,
    construct_multiple_convo_id_corpus,
    construct_no_timestamp_tree_corpus,
)
from convokit.tests.test_utils import reload_corpus_in_db_mode


class CorpusTraversal(unittest.TestCase):
    def broken_convos(self):
        # test broken convo where there are multiple conversation_ids
        convo = self.multiple_convo_id_corpus.get_conversation("convo_id_0")
        self.assertRaises(ValueError, lambda: list(convo.traverse("dfs", as_utterance=True)))

        # test broken convo where utterance replies to something not in Conversation
        convo = self.nonexistent_reply_to_corpus.get_conversation("convo_id_0")
        self.assertRaises(ValueError, lambda: list(convo.traverse("dfs", as_utterance=True)))

    def bfs_traversal(self):
        convo = self.corpus.get_conversation("0")
        bfs_traversal = [utt.id for utt in convo.traverse("bfs", as_utterance=True)]
        self.assertEqual(bfs_traversal, [str(i) for i in range(12)])

    def dfs_traversal(self):
        convo = self.corpus.get_conversation("0")
        dfs_traversal = [utt.id for utt in convo.traverse("dfs", as_utterance=True)]
        self.assertEqual(dfs_traversal, [str(i) for i in [0, 1, 4, 10, 5, 6, 2, 7, 8, 3, 9, 11]])

    def postorder_traversal(self):
        convo = self.corpus.get_conversation("0")
        postorder_traversal = [utt.id for utt in convo.traverse("postorder", as_utterance=True)]
        self.assertEqual(
            postorder_traversal, ["10", "4", "5", "6", "1", "7", "8", "2", "11", "9", "3", "0"]
        )

    def preorder_traversal(self):
        convo = self.corpus.get_conversation("0")
        preorder_traversal = [utt.id for utt in convo.traverse("preorder", as_utterance=True)]
        self.assertEqual(
            preorder_traversal, ["0", "1", "4", "10", "5", "6", "2", "7", "8", "3", "9", "11"]
        )

    def subtree(self):
        convo = self.corpus.get_conversation("0")
        node = convo.get_subtree("1")
        self.assertEqual([node.utt.id for node in node.bfs_traversal()], ["1", "4", "5", "6", "10"])

    def conversation_id_to_leaf_paths(self):
        convo = self.corpus.get_conversation("0")
        paths = convo.get_root_to_leaf_paths()
        path_tuples = [tuple(utt.id for utt in paths[i]) for i in range(6)]
        self.assertIn(("0", "1", "4", "10"), path_tuples)
        self.assertIn(("0", "1", "5"), path_tuples)
        self.assertIn(("0", "1", "6"), path_tuples)
        self.assertIn(("0", "2", "7"), path_tuples)
        self.assertIn(("0", "2", "8"), path_tuples)
        self.assertIn(("0", "3", "9", "11"), path_tuples)

    def one_utt_convo(self):
        convo = self.corpus.get_conversation("other")
        self.assertEqual([utt.id for utt in convo.traverse("bfs")], ["other"])
        self.assertEqual([utt.id for utt in convo.traverse("dfs")], ["other"])
        self.assertEqual([utt.id for utt in convo.traverse("postorder")], ["other"])
        self.assertEqual([utt.id for utt in convo.traverse("preorder")], ["other"])

    def traverse_no_timestamps(self):
        # sibling utterances without timestamps must not crash tree traversal
        convo = self.no_timestamp_corpus.get_conversation("0")
        bfs_traversal = [utt.id for utt in convo.traverse("bfs", as_utterance=True)]
        self.assertEqual(bfs_traversal, ["0", "1", "2"])

    def reindex_corpus(self):
        original_convo_meta = {
            k: v for k, v in self.corpus.get_conversation("0").meta.to_dict().items()
        }
        original_corpus_meta = {k: v for k, v in self.corpus.meta.to_dict().items()}
        new_convo_conversation_ids = ["1", "2", "3"]
        new_corpus = Corpus.reindex_conversations(self.corpus, new_convo_conversation_ids)
        # checking for correct number of conversations and utterances
        self.assertEqual(len(list(new_corpus.iter_conversations())), 3)
        self.assertEqual(len(list(new_corpus.iter_utterances())), 11)

        # checking that corpus and conversation metadata was preserved
        for convo in new_corpus.iter_conversations():
            self.assertEqual(convo.meta["original_convo_meta"], original_convo_meta)

        self.assertEqual(original_corpus_meta, new_corpus.meta)

    def reindex_corpus_2(self):
        new_convo_conversation_ids = ["1", "2", "3"]
        new_corpus = Corpus.reindex_conversations(
            self.corpus,
            new_convo_conversation_ids,
            preserve_convo_meta=False,
            preserve_corpus_meta=False,
        )
        # checking for correct number of conversations and utterances
        self.assertEqual(len(list(new_corpus.iter_conversations())), 3)
        self.assertEqual(len(list(new_corpus.iter_utterances())), 11)

        # checking that corpus and conversation metadata was preserved
        for convo in new_corpus.iter_conversations():
            self.assertEqual(convo.meta, dict())

        self.assertEqual(new_corpus.meta, dict())


class TestWithDB(CorpusTraversal):
    def setUp(self) -> None:
        self.corpus = reload_corpus_in_db_mode(construct_tree_corpus())
        self.multiple_convo_id_corpus = reload_corpus_in_db_mode(
            construct_multiple_convo_id_corpus()
        )
        self.nonexistent_reply_to_corpus = reload_corpus_in_db_mode(
            construct_nonexistent_reply_to_corpus()
        )
        self.no_timestamp_corpus = reload_corpus_in_db_mode(
            construct_no_timestamp_tree_corpus()
        )

    def test_broken_convos(self):
        self.broken_convos()

    def test_bfs_traversal(self):
        self.bfs_traversal()

    def test_dfs_traversal(self):
        self.dfs_traversal()

    def test_postorder_traversal(self):
        self.postorder_traversal()

    def test_preorder_traversal(self):
        self.preorder_traversal()

    def test_subtree(self):
        self.subtree()

    def test_conversation_id_to_leaf_paths(self):
        self.conversation_id_to_leaf_paths()

    def test_one_utt_convo(self):
        self.one_utt_convo()

    def test_traverse_no_timestamps(self):
        self.traverse_no_timestamps()

    def test_reindex_corpus(self):
        self.reindex_corpus()

    def test_reindex_corpus_2(self):
        self.reindex_corpus_2()


class TestWithMem(CorpusTraversal):
    def setUp(self) -> None:
        self.corpus = construct_tree_corpus()
        self.multiple_convo_id_corpus = construct_multiple_convo_id_corpus()
        self.nonexistent_reply_to_corpus = construct_nonexistent_reply_to_corpus()
        self.no_timestamp_corpus = construct_no_timestamp_tree_corpus()

    def test_broken_convos(self):
        self.broken_convos()

    def test_bfs_traversal(self):
        self.bfs_traversal()

    def test_dfs_traversal(self):
        self.dfs_traversal()

    def test_postorder_traversal(self):
        self.postorder_traversal()

    def test_preorder_traversal(self):
        self.preorder_traversal()

    def test_subtree(self):
        self.subtree()

    def test_conversation_id_to_leaf_paths(self):
        self.conversation_id_to_leaf_paths()

    def test_one_utt_convo(self):
        self.one_utt_convo()

    def test_traverse_no_timestamps(self):
        self.traverse_no_timestamps()

    def test_reindex_corpus(self):
        self.reindex_corpus()

    def test_reindex_corpus_2(self):
        self.reindex_corpus_2()


if __name__ == "__main__":
    unittest.main()
