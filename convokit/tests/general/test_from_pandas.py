import unittest
from convokit.model import Utterance, Speaker, Corpus
from convokit import download, StorageManager


class CorpusFromPandas(unittest.TestCase):
    def setUp(self) -> None:
        print('default: ', StorageManager.default_storage_mode())
        StorageManager.purge_db()
        self.corpus = Corpus(download('subreddit-hey'), storage_type='mem')
        if StorageManager.default_storage_mode() == 'db':
            self.corpus = self.corpus.copy_as('db', corpus_id='subreddit-hey')

        self.utt_df = self.corpus.get_utterances_dataframe()
        self.convo_df = self.corpus.get_conversations_dataframe()
        self.speaker_df = self.corpus.get_speakers_dataframe()
        self.new_corpus = Corpus.from_pandas(self.utt_df, self.speaker_df,
                                             self.convo_df)

    def test_reconstruction_stats(self):
        """
        Test that reconstructing the Corpus from outputted dataframes results in the same number of corpus components
        """
        self.assertEqual(set(self.new_corpus.iter_utterances()),
                         set(self.corpus.iter_utterances()))
        self.assertEqual(set(self.new_corpus.iter_speakers()),
                         set(self.corpus.iter_speakers()))
        self.assertEqual(set(self.new_corpus.iter_conversations()),
                         set(self.corpus.iter_conversations()))

    def _test_reconstruction_metadata(self, new_corpus):
        assert set(self.corpus.random_utterance().meta) == set(
            new_corpus.random_utterance().meta)
        assert set(self.corpus.random_conversation().meta) == set(
            new_corpus.random_conversation().meta)
        assert set(self.corpus.random_speaker().meta) == set(
            new_corpus.random_speaker().meta)

    def _test_convo_reconstruction(self, new_corpus):
        for convo in new_corpus.iter_conversations():
            assert convo.check_integrity(verbose=False)

    def test_full_reconstruction(self):
        self._test_reconstruction_metadata(self.new_corpus)
        self._test_convo_reconstruction(self.new_corpus)

    def test_no_speakers_df(self):
        test_corpus = Corpus.from_pandas(utterances_df=self.utt_df,
                                         speakers_df=None,
                                         conversations_df=self.convo_df)
        self._test_convo_reconstruction(test_corpus)

    def test_no_convos_df(self):
        test_corpus = Corpus.from_pandas(utterances_df=self.utt_df,
                                         speakers_df=self.speaker_df,
                                         conversations_df=None)
        self._test_convo_reconstruction(test_corpus)

    def test_no_speaker_convo_dfs(self):
        test_corpus = Corpus.from_pandas(utterances_df=self.utt_df)
        self._test_convo_reconstruction(test_corpus)


if __name__ == '__main__':
    unittest.main()
