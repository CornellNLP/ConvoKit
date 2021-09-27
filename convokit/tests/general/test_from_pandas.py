import unittest
from convokit.model import Utterance, Speaker, Corpus
from convokit import download

class CorpusFromPandas(unittest.TestCase):

    def setUp(self) -> None:
        self.corpus = Corpus(download('subreddit-hey'))
        utt_df = self.corpus.get_utterances_dataframe()
        convo_df = self.corpus.get_conversations_dataframe()
        speaker_df = self.corpus.get_speakers_dataframe()
        self.new_corpus = Corpus.from_pandas(speaker_df, utt_df, convo_df)

    def test_reconstruction_stats(self):
        """
        Test that reconstructing the Corpus from outputted dataframes results in the same number of corpus components
        """
        assert len(self.new_corpus.speakers) == len(self.corpus.speakers)
        assert len(self.new_corpus.conversations) == len(self.corpus.conversations)
        assert len(self.new_corpus.utterances) == len(self.corpus.utterances)

    def test_reconstruction_metadata(self):
        assert set(self.corpus.random_utterance().meta) == set(self.new_corpus.random_utterance().meta)
        assert set(self.corpus.random_conversation().meta) == set(self.new_corpus.random_conversation().meta)
        assert set(self.corpus.random_speaker().meta) == set(self.new_corpus.random_speaker().meta)

    def test_convo_reconstruction(self):
        for convo in self.new_corpus.iter_conversations():
            assert convo.check_integrity(verbose=False)

if __name__ == '__main__':
    unittest.main()
