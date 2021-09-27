import unittest
from convokit.model import Utterance, Speaker, Corpus
from convokit import download

class CorpusFromPandas(unittest.TestCase):

    def setUp(self) -> None:
        self.corpus = Corpus(download('subreddit-Cornell'))

    def test_reconstruction(self):
        """
        Test that reconstructing the Corpus from outputted dataframes results in the same number of corpus components
        """

        utt_df = self.corpus.get_utterances_dataframe()
        convo_df = self.corpus.get_conversations_dataframe()
        speaker_df = self.corpus.get_speakers_dataframe()
        new_corpus = Corpus.from_pandas(speaker_df, utt_df, convo_df)
        assert len(new_corpus.speakers) == len(self.corpus.speakers)
        assert len(new_corpus.conversations) == len(self.corpus.conversations)
        assert len(new_corpus.utterances) == len(self.corpus.utterances)

if __name__ == '__main__':
    unittest.main()
