import unittest
from convokit.model import Utterance, Speaker, Corpus
from convokit import download

class CorpusFromPandas(unittest.TestCase):

    def setUp(self) -> None:
        self.corpus = Corpus(download('subreddit-Cornell'))

    def test_reconstruction(self):
        """
        Test basic meta functions
        """

        utt_df = self.corpus.get_utterances_dataframe()
        convo_df = self.corpus.get_conversations_dataframe()
        speaker_df = self.corpus.get_speakers_dataframe()
        print(utt_df.columns)
        new_corpus = Corpus.from_pandas(speaker_df, utt_df, convo_df)
        self.corpus.print_summary_stats()
        new_corpus.print_summary_stats()

if __name__ == '__main__':
    unittest.main()
