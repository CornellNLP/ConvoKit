import unittest
from convokit.model import Utterance, Speaker, Corpus


class CorpusIndexMeta(unittest.TestCase):
    def test_basic_functions(self):
        """
        Test basic meta functions
        """
        corpus1_db = Corpus(utterances=[
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob",
                      speaker=Speaker(id="bob")),
            Utterance(id="2",
                      text="this is a test",
                      speaker=Speaker(id="charlie")),
        ],
                            storage_type='db')

        corpus1_mem = Corpus(utterances=[
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob",
                      speaker=Speaker(id="bob")),
            Utterance(id="2",
                      text="this is a test",
                      speaker=Speaker(id="charlie")),
        ],
                             storage_type='mem')

        first_utt_mem = corpus1_mem.get_utterance("0")
        first_utt_mem.meta['hey_mem'] = 9

        first_utt_db = corpus1_db.get_utterance("0")
        first_utt_db.meta['hey_db'] = 9

        # correct class type stored
        self.assertEqual(
            first_utt_mem.storage.index.utterances_index['hey_mem'],
            first_utt_db.storage.index.utterances_index['hey_db'])

        # keyErrors result in None output
        with self.assertRaises(KeyError):
            first_utt_mem.meta['hey_db']

        with self.assertRaises(KeyError):
            first_utt_db.meta['hey_mem']

        # test that setting a custom get still works
        self.assertEqual(first_utt_mem.meta.get('nonexistent_key', {}), {})
        self.assertEqual(first_utt_db.meta.get('nonexistent_key', {}), {})

    def test_init_with_meta(self):
        corpus1_mem = Corpus(utterances=[
            Utterance(id="0",
                      text="hello world",
                      speaker=Speaker(id="alice", meta={'surname': 1.0}),
                      meta={'foo': 'bar'}),
            Utterance(id="1",
                      text="my name is bob",
                      speaker=Speaker(id="bob"),
                      meta={'foo': 'bar2'}),
            Utterance(id="2",
                      text="this is a test",
                      speaker=Speaker(id="charlie"),
                      meta={'hey': 'jude'}),
        ],
                             storage_type='mem')
        corpus1_db = Corpus(utterances=[
            Utterance(id="0",
                      text="hello world",
                      speaker=Speaker(id="alice", meta={'surname': 1.0}),
                      meta={'foo': 'bar'}),
            Utterance(id="1",
                      text="my name is bob",
                      speaker=Speaker(id="bob"),
                      meta={'foo': 'bar2'}),
            Utterance(id="2",
                      text="this is a test",
                      speaker=Speaker(id="charlie"),
                      meta={'hey': 'jude'}),
        ],
                            storage_type='db')

        corpus1_mem.get_conversation(None).meta['convo_meta'] = 1
        corpus1_db.get_conversation(None).meta['convo_meta'] = 1

        self.assertEqual(corpus1_mem.get_utterance("0"),
                         corpus1_db.get_utterance("0"))
        self.assertEqual(
            corpus1_mem.get_utterance("0").meta,
            corpus1_db.get_utterance("0").meta)
        self.assertEqual(corpus1_mem.get_utterance("1"),
                         corpus1_db.get_utterance("1"))
        self.assertEqual(
            corpus1_mem.get_utterance("1").meta,
            corpus1_db.get_utterance("1").meta)
        self.assertEqual(corpus1_mem.get_utterance("2"),
                         corpus1_db.get_utterance("2"))
        self.assertEqual(
            corpus1_mem.get_utterance("2").meta,
            corpus1_db.get_utterance("2").meta)

        self.assertEqual(corpus1_mem.get_conversation(None),
                         corpus1_db.get_conversation(None))
        self.assertEqual(
            corpus1_mem.get_conversation(None).meta,
            corpus1_db.get_conversation(None).meta)
        self.assertEqual(corpus1_mem.get_speaker("alice"),
                         corpus1_db.get_speaker("alice"))
        self.assertEqual(
            corpus1_mem.get_speaker("alice").meta,
            corpus1_db.get_speaker("alice").meta)
        self.assertEqual(corpus1_mem.get_speaker("charlie"),
                         corpus1_db.get_speaker("charlie"))
        self.assertEqual(
            corpus1_mem.get_speaker("charlie").meta,
            corpus1_db.get_speaker("charlie").meta)

    def test_key_insertion_deletion(self):
        corpus1_mem = Corpus(utterances=[
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob",
                      speaker=Speaker(id="bob")),
            Utterance(id="2",
                      text="this is a test",
                      speaker=Speaker(id="charlie")),
        ],
                             storage_type='mem')
        corpus1_db = Corpus(utterances=[
            Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
            Utterance(id="1", text="my name is bob",
                      speaker=Speaker(id="bob")),
            Utterance(id="2",
                      text="this is a test",
                      speaker=Speaker(id="charlie")),
        ],
                            storage_type='db')

        corpus1_mem.get_utterance("0").meta['foo'] = 'bar'
        corpus1_mem.get_utterance("1").meta['foo'] = 'bar2'
        corpus1_mem.get_utterance("2").meta['hey'] = 'jude'
        corpus1_mem.get_conversation(None).meta['convo_meta'] = 1
        corpus1_mem.get_speaker("alice").meta['surname'] = 1.0

        corpus1_db.get_utterance("0").meta['foo'] = 'bar'
        corpus1_db.get_utterance("1").meta['foo'] = 'bar2'
        corpus1_db.get_utterance("2").meta['hey'] = 'jude'
        corpus1_db.get_conversation(None).meta['convo_meta'] = 1
        corpus1_db.get_speaker("alice").meta['surname'] = 1.0

        self.assertEqual(corpus1_mem.get_utterance("0"),
                         corpus1_db.get_utterance("0"))
        self.assertEqual(
            corpus1_mem.get_utterance("0").meta,
            corpus1_db.get_utterance("0").meta)
        self.assertEqual(corpus1_mem.get_utterance("1"),
                         corpus1_db.get_utterance("1"))
        self.assertEqual(
            corpus1_mem.get_utterance("1").meta,
            corpus1_db.get_utterance("1").meta)
        self.assertEqual(corpus1_mem.get_utterance("2"),
                         corpus1_db.get_utterance("2"))
        self.assertEqual(
            corpus1_mem.get_utterance("2").meta,
            corpus1_db.get_utterance("2").meta)
        self.assertEqual(corpus1_mem.get_conversation(None),
                         corpus1_db.get_conversation(None))
        self.assertEqual(
            corpus1_mem.get_conversation(None).meta,
            corpus1_db.get_conversation(None).meta)
        self.assertEqual(corpus1_mem.get_speaker("alice"),
                         corpus1_db.get_speaker("alice"))
        self.assertEqual(
            corpus1_mem.get_speaker("alice").meta,
            corpus1_db.get_speaker("alice").meta)
        self.assertEqual(corpus1_mem.get_speaker("charlie"),
                         corpus1_db.get_speaker("charlie"))
        self.assertEqual(
            corpus1_mem.get_speaker("charlie").meta,
            corpus1_db.get_speaker("charlie").meta)


if __name__ == '__main__':
    unittest.main()