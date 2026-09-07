import unittest

from convokit.forecaster.forecaster import ContextTuple


class ContextTupleDocsTest(unittest.TestCase):
    def test_fields_have_descriptive_docstrings(self):
        self.assertEqual(
            {
                "context": ContextTuple.context.__doc__,
                "current_utterance": ContextTuple.current_utterance.__doc__,
                "future_context": ContextTuple.future_context.__doc__,
                "conversation_id": ContextTuple.conversation_id.__doc__,
            },
            {
                "context": "Chronological utterances up to and including the current utterance.",
                "current_utterance": "Most recent utterance when this context was generated.",
                "future_context": (
                    "Later utterances during training, or None when future context is unavailable."
                ),
                "conversation_id": "Identifier of the conversation that produced this context.",
            },
        )
