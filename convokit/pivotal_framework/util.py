from collections import namedtuple

DEFAULT_LABELER = "has_removed_comment"

ContextTuple = namedtuple(
    "ContextTuple", ["context", "current_utterance", "future_context", "conversation_id"]
)
