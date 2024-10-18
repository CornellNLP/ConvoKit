from typing import Dict, Optional

from convokit.util import warn
from .corpusComponent import CorpusComponent
from .speaker import Speaker


class Utterance(CorpusComponent):
    """Represents a single utterance in the dataset.

    :param id: the unique id of the utterance.
    :param speaker: the speaker giving the utterance.
    :param conversation_id: the id of the root utterance of the conversation.
    :param reply_to: id of the utterance this was a reply to.
    :param timestamp: timestamp of the utterance. Can be any
        comparable type.
    :param text: text of the utterance.

    :ivar id: the unique id of the utterance.
    :ivar speaker: the speaker giving the utterance.
    :ivar conversation_id: the id of the root utterance of the conversation.
    :ivar reply_to: id of the utterance this was a reply to.
    :ivar timestamp: timestamp of the utterance.
    :ivar text: text of the utterance.
    :ivar meta: A dictionary-like view object providing read-write access to
        utterance-level metadata.
    """

    def __init__(
        self,
        owner=None,
        id: Optional[str] = None,
        speaker: Optional[Speaker] = None,
        conversation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        timestamp: Optional[int] = None,
        text: str = "",
        meta: Optional[Dict] = None,
    ):
        # check arguments that have alternate naming due to backwards compatibility
        if speaker is None:
            raise ValueError("No Speaker found: Utterance must be initialized with a Speaker.")

        if conversation_id is not None and not isinstance(conversation_id, str):
            warn(
                "Utterance conversation_id must be a string: conversation_id of utterance with ID: {} "
                "has been casted to a string.".format(id)
            )
            conversation_id = str(conversation_id)
        if not isinstance(text, str):
            warn(
                "Utterance text must be a string: text of utterance with ID: {} "
                "has been casted to a string.".format(id)
            )
            text = "" if text is None else str(text)

        props = {
            "speaker_id": speaker.id,
            "conversation_id": conversation_id,
            "reply_to": reply_to,
            "timestamp": timestamp,
            "text": text,
        }
        super().__init__(obj_type="utterance", owner=owner, id=id, initial_data=props, meta=meta)
        self.speaker_ = speaker

    ############################################################################
    ## directly-accessible class properties (roughly equivalent to keys in the
    ## JSON, plus aliases for compatibility)
    ############################################################################

    def _get_speaker(self):
        return self.speaker_

    def _set_speaker(self, val):
        self.speaker_ = val
        self.set_data("speaker_id", self.speaker.id)

    speaker = property(_get_speaker, _set_speaker)

    def _get_conversation_id(self):
        return self.get_data("conversation_id")

    def _set_conversation_id(self, val):
        self.set_data("conversation_id", val)

    conversation_id = property(_get_conversation_id, _set_conversation_id)

    def _get_reply_to(self):
        return self.get_data("reply_to")

    def _set_reply_to(self, val):
        self.set_data("reply_to", val)

    reply_to = property(_get_reply_to, _set_reply_to)

    def _get_timestamp(self):
        return self.get_data("timestamp")

    def _set_timestamp(self, val):
        self.set_data("timestamp", val)

    timestamp = property(_get_timestamp, _set_timestamp)

    def _get_text(self):
        return self.get_data("text")

    def _set_text(self, val):
        self.set_data("text", val)

    text = property(_get_text, _set_text)

    ############################################################################
    ## end properties
    ############################################################################

    def get_conversation(self):
        """
        Get the Conversation (identified by Utterance.conversation_id) this Utterance belongs to

        :return: a Conversation object
        """
        return self.owner.get_conversation(self.conversation_id)

    def get_speaker(self):
        """
        Get the Speaker that made this Utterance.

        :return: a Speaker object
        """

        return self.speaker

    def to_dict(self):
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "reply_to": self.reply_to,
            "speaker": self.speaker,
            "timestamp": self.timestamp,
            "text": self.text,
            "vectors": self.vectors,
            "meta": self.meta if type(self.meta) == dict else self.meta.to_dict(),
        }

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Utterance):
            return False
        try:
            return (
                self.id == other.id
                and (
                    self.conversation_id is None
                    or other.conversation_id is None
                    or self.conversation_id == other.conversation_id
                )
                and self.reply_to == other.reply_to
                and self.speaker == other.speaker
                and self.timestamp == other.timestamp
                and self.text == other.text
            )
        except AttributeError:  # for backwards compatibility with wikiconv
            return self.__dict__ == other.__dict__

    def __str__(self):
        return (
            "Utterance(id: {}, conversation_id: {}, reply-to: {}, "
            "speaker: {}, timestamp: {}, text: {}, vectors: {}, meta: {})".format(
                repr(self.id),
                self.conversation_id,
                self.reply_to,
                self.speaker,
                self.timestamp,
                repr(self.text),
                self.vectors,
                self.meta,
            )
        )
