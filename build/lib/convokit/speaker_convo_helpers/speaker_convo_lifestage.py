from convokit.transformer import Transformer


class SpeakerConvoLifestage(Transformer):
    """
    Transformer that, for each speaker in a conversation, computes the lifestage of the speaker in that conversation. For instance, if lifestages are 20 conversations long, then the first 20 conversations a speaker participates in will be in lifestage 0, and the second 20 will be in lifestage 1.

    Assumes that `corpus.organize_speaker_convo_history` has already been called.

    :param lifestage_size: size of the lifestage
    :param output_field: name of speaker conversation attribute to output, defaults to "lifestage"
    """

    def __init__(self, lifestage_size, output_field="lifestage"):
        self.output_field = output_field
        self.lifestage_size = lifestage_size

    def transform(self, corpus):
        for speaker in corpus.iter_speakers():
            if "conversations" not in speaker.meta:
                continue
            for convo_id, convo in speaker.meta["conversations"].items():
                corpus.set_speaker_convo_info(
                    speaker.id,
                    convo_id,
                    self.output_field,
                    int(convo["idx"] // self.lifestage_size),
                )
        return corpus
