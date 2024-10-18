from sklearn.pipeline import Pipeline
from convokit.model import Utterance, Speaker


class ConvokitPipeline(Pipeline):
    """
    A pipeline of transformers. Builds on and inherits functionality from scikit-learn's Pipeline class.

    :param steps: a list of (name, transformer) tuples in the order that they are to be called.
    """

    def __init__(self, steps):
        Pipeline.__init__(self, steps)

    def _parse_param_steps(self, params):
        params_steps = {}
        for pname, pval in params.items():
            if "__" not in pname:
                continue
            step, param = pname.split("__", 1)
            if step in params_steps:
                params_steps[step][param] = pval
            else:
                params_steps[step] = {param: pval}
        return params_steps

    def transform(self, corpus, **params):
        params_steps = self._parse_param_steps(params)
        for name, transform in self.steps:
            if name in params_steps:
                corpus = transform.transform(corpus, **params_steps[name])
            else:
                corpus = transform.transform(corpus)
        return corpus

    def transform_utterance(self, utt, **params):
        """
        Computes attributes of an individual string or utterance using all of the transformers in the pipeline.

        :param utt: the utterance to compute attributes for.
        :return: the utterance, with new attributes.
        """
        params_steps = self._parse_param_steps(params)

        if isinstance(utt, str):
            utt = Utterance(text=utt, speaker=Speaker(id="speaker"))
        for name, transform in self.steps:
            if name in params_steps:
                utt = transform.transform_utterance(utt, **params_steps[name])
            else:
                utt = transform.transform_utterance(utt)
        return utt
