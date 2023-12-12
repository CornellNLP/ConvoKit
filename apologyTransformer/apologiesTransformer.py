import convokit
from convokit import Corpus, download
from convokit.transformer import Transformer
from inspect import signature
import string
import re

apology_list = ['sorry', 'apologize', 'apologies', 'oops', 'whoops', 'woops', 'forgive me', 'forgive my', 'excuse me', 'excuse my', 'my mistake', 'my bad']
first_person = ['i', 'me', 'my', 'myself', 'mine']
second_person = ['you', 'your', 'u', 'ur', 'yours', 'yourself', 'urself']
clarification = ['mean', 'meant', 'clarify','clear','clarification','explain','understand','confused','confusing','what','context','worded','wording','are you','do you','talking about','referring','rephrase','reword','intend','intent','term']
contradictory = ['but','however','while','although']
disagreement = ['wrong','incorrect','inaccurate','false','mistaken','error','bad','nonsensical','stupid','disagree','dumb','bullshit','bs','insufficient','hypocritical','break it']
agreement = ['right','correct','sense','true','accurate','case','work','agree']
negatives = ['no','not','don\'t','dont','doesn\'t','doesnt', 'isn\'t', 'isnt']
wrongdoing = ['regret','mistake','misunderstand','misunderstood','fault','offend','hurt','misread','misspoke','wrong','incorrect','accident','misconception','truly','genuine','sincere']
potential = ['for','if','because','that','about']
requests = ['could','would','can']

apology_pattern = r"\b(" + "|".join(re.escape(word) for word in apology_list) + r")\b"

clarify_pattern = r"\b(" + "|".join(re.escape(word) for word in clarification) + r")\b"
contradictory_pattern = fr"{apology_pattern}(.{{0,20}}(?:but|however|while|although))\b"
disagree_pattern = r"\b(" + "|".join(re.escape(word) for word in disagreement) + r")\b"
negatives_pattern = r"\b(" + "|".join(re.escape(word) for word in negatives) + r")\b"
agreement_pattern = r"\b(" + "|".join(re.escape(word) for word in agreement) + r")\b"
not_agree_pattern = fr"{negatives_pattern}.{{0,10}}{agreement_pattern}"
potential_pattern = fr"{apology_pattern}.{{0,3}}\b(" + "|".join(re.escape(word) for word in potential) + r")\b"
first_person_pattern = r"\b(" + "|".join(re.escape(word) for word in first_person) + r")\b"
second_person_pattern = r"\b(" + "|".join(re.escape(word) for word in second_person) + r")\b"
wrong_pattern = r"\b(" + "|".join(re.escape(word) for word in wrongdoing) + r")\b"
wrongdoing_pattern = fr"{first_person_pattern}.{{0,10}}{wrong_pattern}"
ask_pattern = r"\b(" + "|".join(re.escape(word) for word in requests) + r")\b"
requests_pattern = fr"({ask_pattern}.{{0,10}}{second_person_pattern})|please"

class ApologyLabeler(Transformer):
    """
    A transformer to label diffferent types of apologies in a corpus.

    :param 
    """

    def __init__(
        self,
        obj_type='utterance',
        output_field='apology_type',
        input_field=None,
        input_filter=None,
        verbosity=10000,
    ):
        if input_filter:
            if len(signature(input_filter).parameters) == 1:
                self.input_filter = lambda utt: input_filter(utt)
            else:
                self.input_filter = input_filter
        else:
            self.input_filter = lambda utt: True
        self.obj_type = obj_type
        self.input_field = input_field
        self.output_field = output_field
        self.verbosity = verbosity

    def _print_output(self, i):
        return (self.verbosity > 0) and (i > 0) and (i % self.verbosity == 0)

    def transform(self, corpus: Corpus) -> Corpus:

        if self.obj_type == 'utterance':
          total = len(list(corpus.iter_utterances()))

          for idx, utterance in enumerate(corpus.iter_utterances()):
              if self._print_output(idx):
                  print(f"%03d/%03d {self.obj_type} processed" % (idx, total))

              text = remove_quotes(utterance.text)
              text = text.lower()
              sentences = re.split(r'(?<=[.!?])\s+', text)

              apology = False
              apology_loc = 0
              for i, sentence in enumerate(sentences):
                apology_match = re.search(apology_pattern, sentence) #start index of match
                if apology_match:
                  apology_loc = apology_match.span()[0]
                  apology_sentence = sentence.strip()
                  next_sentence = " "
                  if (i != len(sentences)-1):
                    next_sentence = sentences[i+1].strip()

                  apology_segment = apology_sentence + next_sentence
                  apology = True

              if apology:

                  pattern_meta_mapping = [
                      (clarify_pattern, 'clarifying_apology'),
                      (potential_pattern, 'wrongdoing_apology'),
                      (wrongdoing_pattern, 'wrongdoing_apology'),
                      (contradictory_pattern, 'disagree_apology'),
                      (disagree_pattern, 'disagree_apology'),
                      (not_agree_pattern, 'disagree_apology'),
                      (requests_pattern, 'request_apology')
                  ]

                  closest_match = min(
                      [(re.search(pattern, apology_segment), meta) for pattern, meta in pattern_meta_mapping if re.search(pattern, apology_segment)],
                      key=lambda x: abs(x[0].start() - apology_loc),
                      default=None
                  )

                  if closest_match:
                      _, meta = closest_match
                      utterance.add_meta(self.output_field, meta)
                  else:
                      utterance.add_meta(self.output_field, 'other_apology')

              else:
                utterance.add_meta(self.output_field, 'no_apology')
