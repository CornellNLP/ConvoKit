try:
    from .politenessStrategies import *
except ImportError:
    raise ImportError("spaCy is not installed, please install install spaCy with \"pip install convokit[spacy]\" to use this package.")