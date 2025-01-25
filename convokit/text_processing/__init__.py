from .textProcessor import *
from .textToArcs import *
from .textCleaner import TextCleaner
import warnings
try:
    from .textParser import * 
except Exception:
    class TextParser:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "spaCy is required to use TextParser. "
                "Please install it using `pip install spacy`."
            )
    warnings.warn("spaCy is not installed, textParser and textParser dependent subpackages are skipped.")