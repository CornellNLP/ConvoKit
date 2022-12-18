import importlib.util
import sys

from .convokit_lm import *
from .language_model import *
from .surprise import *

if "kenlm" in sys.modules:
    from .kenlm import *
elif (spec := importlib.util.find_spec("kenlm")) is not None:
    module = importlib.util.module_from_spec(spec)
    sys.modules["kenlm"] = module
    spec.loader.exec_module(module)

    from .kenlm import *
