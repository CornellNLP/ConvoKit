import warnings

from .model import *
from .util import *
from .coordination import *
from .politenessStrategies import *
from .transformer import *
from .convokitPipeline import *
from .hyperconvo import *
from .speakerConvoDiversity import *
from .text_processing import *
from .phrasing_motifs import *
from .prompt_types import *
from .classifier.classifier import *
from .ranker import *
from .fighting_words import *
from .paired_prediction import *
from .bag_of_words import *
from .expected_context_framework import *
from .surprise import *
from .convokitConfig import *

# Modules with optional dependencies - import with error handling
try:
    from .forecaster import *
except ImportError as e:
    if "not currently installed" not in str(e):
        warnings.warn(f"Forecaster module could not be imported: {e}")

try:
    from .pivotal_framework import *
except ImportError as e:
    if "not currently installed" not in str(e):
        warnings.warn(f"Pivotal framework module could not be imported: {e}")

try:
    from .utterance_simulator import *
except ImportError as e:
    if "not currently installed" not in str(e):
        warnings.warn(f"Utterance simulator module could not be imported: {e}")

try:
    from .redirection import *
except ImportError as e:
    if "not currently installed" not in str(e):
        warnings.warn(f"Redirection module could not be imported: {e}")
