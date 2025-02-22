import warnings
import sys
import types


class LazyModule(types.ModuleType):
    def __init__(self, module_name):
        self._module_name = module_name

    def __getattr__(self, attr):
        raise ImportError(
            f'{self._module_name} requires spaCy, which is not installed. Please install it with "pip install convokit[spacy]"'
        )


try:
    from .model import *
    from .util import *
    from .coordination import *
    from .transformer import *
    from .convokitPipeline import *
    from .hyperconvo import *
    from .speakerConvoDiversity import *
    from .classifier import *
    from .ranker import *
    from .forecaster import *
    from .fighting_words import *
    from .paired_prediction import *
    from .bag_of_words import *
    from .surprise import *
    from .convokitConfig import *

    try:
        import spacy
        from .politenessStrategies import *
        from .text_processing import *
        from .phrasing_motifs import *
        from .expected_context_framework import *
        from .prompt_types import *
    except ImportError:
        warnings.warn("spaCy is not installed, skipping spaCy-dependent modules.")
        sys.modules["convokit.politenessStrategies"] = LazyModule("politenessStrategies")
        sys.modules["convokit.text_processing"] = LazyModule("text_processing")
        sys.modules["convokit.phrasing_motifs"] = LazyModule("phrasing_motifs")
        sys.modules["convokit.expected_context_framework"] = LazyModule(
            "expected_context_framework"
        )
        sys.modules["convokit.prompt_types"] = LazyModule("prompt_types")

except Exception as e:
    print(f"An error occurred: {e}")
    warnings.warn(
        "If you are using ConvoKit with Google Colab, incorrect versions of some packages (ex. scipy) may be imported while runtime start. To fix the issue, restart the session and run all codes again. Thank you!"
    )
