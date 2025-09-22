import warnings
from .forecaster import *
from .forecasterModel import *
from .cumulativeBoW import *
import sys

# Import CRAFT models if torch is available
if "torch" in sys.modules:
    from .CRAFTModel import *
    from .CRAFT import *

try:
    from .TransformerDecoderModel import *
except ImportError as e:
    if "Unsloth GPU requirement not met" in str(e):
        warnings.warn(
            "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
        )
    elif "Unsloth is not currently available on macOS" in str(e):
        warnings.warn(
            "TransformerDecoderModel: If you are a mac user, unsloth is currently not available on macOS. For other users, please use 'pip install convokit[llm]' to install LLM related dependencies."
        )
    elif "not currently installed" in str(e):
        warnings.warn(
            "TransformerDecoderModel: LLM dependencies are not currently installed. Run 'pip install convokit[llm]' to install them (or 'pip install convokit[llmmac]' for macOS users)."
        )
    elif "unsloth" in str(e).lower():
        warnings.warn(
            "TransformerDecoderModel: If you are a mac user, unsloth is currently not available on macOS. For other users, please use 'pip install convokit[llm]' to install LLM related dependencies."
        )
    else:
        warnings.warn(f"TransformerDecoderModel could not be imported: {e}")

try:
    from .TransformerEncoderModel import *
except (ImportError, ModuleNotFoundError) as e:
    if "not currently installed" in str(e):
        warnings.warn(
            "TransformerEncoderModel: LLM dependencies are not currently installed. Run 'pip install convokit[llm]' to install them (or 'pip install convokit[llmmac]' for macOS users)."
        )
    else:
        warnings.warn(f"TransformerEncoderModel could not be imported: {e}")

from .TransformerForecasterConfig import *
