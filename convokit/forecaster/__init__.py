from .forecaster import *
from .forecasterModel import *
from .cumulativeBoW import *
import sys

# Import CRAFT models if torch is available
if "torch" in sys.modules:
    from .CRAFTModel import *
    from .CRAFT import *

# Import Transformer models with proper error handling
try:
    from .TransformerDecoderModel import *
except ImportError as e:
    if "Unsloth GPU requirement not met" in str(e):
        print(
            "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
        )
    elif "not currently installed" in str(e):
        print(
            "TransformerDecoderModel requires ML dependencies. Run 'pip install convokit[ml]' to install them."
        )
    else:
        raise

try:
    from .TransformerEncoderModel import *
except ImportError as e:
    if "not currently installed" in str(e):
        print(
            "TransformerEncoderModel requires ML dependencies. Run 'pip install convokit[ml]' to install them."
        )
    else:
        raise

from .TransformerForecasterConfig import *
