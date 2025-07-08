from .forecaster import *
from .forecasterModel import *
from .cumulativeBoW import *
import sys

if "torch" in sys.modules:
    from .CRAFTModel import *
    from .CRAFT import *
    from .TransformerEncoderModel import *
    from .TransformerForecasterConfig import *

    try:
        from .TransformerDecoderModel import *
    except ImportError as e:
        if "Unsloth GPU requirement not met" in str(e):
            print(
                "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
            )
        else:
            raise
