from .utteranceSimulator import *

try:
    from .utteranceSimulatorModel import UtteranceSimulatorModel
    from .unslothUtteranceSimulatorModel import *
except NotImplementedError as e:
    if "Unsloth GPU requirement not met" in str(e):
        print(
            "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
        )
    else:
        raise
