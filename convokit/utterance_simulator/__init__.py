from .utteranceSimulator import *

try:
    from .utteranceSimulatorModel import UtteranceSimulatorModel
    from .unslothUtteranceSimulatorModel import *
except ImportError as e:
    if "Unsloth GPU requirement not met" in str(e):
        print(
            "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
        )
    elif "not currently installed" in str(e):
        print(
            "UnslothUtteranceSimulatorModel requires ML dependencies. Run 'pip install convokit[ml]' to install them."
        )
    else:
        raise
