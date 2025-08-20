import warnings
from .utteranceSimulator import *
from .utteranceSimulatorModel import UtteranceSimulatorModel

try:
    from .unslothUtteranceSimulatorModel import *
except ImportError as e:
    if "Unsloth GPU requirement not met" in str(e):
        warnings.warn(
            "Error from Unsloth: NotImplementedError: Unsloth currently only works on NVIDIA GPUs and Intel GPUs."
        )
    elif "Unsloth is not currently available on macOS" in str(e):
        warnings.warn(
            "UnslothUtteranceSimulatorModel: If you are a mac user, unsloth is currently not available on macOS. For other users, please use 'pip install convokit[llm]' to install LLM related dependencies."
        )
    elif "not currently installed" in str(e):
        warnings.warn(
            "UnslothUtteranceSimulatorModel: LLM dependencies are not currently installed. Run 'pip install convokit[llm]' to install them (or 'pip install convokit[llmmac]' for macOS users)."
        )
    elif "unsloth" in str(e).lower():
        warnings.warn(
            "UnslothUtteranceSimulatorModel: If you are a mac user, unsloth is currently not available on macOS. For other users, please use 'pip install convokit[llm]' to install LLM related dependencies."
        )
    else:
        warnings.warn(f"UnslothUtteranceSimulatorModel could not be imported: {e}")
