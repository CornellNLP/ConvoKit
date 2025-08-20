import warnings
from .redirection import *

try:
    from .likelihoodModel import *
except ImportError as e:
    if "not currently installed" in str(e):
        warnings.warn(
            "LikelihoodModel: LLM dependencies are not currently installed. Run 'pip install convokit[llm]' to install them (or 'pip install convokit[llmmac]' for macOS users)."
        )
    else:
        raise
