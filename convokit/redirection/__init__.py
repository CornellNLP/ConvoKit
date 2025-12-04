try:
    from .redirection import *
except (ImportError, ModuleNotFoundError) as e:
    if "torch" in str(e) or "not currently installed" in str(e):
        raise ImportError(
            "Redirection module requires ML dependencies. Run 'pip install convokit[llm]' to install them."
        ) from e
    else:
        raise

try:
    from .likelihoodModel import *
except (ImportError, ModuleNotFoundError) as e:
    if "not currently installed" in str(e):
        raise ImportError(
            "LikelihoodModel requires ML dependencies. Run 'pip install convokit[llm]' to install them."
        ) from e
    else:
        raise
