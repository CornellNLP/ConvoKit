try:
    from .utteranceLikelihood import *
except (ImportError, ModuleNotFoundError) as e:
    if "torch" in str(e) or "datasets" in str(e) or "not currently installed" in str(e):
        raise ImportError(
            "UtteranceLikelihood requires ML dependencies. Run 'pip install convokit[llm]' to install them (or 'pip install convokit[llm-mac]' if you are on macOS)."
        ) from e
    else:
        raise
